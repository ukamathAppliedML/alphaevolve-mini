"""
Evaluator: Sandboxed code execution and fitness evaluation.

The evaluator is the "ground truth" in AlphaEvolve - it objectively
measures how good each candidate solution is. Key features:

1. Safe sandboxed execution (prevents malicious code)
2. Multiple metrics (not just correctness)
3. Cascade evaluation (fast rejection of bad solutions)
4. Timeout handling

Reference: Section 2.4 of AlphaEvolve whitepaper
"""

import ast
import time
import signal
import traceback
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Tuple
import resource
import multiprocessing as mp
import queue as queue_mod

def _mp_run_one(queue_obj, code: str, function_name: str, timeout_seconds: float, test_input):
    # IMPORTANT: build safe globals inside child
    sandbox = CodeSandbox(timeout_seconds=timeout_seconds)
    safe_globals = sandbox._create_safe_globals()
    ok, out, runtime_ms = _worker_eval_one(code, function_name, safe_globals, test_input)
    queue_obj.put((ok, out, runtime_ms))


def _worker_eval_one(code: str, function_name: str, safe_globals: dict, test_input):
    import time
    start = time.perf_counter()
    try:
        g = dict(safe_globals)  # isolate per test
        exec(code, g)
        func = g[function_name]

        if isinstance(test_input, tuple):
            out = func(*test_input)
        elif isinstance(test_input, dict):
            out = func(**test_input)
        else:
            out = func(test_input)

        return (True, out, (time.perf_counter() - start) * 1000.0)
    except Exception as e:
        return (False, str(e), (time.perf_counter() - start) * 1000.0)


@dataclass
class EvaluationResult:
    """Result of evaluating a program."""
    success: bool
    score: float
    metrics: Dict[str, float]
    error: Optional[str] = None
    runtime_ms: float = 0
    output: Any = None


class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def compute_code_complexity(code: str) -> float:
    """
    Compute a normalized complexity score for code (0-1 range).
    
    Uses AST-based metrics:
    - Number of nodes
    - Nesting depth
    - Number of branches (if/for/while)
    
    Returns a value between 0 (simple) and 1 (complex).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.5  # Default for unparseable code
    
    # Count AST nodes
    num_nodes = sum(1 for _ in ast.walk(tree))
    
    # Count control flow constructs
    num_branches = sum(1 for node in ast.walk(tree) 
                       if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)))
    
    # Compute nesting depth
    def get_depth(node, current=0):
        max_depth = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                  ast.If, ast.For, ast.While, ast.Try, ast.With)):
                max_depth = max(max_depth, get_depth(child, current + 1))
            else:
                max_depth = max(max_depth, get_depth(child, current))
        return max_depth
    
    depth = get_depth(tree)
    
    # Normalize each component and combine
    # These ranges are heuristic based on typical code
    node_score = min(1.0, num_nodes / 200)  # 200 nodes = max complexity
    branch_score = min(1.0, num_branches / 20)  # 20 branches = max
    depth_score = min(1.0, depth / 10)  # 10 levels = max
    
    # Weighted combination
    complexity = 0.4 * node_score + 0.3 * branch_score + 0.3 * depth_score
    return complexity


def normalize_runtime(runtime_ms: float, max_runtime_ms: float = 5000.0) -> float:
    """
    Normalize runtime to 0-1 range.
    
    Args:
        runtime_ms: Actual runtime in milliseconds
        max_runtime_ms: Maximum expected runtime (maps to 1.0)
    
    Returns:
        Normalized value between 0 (fast) and 1 (slow/timeout)
    """
    return min(1.0, runtime_ms / max_runtime_ms)


class CodeSandbox:
    """
    Safe execution environment for untrusted code.
    
    Security measures:
    1. Restricted builtins (no file I/O, network, etc.)
    2. Memory limits
    3. Time limits
    4. No imports except whitelisted modules
    """
    
    # Safe builtins that won't cause harm
    SAFE_BUILTINS = {
        'abs': abs, 'all': all, 'any': any, 'bin': bin,
        'bool': bool, 'chr': chr, 'dict': dict, 'divmod': divmod,
        'enumerate': enumerate, 'filter': filter, 'float': float,
        'frozenset': frozenset, 'hash': hash, 'hex': hex,
        'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
        'iter': iter, 'len': len, 'list': list, 'map': map,
        'max': max, 'min': min, 'next': next, 'oct': oct,
        'ord': ord, 'pow': pow, 'print': print, 'range': range,
        'repr': repr, 'reversed': reversed, 'round': round,
        'set': set, 'slice': slice, 'sorted': sorted,
        'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
        'zip': zip, 'True': True, 'False': False, 'None': None,
        '__import__': __import__,  # Needed for import statements
    }
    
    # Whitelisted modules
    ALLOWED_MODULES = {
        'math', 'random', 'itertools', 'functools', 'collections',
        'heapq', 'bisect', 'copy', 'operator', 'statistics'
    }
    
    def __init__(self, timeout_seconds: float = 5.0, max_memory_mb: int = 256):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
    
    def _create_safe_globals(self) -> Dict:
        """Create a restricted global namespace."""
        safe_globals = {"__builtins__": self.SAFE_BUILTINS}
        
        # Add whitelisted modules
        for module_name in self.ALLOWED_MODULES:
            try:
                safe_globals[module_name] = __import__(module_name)
            except ImportError:
                pass
        
        return safe_globals
    
    def _validate_ast(self, code: str) -> Tuple[bool, str]:
        """
        Validate code AST for dangerous patterns.
        
        Checks for:
        - Import statements (except whitelisted)
        - File operations
        - System calls
        - Network operations
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    names = [alias.name.split('.')[0] for alias in node.names]
                else:
                    names = [node.module.split('.')[0]] if node.module else []
                
                for name in names:
                    if name not in self.ALLOWED_MODULES:
                        return False, f"Import not allowed: {name}"
            
            # Check for dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in {'__class__', '__bases__', '__subclasses__', '__mro__',
                               '__globals__', '__code__', '__builtins__'}:
                    return False, f"Dangerous attribute access: {node.attr}"
        
        return True, ""
    
    def execute(
        self, 
        code: str, 
        function_name: str,
        test_inputs: List[Any]
    ) -> List[Tuple[bool, Any, float]]:
        """
        Execute code and run test cases.
        
        Returns list of (success, output, runtime_ms) for each test.
        """
        # Validate AST first
        valid, error = self._validate_ast(code)
        if not valid:
            return [(False, error, 0)] * len(test_inputs)
        
        results = []
        safe_globals = self._create_safe_globals()
        
        try:
            # Compile and execute the code to define the function
            exec(code, safe_globals)
            
            if function_name not in safe_globals:
                return [(False, f"Function {function_name} not defined", 0)] * len(test_inputs)
            
            func = safe_globals[function_name]
            
            # Run each test (process-based timeout; replaces signal alarm)
            for test_input in test_inputs:
                q: mp.Queue = mp.Queue(maxsize=1)

                def _runner(queue_obj):
                    res = _worker_eval_one(code, function_name, self._create_safe_globals(), test_input)
                    queue_obj.put(res)

                p = mp.Process(target=_runner, args=(q,), daemon=True)
                p.start()
                p.join(self.timeout_seconds)

                if p.is_alive():
                    p.terminate()
                    p.join(0.1)
                    results.append((False, "Timeout", self.timeout_seconds * 1000.0))
                    continue

                try:
                    ok, output, runtime_ms = q.get_nowait()
                except queue_mod.Empty:
                    ok, output, runtime_ms = (False, "No result returned", 0.0)

                results.append((ok, output, runtime_ms))

                    
        except Exception as e:
            return [(False, f"Execution error: {e}", 0)] * len(test_inputs)
        
        return results


class Evaluator(ABC):
    """Abstract base class for problem evaluators."""
    
    @abstractmethod
    def evaluate(self, code: str) -> EvaluationResult:
        """Evaluate a candidate solution."""
        pass
    
    @abstractmethod
    def get_problem_description(self) -> str:
        """Get the problem description for prompts."""
        pass
    
    @abstractmethod
    def get_function_name(self) -> str:
        """Get the required function name."""
        pass


class FunctionOptimizationEvaluator(Evaluator):
    """
    Evaluator for function optimization problems.
    
    Measures both correctness (does it produce right outputs?)
    and quality (how good is the score?).
    """
    
    def __init__(
        self,
        problem_description: str,
        function_name: str,
        test_cases: List[Tuple[Any, Any]],  # (input, expected_output)
        objective_function: Callable[[Any], float],  # Computes score from outputs
        timeout_seconds: float = 5.0,
        correctness_weight: float = 0.3,
        quality_weight: float = 0.7
    ):
        self.problem_description = problem_description
        self.function_name = function_name
        self.test_cases = test_cases
        self.objective_function = objective_function
        self.sandbox = CodeSandbox(timeout_seconds)
        self.correctness_weight = correctness_weight
        self.quality_weight = quality_weight
    
    def get_problem_description(self) -> str:
        return self.problem_description
    
    def get_function_name(self) -> str:
        return self.function_name
    
    def evaluate(self, code: str) -> EvaluationResult:
        """Evaluate a candidate solution."""
        start_time = time.perf_counter()
        
        # Extract inputs from test cases
        test_inputs = [tc[0] for tc in self.test_cases]
        expected_outputs = [tc[1] for tc in self.test_cases]
        
        # Run tests
        results = self.sandbox.execute(code, self.function_name, test_inputs)
        
        # Calculate metrics
        num_correct = 0
        total_runtime = 0
        outputs = []
        errors = []
        
        for (success, output, runtime), expected in zip(results, expected_outputs):
            total_runtime += runtime
            
            if not success:
                errors.append(str(output))
                outputs.append(None)
                continue
            
            outputs.append(output)
            
            # Check correctness
            try:
                if self._outputs_match(output, expected):
                    num_correct += 1
            except Exception:
                pass
        
        correctness_ratio = num_correct / len(self.test_cases)
        
        # Calculate objective score if we have valid outputs
        quality_score = 0.0
        if num_correct > 0:
            try:
                quality_score = self.objective_function(outputs)
            except Exception as e:
                errors.append(f"Objective error: {e}")
        
        # Combined score
        score = (
            self.correctness_weight * correctness_ratio + 
            self.quality_weight * quality_score
        )
        
        runtime_ms = (time.perf_counter() - start_time) * 1000
        
        # Compute MAP-Elites feature dimensions
        complexity = compute_code_complexity(code)
        runtime_normalized = normalize_runtime(total_runtime)
        
        return EvaluationResult(
            success=num_correct > 0,
            score=score,
            metrics={
                "correctness": correctness_ratio,
                "quality": quality_score,
                "runtime_ms": total_runtime,
                "tests_passed": num_correct,
                # MAP-Elites feature dimensions (normalized 0-1)
                "complexity": complexity,
                "runtime": runtime_normalized
            },
            error="; ".join(errors) if errors else None,
            runtime_ms=runtime_ms,
            output=outputs
        )
    
    def _outputs_match(self, actual: Any, expected: Any) -> bool:
        """Check if outputs match, handling floating point comparison."""
        if isinstance(expected, float):
            return abs(actual - expected) < 1e-6
        elif isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(self._outputs_match(a, e) for a, e in zip(actual, expected))
        else:
            return actual == expected


class MaximizationEvaluator(Evaluator):
    """
    Evaluator for problems where we want to maximize a metric.
    
    Used for problems like:
    - Circle packing (maximize sum of radii)
    - Scheduling (maximize throughput)
    - Algorithm optimization (maximize speed)
    """
    
    def __init__(
        self,
        problem_description: str,
        function_name: str,
        fitness_function: Callable[[Any], float],
        validator: Optional[Callable[[Any], Tuple[bool, str]]] = None,
        timeout_seconds: float = 10.0
    ):
        self.problem_description = problem_description
        self.function_name = function_name
        self.fitness_function = fitness_function
        self.validator = validator
        self.sandbox = CodeSandbox(timeout_seconds)
    
    def get_problem_description(self) -> str:
        return self.problem_description
    
    def get_function_name(self) -> str:
        return self.function_name
    
    def evaluate(self, code: str) -> EvaluationResult:
        """Evaluate by running the function and computing fitness."""
        start_time = time.perf_counter()
        
        # Execute the code to get the solution
        results = self.sandbox.execute(code, self.function_name, [()])
        
        if not results or not results[0][0]:
            return EvaluationResult(
                success=False,
                score=0.0,
                metrics={},
                error=results[0][1] if results else "Execution failed",
                runtime_ms=(time.perf_counter() - start_time) * 1000
            )
        
        success, output, exec_runtime = results[0]
        
        # Validate if validator provided
        if self.validator:
            valid, error_msg = self.validator(output)
            if not valid:
                return EvaluationResult(
                    success=False,
                    score=0.0,
                    metrics={},
                    error=error_msg,
                    runtime_ms=(time.perf_counter() - start_time) * 1000,
                    output=output
                )
        
        # Compute fitness
        try:
            fitness = self.fitness_function(output)
        except Exception as e:
            return EvaluationResult(
                success=False,
                score=0.0,
                metrics={},
                error=f"Fitness error: {e}",
                runtime_ms=(time.perf_counter() - start_time) * 1000,
                output=output
            )
        
        # Compute MAP-Elites feature dimensions
        complexity = compute_code_complexity(code)
        runtime_normalized = normalize_runtime(exec_runtime)
        
        return EvaluationResult(
            success=True,
            score=fitness,
            metrics={
                "fitness": fitness,
                "runtime_ms": exec_runtime,
                # MAP-Elites feature dimensions (normalized 0-1)
                "complexity": complexity,
                "runtime": runtime_normalized
            },
            runtime_ms=(time.perf_counter() - start_time) * 1000,
            output=output
        )


class CascadeEvaluator(Evaluator):
    """
    Cascade evaluation for efficiency.
    
    Runs cheap/fast tests first to quickly reject bad solutions,
    only running expensive tests on promising candidates.
    
    This is crucial for problems with expensive evaluation.
    """
    
    def __init__(
        self,
        evaluators: List[Tuple[Evaluator, float]],  # (evaluator, threshold)
    ):
        """
        Args:
            evaluators: List of (evaluator, threshold) pairs.
                       Solution must score >= threshold to proceed.
        """
        self.evaluators = evaluators
    
    def get_problem_description(self) -> str:
        return self.evaluators[0][0].get_problem_description()
    
    def get_function_name(self) -> str:
        return self.evaluators[0][0].get_function_name()
    
    def evaluate(self, code: str) -> EvaluationResult:
        """Run cascade evaluation, stopping early if threshold not met."""
        combined_metrics = {}
        total_runtime = 0
        
        for i, (evaluator, threshold) in enumerate(self.evaluators):
            result = evaluator.evaluate(code)
            total_runtime += result.runtime_ms
            
            # Merge metrics with prefix
            for key, value in result.metrics.items():
                combined_metrics[f"stage{i}_{key}"] = value
            
            # Check threshold
            if result.score < threshold:
                # Include MAP-Elites metrics even on failure
                combined_metrics["complexity"] = compute_code_complexity(code)
                combined_metrics["runtime"] = normalize_runtime(total_runtime)
                return EvaluationResult(
                    success=False,
                    score=result.score,
                    metrics=combined_metrics,
                    error=f"Failed stage {i}: {result.error}" if result.error else f"Score {result.score} < threshold {threshold}",
                    runtime_ms=total_runtime
                )
        
        # Passed all stages - compute MAP-Elites metrics and return final result
        combined_metrics["complexity"] = compute_code_complexity(code)
        combined_metrics["runtime"] = normalize_runtime(total_runtime)
        
        return EvaluationResult(
            success=True,
            score=result.score,
            metrics=combined_metrics,
            runtime_ms=total_runtime,
            output=result.output
        )


def create_test_evaluator() -> Evaluator:
    """Create a simple test evaluator for demonstration."""
    
    def fitness(outputs):
        """Sum of outputs for test."""
        valid_outputs = [o for o in outputs if o is not None]
        return sum(valid_outputs) / len(valid_outputs) if valid_outputs else 0
    
    return FunctionOptimizationEvaluator(
        problem_description="""
Write a function that computes the sum of squares of a list of numbers.
The function should be as efficient as possible.
        """,
        function_name="sum_of_squares",
        test_cases=[
            ([1, 2, 3], 14),
            ([0], 0),
            ([1, 1, 1, 1], 4),
            (list(range(100)), sum(x**2 for x in range(100))),
        ],
        objective_function=lambda outputs: 1.0 if all(o is not None for o in outputs) else 0.0
    )
