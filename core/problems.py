"""
Example Problems for AlphaEvolve.

These are classic optimization problems used in the AlphaEvolve paper
and other evolutionary algorithm research.

1. Circle Packing: Pack circles to maximize sum of radii
2. Bin Packing: Minimize bins needed for items
3. Sorting Network: Find minimal comparison network
4. Function Optimization: Evolve mathematical functions
"""

import math
from typing import List, Tuple, Optional, Callable, Any
from .evaluator import Evaluator, MaximizationEvaluator, FunctionOptimizationEvaluator


# =============================================================================
# CIRCLE PACKING PROBLEM
# =============================================================================
# 
# This is the problem DeepMind highlighted in the AlphaEvolve paper.
# Given N circles of varying sizes, pack them into a unit square
# to maximize the sum of their radii.
#
# Reference: AlphaEvolve whitepaper, Section 3.1
# =============================================================================

def validate_circle_packing(
    circles: List[Tuple[float, float, float]], 
    max_circles: int = 100
) -> Tuple[bool, str]:
    """
    Validate a circle packing solution.
    
    Each circle is (x, y, r) - center coordinates and radius.
    All circles must:
    1. Fit within the unit square [0,1] x [0,1]
    2. Not overlap with each other
    3. Not exceed max_circles (to prevent O(n²) blowup with 200+ circles)
    """
    if not circles:
        return False, "Empty solution"
    
    # Prevent O(n²) blowup from LLM generating 200+ tiny circles
    if len(circles) > max_circles:
        return False, f"Too many circles ({len(circles)} > {max_circles})"
    
    for i, (x, y, r) in enumerate(circles):
        # Check bounds (circle must fit in unit square)
        if x - r < -1e-9 or x + r > 1 + 1e-9:
            return False, f"Circle {i} exceeds x bounds"
        if y - r < -1e-9 or y + r > 1 + 1e-9:
            return False, f"Circle {i} exceeds y bounds"
        if r <= 0:
            return False, f"Circle {i} has non-positive radius"
        
        # Check non-overlap with other circles
        for j, (x2, y2, r2) in enumerate(circles[i+1:], i+1):
            dist = math.sqrt((x - x2)**2 + (y - y2)**2)
            if dist < r + r2 - 1e-9:  # Small tolerance for numerical error
                return False, f"Circles {i} and {j} overlap"
    
    return True, ""


def circle_packing_fitness(circles: List[Tuple[float, float, float]]) -> float:
    """Fitness = sum of radii."""
    return sum(r for x, y, r in circles)


def create_circle_packing_evaluator(n_circles: int = 10) -> Evaluator:
    """Create an evaluator for the circle packing problem."""
    
    problem_description = f"""
Circle Packing Problem
======================

Pack {n_circles} non-overlapping circles into a unit square [0,1] x [0,1]
to MAXIMIZE the sum of their radii.

Your function should return a list of circles, where each circle is 
a tuple (x, y, r):
- x: x-coordinate of center (0 <= x <= 1)
- y: y-coordinate of center (0 <= y <= 1)  
- r: radius of the circle (r > 0)

Constraints:
1. All circles must fit entirely within the unit square
2. No two circles may overlap (they may touch)
3. You must return exactly {n_circles} circles

The score is the sum of all radii. Higher is better.

Example return value:
    [(0.25, 0.25, 0.2), (0.75, 0.75, 0.2), ...]  # List of {n_circles} circles
"""
    
    return MaximizationEvaluator(
        problem_description=problem_description,
        function_name="pack_circles",
        fitness_function=circle_packing_fitness,
        validator=validate_circle_packing,
        timeout_seconds=5.0  # 5s is plenty for circle packing
    )


def get_circle_packing_seeds(n_circles: int = 10) -> List[str]:
    """Get seed programs for circle packing."""
    
    # Simple grid-based seed
    grid_seed = f'''
def pack_circles():
    """Simple grid-based packing."""
    import math
    n = {n_circles}
    
    # Arrange in a grid
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    # Calculate cell size and radius
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows
    r = min(cell_w, cell_h) / 2 * 0.9  # 90% of half-cell to leave margin
    
    circles = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) * cell_w
        y = (row + 0.5) * cell_h
        circles.append((x, y, r))
    
    return circles
'''
    
    # Random placement with collision avoidance
    random_seed = f'''
def pack_circles():
    """Random placement with collision detection."""
    import random
    import math
    
    n = {n_circles}
    circles = []
    max_attempts = 1000
    
    for _ in range(n):
        for attempt in range(max_attempts):
            # Random position and radius
            r = random.uniform(0.02, 0.15)
            x = random.uniform(r, 1 - r)
            y = random.uniform(r, 1 - r)
            
            # Check collision with existing circles
            valid = True
            for cx, cy, cr in circles:
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < r + cr:
                    valid = False
                    break
            
            if valid:
                circles.append((x, y, r))
                break
        else:
            # Couldn't place, add tiny circle
            circles.append((random.uniform(0.01, 0.99), random.uniform(0.01, 0.99), 0.001))
    
    return circles
'''
    
    # Greedy largest-first
    greedy_seed = f'''
def pack_circles():
    """Greedy approach: place largest possible circles first."""
    import math
    
    n = {n_circles}
    circles = []
    
    def can_place(x, y, r, existing):
        """Check if a circle can be placed."""
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
        for cx, cy, cr in existing:
            if math.sqrt((x - cx)**2 + (y - cy)**2) < r + cr:
                return False
        return True
    
    def find_best_position(r, existing):
        """Find best position for a circle of given radius."""
        best_pos = None
        # Try grid of positions
        for xi in range(20):
            for yi in range(20):
                x = r + xi * (1 - 2*r) / 19
                y = r + yi * (1 - 2*r) / 19
                if can_place(x, y, r, existing):
                    return (x, y)
        return None
    
    # Try to place circles with decreasing radius
    for i in range(n):
        for r in [0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]:
            pos = find_best_position(r, circles)
            if pos:
                circles.append((pos[0], pos[1], r))
                break
        else:
            circles.append((0.5, 0.5, 0.001))  # Fallback
    
    return circles
'''
    
    return [grid_seed, random_seed, greedy_seed]


# =============================================================================
# SORTING OPTIMIZATION PROBLEM
# =============================================================================

def create_sorting_evaluator() -> Evaluator:
    """Create an evaluator for optimizing sorting algorithms."""
    
    problem_description = """
Sorting Algorithm Optimization
==============================

Implement a sorting function that sorts a list of integers.
Your function will be evaluated on:
1. Correctness (must produce sorted output)
2. Number of comparisons (fewer is better)
3. Runtime performance

Your function should accept a list and return the sorted list.
You may also return a tuple (sorted_list, num_comparisons) to report
the comparison count explicitly.

Focus on minimizing comparisons while maintaining correctness.
"""
    
    import random
    
    # Generate test cases
    test_cases = [
        ([3, 1, 4, 1, 5, 9, 2, 6], sorted([3, 1, 4, 1, 5, 9, 2, 6])),
        ([1], [1]),
        ([], []),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
        (list(range(10)), list(range(10))),  # Already sorted
        (list(range(10, 0, -1)), list(range(1, 11))),  # Reverse sorted
    ]
    
    # Add some random test cases
    for size in [10, 50, 100]:
        data = [random.randint(0, 1000) for _ in range(size)]
        test_cases.append((data.copy(), sorted(data)))
    
    def objective(outputs):
        """Objective: correctness + speed bonus."""
        correct = sum(1 for o in outputs if o is not None) / len(outputs)
        return correct
    
    return FunctionOptimizationEvaluator(
        problem_description=problem_description,
        function_name="sort",
        test_cases=test_cases,
        objective_function=objective,
        timeout_seconds=5.0,
        correctness_weight=0.8,
        quality_weight=0.2
    )


def get_sorting_seeds() -> List[str]:
    """Seed implementations for sorting."""
    
    builtin_seed = '''
def sort(arr):
    """Use Python's builtin sort."""
    return sorted(arr)
'''
    
    bubble_seed = '''
def sort(arr):
    """Bubble sort - simple but inefficient."""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
'''
    
    insertion_seed = '''
def sort(arr):
    """Insertion sort - good for small/nearly sorted."""
    arr = arr.copy()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
'''
    
    return [builtin_seed, bubble_seed, insertion_seed]


# =============================================================================
# MATHEMATICAL FUNCTION DISCOVERY
# =============================================================================

def create_function_discovery_evaluator(
    target_function: Callable[[float], float],
    x_range: Tuple[float, float] = (-10, 10),
    n_samples: int = 100
) -> Evaluator:
    """
    Create an evaluator for discovering mathematical functions.
    
    Given samples from a target function, evolve a function that
    approximates it.
    """
    import random
    
    # Generate training data
    x_values = [x_range[0] + i * (x_range[1] - x_range[0]) / (n_samples - 1) 
                for i in range(n_samples)]
    test_cases = [(x, target_function(x)) for x in x_values]
    
    problem_description = f"""
Mathematical Function Discovery
===============================

Discover a mathematical function f(x) that fits the given data points.
Your function should take a single float x and return a float.

The function will be evaluated on {n_samples} points in the range [{x_range[0]}, {x_range[1]}].
Your score is based on how closely your function matches the target.

You may use the math module for mathematical operations.

Example:
    def f(x):
        import math
        return math.sin(x) + x**2
"""
    
    def objective(outputs):
        """Mean squared error (inverted for maximization)."""
        errors = []
        for output, (x, expected) in zip(outputs, test_cases):
            if output is not None:
                errors.append((output - expected) ** 2)
        
        if not errors:
            return 0.0
        
        mse = sum(errors) / len(errors)
        # Convert to 0-1 score (higher is better)
        return 1.0 / (1.0 + mse)
    
    return FunctionOptimizationEvaluator(
        problem_description=problem_description,
        function_name="f",
        test_cases=test_cases,
        objective_function=objective,
        timeout_seconds=2.0,
        correctness_weight=0.0,  # We care about approximation quality
        quality_weight=1.0
    )


# =============================================================================
# MATRIX MULTIPLICATION OPTIMIZATION
# =============================================================================
#
# Inspired by AlphaEvolve's discovery of faster matrix multiplication algorithms.
# The goal is to find algorithms that use fewer scalar multiplications.
#
# Naive 2x2: 8 multiplications
# Strassen 2x2: 7 multiplications  
# 
# AlphaEvolve found improvements for 4x4 complex matrices (48 vs 49 for Strassen)
# =============================================================================

class MultiplicationCounter:
    """Context manager to count multiplication operations."""
    
    def __init__(self):
        self.count = 0
        self._original_mul = None
    
    def counted_mul(self, a, b):
        self.count += 1
        return a * b
    
    def reset(self):
        self.count = 0


def validate_matmul(result: Any, A: List[List[float]], B: List[List[float]]) -> Tuple[bool, str]:
    """Validate matrix multiplication result."""
    if result is None:
        return False, "No result returned"
    
    if not isinstance(result, (list, tuple)):
        return False, f"Expected list, got {type(result)}"
    
    n = len(A)
    m = len(B[0]) if B else 0
    
    if len(result) != n:
        return False, f"Wrong number of rows: {len(result)} vs {n}"
    
    for i, row in enumerate(result):
        if not isinstance(row, (list, tuple)):
            return False, f"Row {i} is not a list"
        if len(row) != m:
            return False, f"Row {i} has wrong length: {len(row)} vs {m}"
    
    # Compute expected result
    expected = [[sum(A[i][k] * B[k][j] for k in range(len(B))) 
                 for j in range(m)] for i in range(n)]
    
    # Compare with tolerance
    for i in range(n):
        for j in range(m):
            if abs(result[i][j] - expected[i][j]) > 1e-6:
                return False, f"Wrong value at [{i}][{j}]: {result[i][j]} vs {expected[i][j]}"
    
    return True, ""


def matmul_fitness(result: Any, mul_count: int, n: int) -> float:
    """
    Fitness function for matrix multiplication.
    
    Lower multiplication count = higher fitness.
    Naive n×n multiplication uses n³ multiplications.
    """
    naive_count = n ** 3
    
    # Fitness is how much we beat naive (normalized)
    # Score of 1.0 = naive, >1.0 = better than naive
    if mul_count <= 0:
        return 0.0
    
    return naive_count / mul_count


def create_matmul_evaluator(matrix_size: int = 2) -> Evaluator:
    """Create an evaluator for matrix multiplication optimization."""
    
    problem_description = f"""
Matrix Multiplication Optimization
==================================

Implement a function to multiply two {matrix_size}x{matrix_size} matrices using 
as FEW SCALAR MULTIPLICATIONS as possible.

Naive {matrix_size}x{matrix_size} multiplication uses {matrix_size**3} multiplications.
Strassen-like algorithms can do better!

Your function receives:
- A: First matrix as list of lists
- B: Second matrix as list of lists  

Return: Result matrix C where C = A × B

CRITICAL: Use the provided 'mul' function for ALL multiplications.
This is how we count operations. Example:
    result = mul(a, b)  # Counts as 1 multiplication
    
Do NOT use: a * b directly (won't be counted)

The fewer multiplications while maintaining correctness, the better!

def matmul(A, B, mul):
    # Your implementation using mul() for all multiplications
    # Return the result matrix C = A × B
    pass
"""
    
    return MatMulEvaluator(
        problem_description=problem_description,
        matrix_size=matrix_size,
        timeout_seconds=5.0
    )


class MatMulEvaluator(Evaluator):
    """Evaluator for matrix multiplication optimization."""
    
    def __init__(self, problem_description: str, matrix_size: int = 2, timeout_seconds: float = 5.0):
        self.problem_description = problem_description
        self.matrix_size = matrix_size
        self.timeout_seconds = timeout_seconds
        self.function_name = "matmul"
        
        # Generate test cases
        self.test_cases = self._generate_test_cases()
    
    def _generate_test_cases(self) -> List[Tuple[List[List[float]], List[List[float]]]]:
        """Generate test matrices."""
        import random
        random.seed(42)  # Reproducible
        
        n = self.matrix_size
        cases = []
        
        # Identity test
        I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        A = [[float(i * n + j + 1) for j in range(n)] for i in range(n)]
        cases.append((A, I))  # A × I = A
        
        # Simple integer matrices
        for _ in range(3):
            A = [[float(random.randint(1, 5)) for _ in range(n)] for _ in range(n)]
            B = [[float(random.randint(1, 5)) for _ in range(n)] for _ in range(n)]
            cases.append((A, B))
        
        return cases
    
    def get_problem_description(self) -> str:
        return self.problem_description
    
    def get_function_name(self) -> str:
        return self.function_name
    
    def evaluate(self, code: str) -> 'EvaluationResult':
        """Evaluate matrix multiplication implementation."""
        from .evaluator import EvaluationResult, compute_code_complexity, normalize_runtime
        import time
        
        start_time = time.perf_counter()
        
        # Set up execution environment
        counter = MultiplicationCounter()
        
        safe_globals = {
            '__builtins__': {
                'range': range, 'len': len, 'list': list,
                'sum': sum, 'int': int, 'float': float,
                'True': True, 'False': False, 'None': None,
                'abs': abs, 'min': min, 'max': max,
            }
        }
        
        try:
            exec(code, safe_globals)
        except Exception as e:
            return EvaluationResult(
                success=False, score=0.0, metrics={},
                error=f"Compilation error: {e}",
                runtime_ms=(time.perf_counter() - start_time) * 1000
            )
        
        if self.function_name not in safe_globals:
            return EvaluationResult(
                success=False, score=0.0, metrics={},
                error=f"Function '{self.function_name}' not defined",
                runtime_ms=(time.perf_counter() - start_time) * 1000
            )
        
        func = safe_globals[self.function_name]
        
        # Run test cases
        total_muls = 0
        passed = 0
        
        for A, B in self.test_cases:
            counter.reset()
            
            try:
                result = func(A, B, counter.counted_mul)
                valid, error = validate_matmul(result, A, B)
                
                if valid:
                    passed += 1
                    total_muls += counter.count
            except Exception as e:
                return EvaluationResult(
                    success=False, score=0.0, metrics={},
                    error=f"Runtime error: {e}",
                    runtime_ms=(time.perf_counter() - start_time) * 1000
                )
        
        if passed == 0:
            return EvaluationResult(
                success=False, score=0.0, metrics={},
                error="No test cases passed",
                runtime_ms=(time.perf_counter() - start_time) * 1000
            )
        
        # Average multiplications per test
        avg_muls = total_muls / passed
        naive_muls = self.matrix_size ** 3
        
        # Fitness: ratio of naive to actual (higher = better)
        fitness = naive_muls / avg_muls if avg_muls > 0 else 0.0
        
        runtime_ms = (time.perf_counter() - start_time) * 1000
        
        return EvaluationResult(
            success=True,
            score=fitness,
            metrics={
                "fitness": fitness,
                "avg_multiplications": avg_muls,
                "naive_multiplications": naive_muls,
                "tests_passed": passed,
                "tests_total": len(self.test_cases),
                "complexity": compute_code_complexity(code),
                "runtime": normalize_runtime(runtime_ms)
            },
            runtime_ms=runtime_ms
        )


def get_matmul_seeds(matrix_size: int = 2) -> List[str]:
    """Seed programs for matrix multiplication."""
    
    naive_seed = f'''
def matmul(A, B, mul):
    """Naive O(n³) matrix multiplication."""
    n = len(A)
    m = len(B[0])
    p = len(B)
    
    # Initialize result matrix
    C = [[0.0 for _ in range(m)] for _ in range(n)]
    
    # Standard triple loop
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i][j] = C[i][j] + mul(A[i][k], B[k][j])
    
    return C
'''
    
    strassen_2x2_seed = '''
def matmul(A, B, mul):
    """Strassen-like algorithm for 2x2 matrices (7 multiplications)."""
    # Only works for 2x2!
    if len(A) != 2 or len(B) != 2:
        # Fall back to naive for non-2x2
        n = len(A)
        C = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] = C[i][j] + mul(A[i][k], B[k][j])
        return C
    
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    e, f, g, h = B[0][0], B[0][1], B[1][0], B[1][1]
    
    # Strassen's 7 multiplications
    p1 = mul(a, f - h)
    p2 = mul(a + b, h)
    p3 = mul(c + d, e)
    p4 = mul(d, g - e)
    p5 = mul(a + d, e + h)
    p6 = mul(b - d, g + h)
    p7 = mul(a - c, e + f)
    
    # Combine results
    C = [
        [p5 + p4 - p2 + p6, p1 + p2],
        [p3 + p4, p1 + p5 - p3 - p7]
    ]
    
    return C
'''
    
    return [naive_seed, strassen_2x2_seed]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def list_problems() -> List[str]:
    """List available problems."""
    return [
        "circle_packing",
        "sorting",
        "function_discovery",
        "matmul"
    ]


def create_problem(name: str, **kwargs) -> Tuple[Evaluator, List[str]]:
    """Create a problem evaluator and seed programs by name."""
    
    if name == "circle_packing":
        n = kwargs.get("n_circles", 10)
        return create_circle_packing_evaluator(n), get_circle_packing_seeds(n)
    
    elif name == "sorting":
        return create_sorting_evaluator(), get_sorting_seeds()
    
    elif name == "function_discovery":
        target = kwargs.get("target_function", lambda x: math.sin(x))
        return create_function_discovery_evaluator(target), [
            "def f(x):\n    return x",
            "def f(x):\n    import math\n    return math.sin(x)",
        ]
    
    elif name == "matmul":
        size = kwargs.get("matrix_size", 2)
        return create_matmul_evaluator(size), get_matmul_seeds(size)
    
    else:
        raise ValueError(f"Unknown problem: {name}. Available: {list_problems()}")
