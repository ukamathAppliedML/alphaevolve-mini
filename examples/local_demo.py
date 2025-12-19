#!/usr/bin/env python3
"""
AlphaEvolve-Mini: Local Demo

Run evolutionary code optimization completely offline using local LLMs.

Requirements:
    1. Ollama installed (https://ollama.ai)
    2. A model pulled: ollama pull qwen2.5-coder:1.5b

Usage:
    python examples/local_demo.py                    # Use defaults
    python examples/local_demo.py --model codellama:7b
    python examples/local_demo.py --generations 50
    python examples/local_demo.py --problem sorting
"""

import asyncio
import argparse
import sys
import os
import time
import logging

# Suppress verbose httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import ProgramDatabase, Program
from core.evaluator import MaximizationEvaluator
from core.local_llm import LocalLLMEnsemble, check_ollama_status, setup_ollama_model
class CompactPromptSampler:
    """Compact prompt sampler optimized for local LLMs."""
    
    def __init__(self, database, problem_description, function_name):
        self.database = database
        self.problem_description = problem_description
        self.function_name = function_name
        self.mutation_hints = [
            "optimize the algorithm",
            "improve efficiency", 
            "try a different approach",
            "fix any bugs",
            "simplify the code",
        ]
    
    def generate_batch_prompts(self, n=3, vary_hints=True):
        import random
        prompts = []
        sample = self.database.sample_for_prompt()
        
        parents = []
        if sample["best"]:
            parents.append(sample["best"])
        parents.extend(sample.get("parents", []))
        
        for i in range(n):
            parent = parents[i % len(parents)] if parents else None
            hint = random.choice(self.mutation_hints)
            
            if parent:
                prompt = f"""Improve this circle packing (score: {parent.score:.4f}).

RULES: Only use math/random (NO numpy/scipy/external packages!)

```python
{parent.code}
```

{hint.upper()}. Write complete function:

```python
def pack_circles():
    import math
    # implementation
    return circles  # list of (x, y, radius)
```"""
            else:
                prompt = f"""Write circle packing function. Pack 8 circles in unit square [0,1]x[0,1].
RULES: Only use math/random (NO numpy/scipy!)

```python
def pack_circles():
    import math
    return circles  # list of (x, y, radius)
```"""
            
            prompts.append({
                "prompt": prompt,
                "parent_id": parent.id if parent else None,
                "hint": hint,
                "index": i
            })
        
        return prompts
from core.problems import (
    create_circle_packing_evaluator, 
    get_circle_packing_seeds,
    validate_circle_packing,
    circle_packing_fitness
)


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    import re
    
    # Try to find code block
    match = re.search(r'```python\n?(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try generic code block
    match = re.search(r'```\n?(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If response looks like code, use it directly
    if 'def ' in response:
        # Extract from def to end
        lines = response.split('\n')
        code_lines = []
        in_function = False
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
            if in_function:
                code_lines.append(line)
        if code_lines:
            return '\n'.join(code_lines)
    
    return response


async def run_local_evolution(
    problem: str = "circle_packing",
    model: str = "qwen2.5-coder:1.5b",
    generations: int = 20,
    candidates_per_gen: int = 3,
    verbose: bool = True
):
    """
    Run evolutionary optimization with local LLM.
    """
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         AlphaEvolve-Mini: Local Evolution Demo                ║
╠═══════════════════════════════════════════════════════════════╣
║  Running completely offline with local LLM                    ║
╚═══════════════════════════════════════════════════════════════╝

Configuration:
  Problem: {problem}
  Model: {model}
  Generations: {generations}
  Candidates/generation: {candidates_per_gen}
""")
    
    # Check Ollama status
    print("Checking Ollama status...")
    status = await check_ollama_status()
    
    if not status["running"]:
        print("\n❌ Ollama is not running!")
        print("   Start it with: ollama serve")
        print("   Then pull a model: ollama pull qwen2.5-coder:1.5b")
        return None
    
    # Check if model is available
    if model not in status["models"] and not any(model in m for m in status["models"]):
        print(f"\n⚠️  Model '{model}' not found. Available models:")
        for m in status["models"]:
            print(f"   - {m}")
        print(f"\n   Pull it with: ollama pull {model}")
        
        # Try to use an available model
        if status["models"]:
            model = status["models"][0]
            print(f"\n   Using {model} instead...")
        else:
            return None
    
    print(f"✓ Using model: {model}\n")
    
    # Create components
    # 1. Evaluator
    if problem == "circle_packing":
        n_circles = 8  # Smaller for faster iteration
        evaluator = create_circle_packing_evaluator(n_circles)
        seeds = get_circle_packing_seeds(n_circles)
        problem_desc = f"Pack {n_circles} non-overlapping circles in unit square to maximize sum of radii"
        func_name = "pack_circles"
    elif problem == "sorting":
        from core.problems import create_sorting_evaluator, get_sorting_seeds
        evaluator = create_sorting_evaluator()
        seeds = get_sorting_seeds()
        problem_desc = "Implement efficient sorting"
        func_name = "sort"
    else:
        print(f"Unknown problem: {problem}")
        return None
    
    # 2. Database
    database = ProgramDatabase(num_islands=3, island_capacity=30)
    
    # 3. LLM
    llm = LocalLLMEnsemble(provider="ollama", model_name=model)
    
    # 4. Prompt sampler (compact for local models)
    prompt_sampler = CompactPromptSampler(
        database=database,
        problem_description=problem_desc,
        function_name=func_name
    )
    
    # Initialize with seeds
    print("Evaluating seed solutions...")
    for i, seed_code in enumerate(seeds):
        result = evaluator.evaluate(seed_code)
        if result.success:
            program = Program(code=seed_code, score=result.score, metrics=result.metrics)
            database.add_program(program)
            if verbose:
                print(f"  Seed {i+1}: score = {result.score:.4f}")
    
    print(f"\nInitial best score: {database.best_program.score:.4f}")
    print(f"\n{'='*50}")
    print("Starting evolution...")
    print(f"{'='*50}\n")
    
    # Evolution loop
    start_time = time.time()
    total_evaluations = 0
    improvements = 0
    
    for gen in range(generations):
        gen_start = time.time()
        best_before = database.best_program.score
        
        # Generate prompts
        prompts_data = prompt_sampler.generate_batch_prompts(n=candidates_per_gen)
        
        # Generate and evaluate candidates
        for prompt_info in prompts_data:
            try:
                # Generate mutation
                response = await llm.generate(
                    prompt_info["prompt"],
                    temperature=0.7 + (gen % 3) * 0.1,  # Vary temperature
                    max_tokens=2000
                )
                
                # Extract code
                code = extract_code(response.content)
                if not code:
                    continue
                
                # Evaluate
                result = evaluator.evaluate(code)
                total_evaluations += 1
                
                if result.success and result.score > 0:
                    parent_id = prompt_info.get("parent_id")
                    program = Program(
                        code=code,
                        score=result.score,
                        metrics=result.metrics,
                        parent_ids=[parent_id] if parent_id else []
                    )
                    database.add_program(program)
                    
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                continue
        
        # Step generation
        database.step_generation()
        
        # Check for improvement
        if database.best_program.score > best_before:
            improvements += 1
            print(f"Gen {gen+1:3d}: 🎉 NEW BEST = {database.best_program.score:.4f} "
                  f"(+{database.best_program.score - best_before:.4f})")
        elif verbose and gen % 5 == 0:
            gen_time = time.time() - gen_start
            print(f"Gen {gen+1:3d}: best = {database.best_program.score:.4f}, "
                  f"programs = {len(database.all_programs)}, "
                  f"time = {gen_time:.1f}s")
    
    # Results
    elapsed = time.time() - start_time
    
    print(f"\n{'='*50}")
    print("EVOLUTION COMPLETE")
    print(f"{'='*50}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Generations: {generations}")
    print(f"  Total evaluations: {total_evaluations}")
    print(f"  Improvements found: {improvements}")
    print(f"  Final best score: {database.best_program.score:.4f}")
    print(f"\n  LLM Stats: {llm.get_stats()}")
    
    print(f"\n{'='*50}")
    print("BEST SOLUTION")
    print(f"{'='*50}")
    print(database.best_program.code)
    
    # Verify and show result for circle packing
    if problem == "circle_packing":
        try:
            exec_globals = {}
            exec(database.best_program.code, exec_globals)
            circles = exec_globals['pack_circles']()
            
            valid, msg = validate_circle_packing(circles)
            if valid:
                fitness = circle_packing_fitness(circles)
                print(f"\n✓ Valid packing with {len(circles)} circles")
                print(f"  Sum of radii: {fitness:.4f}")
            else:
                print(f"\n✗ Invalid: {msg}")
        except Exception as e:
            print(f"\n✗ Execution error: {e}")
    
    return database.best_program


async def main():
    parser = argparse.ArgumentParser(
        description="AlphaEvolve-Mini Local Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python local_demo.py                           # Run with defaults
  python local_demo.py --model codellama:7b      # Use different model
  python local_demo.py --generations 50          # More generations
  python local_demo.py --problem sorting         # Different problem

Recommended models (pull with 'ollama pull <model>'):
  - qwen2.5-coder:1.5b  (fast, good for code, ~2GB)
  - qwen2.5-coder:7b    (best quality, ~8GB)
  - deepseek-coder:1.3b (very fast, ~1.5GB)
  - codellama:7b        (good for Python, ~8GB)
        """
    )
    
    parser.add_argument("--model", default="qwen2.5-coder:1.5b",
                       help="Ollama model to use")
    parser.add_argument("--problem", default="circle_packing",
                       choices=["circle_packing", "sorting"],
                       help="Problem to solve")
    parser.add_argument("--generations", type=int, default=20,
                       help="Number of generations")
    parser.add_argument("--candidates", type=int, default=3,
                       help="Candidates per generation")
    parser.add_argument("--quiet", action="store_true",
                       help="Less verbose output")
    
    args = parser.parse_args()
    
    best = await run_local_evolution(
        problem=args.problem,
        model=args.model,
        generations=args.generations,
        candidates_per_gen=args.candidates,
        verbose=not args.quiet
    )
    
    return best


if __name__ == "__main__":
    asyncio.run(main())
