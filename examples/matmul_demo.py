#!/usr/bin/env python3
"""
Matrix Multiplication Optimization Demo

Inspired by AlphaEvolve's discovery of faster matrix multiplication algorithms.
Goal: Find algorithms that use fewer scalar multiplications than naive O(n³).

Naive 2x2: 8 multiplications
Strassen 2x2: 7 multiplications (discovered 1969)
AlphaEvolve: Found 48 multiplications for 4x4 complex (vs 49 for Strassen)

Usage:
    python examples/matmul_demo.py --model qwen2.5-coder:7b --generations 20
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import argparse
import time
import re
import logging

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from core.database import ProgramDatabase, Program
from core.local_llm import LocalLLMEnsemble, check_ollama_status
from core.problems import create_matmul_evaluator, get_matmul_seeds


def extract_code(response: str, function_name: str = "matmul") -> str:
    """Extract Python code from LLM response."""
    
    # Strategy 1: Find ```python block
    match = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if f'def {function_name}' in code:
            return code
    
    # Strategy 2: Find ``` block
    match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if f'def {function_name}' in code:
            return code
    
    # Strategy 3: Find function definition
    match = re.search(rf'(def {function_name}\s*\(.*?(?=\ndef |\Z))', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None


def create_matmul_prompt(code: str, score: float, muls: float, hint: str) -> str:
    """Create a prompt for matrix multiplication optimization."""
    
    return f"""Optimize this matrix multiplication to use FEWER multiplications.

Current implementation (score: {score:.4f}, uses {muls:.0f} multiplications):

```python
{code}
```

RULES:
1. Use mul(a, b) for ALL multiplications - this is how we count them
2. Function signature: def matmul(A, B, mul)
3. Must return correct result matrix C = A × B
4. Fewer multiplications = higher score

HINT: {hint}

REFERENCE:
- Naive 2x2 uses 8 multiplications
- Strassen's 2x2 uses only 7 multiplications (via clever combinations)
- Can you find a pattern with even fewer, or generalize efficiently?

Write the COMPLETE improved function:

```python
def matmul(A, B, mul):
    # Your optimized implementation
    pass
```"""


async def run_matmul_evolution(
    model: str = "qwen2.5-coder:7b",
    matrix_size: int = 2,
    generations: int = 20,
    candidates_per_gen: int = 4,
    timeout_seconds: float = 60.0,
    debug: bool = True
):
    """Run evolution to optimize matrix multiplication."""
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║       AlphaEvolve-Mini: Matrix Multiplication Demo            ║
╠═══════════════════════════════════════════════════════════════╣
║  Evolve algorithms that use fewer scalar multiplications      ║
╚═══════════════════════════════════════════════════════════════╝

Model: {model}
Matrix size: {matrix_size}x{matrix_size}
Naive multiplications: {matrix_size**3}
Generations: {generations}
Candidates/gen: {candidates_per_gen}
""")
    
    # Check Ollama
    status = await check_ollama_status()
    if not status["running"]:
        print("❌ Ollama not running! Start with: ollama serve")
        return None
    
    # Setup
    evaluator = create_matmul_evaluator(matrix_size)
    seeds = get_matmul_seeds(matrix_size)
    database = ProgramDatabase(num_islands=2, island_capacity=20)
    llm = LocalLLMEnsemble(provider="ollama", model_name=model, timeout_seconds=timeout_seconds)
    
    # Evaluate seeds
    print("Evaluating seed implementations...")
    for i, seed in enumerate(seeds):
        result = evaluator.evaluate(seed)
        if result.success:
            prog = Program(code=seed, score=result.score, metrics=result.metrics)
            database.add_program(prog)
            muls = result.metrics.get("avg_multiplications", "?")
            print(f"  Seed {i+1}: score={result.score:.4f}, multiplications={muls}")
    
    print(f"\nInitial best: {database.best_program.score:.4f}")
    naive_muls = matrix_size ** 3
    print(f"Target: Beat {naive_muls} multiplications (score > 1.0)")
    print(f"\n{'='*60}")
    print("Starting evolution...")
    print(f"{'='*60}\n")
    
    # Hints for optimization
    hints = [
        "Try Strassen's approach: compute 7 products and combine them cleverly",
        "Look for common subexpressions that can be reused",
        "Can you precompute sums/differences of matrix elements?",
        "Think about which products can be shared across output elements",
        "The key insight: a*e + b*g can sometimes share computation with a*f + b*h",
        "Try breaking the problem into smaller subproblems",
        "Consider Karatsuba-like tricks: (a+b)(c+d) = ac + ad + bc + bd",
        "What if you compute (A[0][0] + A[1][1]) * (B[0][0] + B[1][1]) first?",
    ]
    
    # Stats
    total_evals = 0
    successful_evals = 0
    improvements = 0
    
    start_time = time.time()
    
    for gen in range(generations):
        gen_start = time.time()
        best_before = database.best_program.score
        
        for c in range(candidates_per_gen):
            # Select parent
            sample = database.sample_for_prompt()
            parent = sample["best"]
            
            # Create prompt
            muls = parent.metrics.get("avg_multiplications", naive_muls)
            hint = hints[(gen * candidates_per_gen + c) % len(hints)]
            prompt = create_matmul_prompt(parent.code, parent.score, muls, hint)
            
            try:
                # Generate
                temp = 0.7 + (c * 0.1)
                response = await asyncio.wait_for(
                    llm.generate(prompt, temperature=temp, max_tokens=1500),
                    timeout=timeout_seconds
                )
                
                if not response.content:
                    continue
                
                # Extract code
                code = extract_code(response.content, "matmul")
                if not code:
                    if debug and gen < 3:
                        print(f"  [Gen {gen+1}.{c+1}] Failed to extract code")
                    continue
                
                # Evaluate
                total_evals += 1
                result = evaluator.evaluate(code)
                
                if not result.success:
                    if debug and gen < 3:
                        print(f"  [Gen {gen+1}.{c+1}] Failed: {result.error}")
                    continue
                
                successful_evals += 1
                
                # Add to database
                prog = Program(
                    code=code,
                    score=result.score,
                    metrics=result.metrics,
                    parent_ids=[parent.id]
                )
                is_best = database.add_program(prog)
                
                muls = result.metrics.get("avg_multiplications", "?")
                if debug:
                    status = "🎉 NEW BEST!" if result.score > best_before else ""
                    print(f"  [Gen {gen+1}.{c+1}] Score: {result.score:.4f}, muls: {muls} {status}")
                
            except asyncio.TimeoutError:
                print(f"  [Gen {gen+1}.{c+1}] ⏱️ Timeout")
            except Exception as e:
                if debug:
                    print(f"  [Gen {gen+1}.{c+1}] Error: {e}")
        
        database.step_generation()
        
        # Report progress
        if database.best_program.score > best_before + 0.001:
            improvements += 1
            best_muls = database.best_program.metrics.get("avg_multiplications", "?")
            print(f"\n🎉 Gen {gen+1}: NEW BEST = {database.best_program.score:.4f} ({best_muls} multiplications)\n")
        elif gen % 5 == 0:
            gen_time = time.time() - gen_start
            best_muls = database.best_program.metrics.get("avg_multiplications", "?")
            print(f"Gen {gen+1}: best={database.best_program.score:.4f} ({best_muls} muls), time={gen_time:.1f}s")
    
    # Results
    elapsed = time.time() - start_time
    best = database.best_program
    best_muls = best.metrics.get("avg_multiplications", "?")
    
    print(f"\n{'='*60}")
    print("EVOLUTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Evaluations: {total_evals} ({successful_evals} successful)")
    print(f"  Improvements: {improvements}")
    print(f"  Best score: {best.score:.4f}")
    print(f"  Best multiplications: {best_muls} (naive: {naive_muls})")
    
    if best.score > 1.0:
        savings = (1 - naive_muls / best_muls) * 100 if isinstance(best_muls, (int, float)) else 0
        print(f"  🎉 Beat naive by {savings:.1f}%!")
    
    print(f"\n{'='*60}")
    print("BEST SOLUTION")
    print(f"{'='*60}")
    print(best.code)
    
    return best


async def main():
    parser = argparse.ArgumentParser(description="Matrix Multiplication Optimization Demo")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--size", type=int, default=2, help="Matrix size (default: 2)")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    print(f"Max estimated time: {args.generations * args.candidates * args.timeout * 1.5 / 60:.0f} minutes")
    print("Press Ctrl+C to stop early\n")
    
    try:
        await run_matmul_evolution(
            model=args.model,
            matrix_size=args.size,
            generations=args.generations,
            candidates_per_gen=args.candidates,
            timeout_seconds=args.timeout,
            debug=not args.quiet
        )
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")


if __name__ == "__main__":
    asyncio.run(main())
