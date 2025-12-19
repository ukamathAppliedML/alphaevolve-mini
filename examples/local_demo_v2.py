#!/usr/bin/env python3
"""
AlphaEvolve-Mini: Improved Local Demo

Better prompts for Qwen/CodeLlama models with debugging to see what's happening.
"""

import asyncio
import argparse
import sys
import os
import time
import re
import logging

# Suppress verbose httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import ProgramDatabase, Program
from core.local_llm import LocalLLMEnsemble, check_ollama_status
from core.problems import (
    create_circle_packing_evaluator, 
    get_circle_packing_seeds,
    validate_circle_packing,
    circle_packing_fitness
)


def extract_code(response: str, function_name: str = "pack_circles") -> str:
    """Extract Python code - improved version with multiple strategies."""
    
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
    
    # Strategy 3: Find the function definition directly
    match = re.search(rf'(def {function_name}\s*\(.*?(?=\ndef |\Z))', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Strategy 4: If response contains code-like content
    if f'def {function_name}' in response:
        lines = response.split('\n')
        code_lines = []
        in_function = False
        indent_level = 0
        
        for line in lines:
            if f'def {function_name}' in line:
                in_function = True
                code_lines = [line]
                indent_level = len(line) - len(line.lstrip())
            elif in_function:
                if line.strip() == '':
                    code_lines.append(line)
                elif line.startswith(' ' * (indent_level + 1)) or line.startswith('\t'):
                    code_lines.append(line)
                elif line.strip().startswith('#'):
                    code_lines.append(line)
                elif line.strip() and not line.startswith(' '):
                    break  # New top-level definition
                else:
                    code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
    
    return None


def create_qwen_prompt(code: str, score: float, hint: str, best_code: str = None, best_score: float = None) -> str:
    """
    Create a prompt optimized for Qwen-Coder models.
    
    Qwen models respond well to:
    - Clear task description
    - Explicit output format
    - Showing the expected improvement
    """
    
    prompt = f"""You are an expert Python programmer optimizing a circle packing algorithm.

TASK: Pack 8 non-overlapping circles into a unit square [0,1]×[0,1] to MAXIMIZE the sum of radii.

IMPORTANT CONSTRAINTS:
- ONLY use: import math, import random (NO numpy, scipy, or other packages!)
- Return a list of tuples: [(x, y, radius), ...]
- All circles must fit within [0,1]×[0,1]
- No two circles may overlap

CURRENT SOLUTION (score={score:.4f}):
```python
{code}
```

"""
    
    if best_code and best_score and best_score > score:
        prompt += f"""BEST KNOWN (score={best_score:.4f}):
```python
{best_code[:400]}
```

"""
    
    prompt += f"""YOUR TASK: {hint}

Key insights for better packing:
- One large circle (r≈0.29) in center leaves room for corners
- Try varying the search grid resolution (30x30 or 50x50)
- Place circles at tangent points of existing circles

Write the COMPLETE improved function using only math/random:

```python
def pack_circles():
    import math
    # Your improved implementation
    return circles
```"""
    
    return prompt


def create_creative_prompt(code: str, score: float, generation: int) -> str:
    """Create prompts that encourage more creative solutions."""
    
    strategies = [
        "Try a completely different approach - maybe hexagonal packing",
        "Optimize the radius selection - try finer gradations like [0.29, 0.25, 0.2, 0.15, ...]",
        "Place circles at tangent points of existing circles for tighter packing",
        "Use a finer search grid (40x40 instead of 20x20) for better positions",
        "Try placing the largest circle off-center to create more usable space",
        "Implement local optimization - nudge each circle to improve fit",
        "Use random restarts - try multiple configurations and keep best",
        "Place medium circles in corners first, then fill center",
    ]
    
    hint = strategies[generation % len(strategies)]
    
    return f"""Improve this circle packing (score: {score:.4f}). Use ONLY math/random (NO numpy/scipy)!

```python
{code}
```

SUGGESTION: {hint}

Write the COMPLETE improved function:

```python
def pack_circles():
    import math
    # Your implementation (no numpy/scipy!)
    return circles
```"""


async def run_improved_evolution(
    model: str = "qwen2.5-coder:7b",
    generations: int = 30,
    candidates_per_gen: int = 4,
    timeout_seconds: float = 60.0,
    debug: bool = True
):
    """Run evolution with improved prompts and debugging."""
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║       AlphaEvolve-Mini: Improved Local Demo                   ║
╠═══════════════════════════════════════════════════════════════╣
║  Better prompts for Qwen/CodeLlama + debugging output         ║
╚═══════════════════════════════════════════════════════════════╝

Model: {model}
Generations: {generations}
Candidates/gen: {candidates_per_gen}
LLM timeout: {timeout_seconds}s
""")
    
    # Check Ollama
    status = await check_ollama_status()
    if not status["running"]:
        print("❌ Ollama not running! Start with: ollama serve")
        return None
    
    # Setup
    n_circles = 8
    evaluator = create_circle_packing_evaluator(n_circles)
    seeds = get_circle_packing_seeds(n_circles)
    database = ProgramDatabase(num_islands=3, island_capacity=30)
    llm = LocalLLMEnsemble(provider="ollama", model_name=model, timeout_seconds=timeout_seconds)
    
    # Evaluate seeds
    print("Evaluating seeds...")
    for i, seed in enumerate(seeds):
        result = evaluator.evaluate(seed)
        if result.success:
            prog = Program(code=seed, score=result.score, metrics=result.metrics)
            database.add_program(prog)
            print(f"  Seed {i+1}: {result.score:.4f}")
    
    print(f"\nInitial best: {database.best_program.score:.4f}")
    print(f"\n{'='*60}")
    print("Starting evolution with improved prompts...")
    print(f"{'='*60}\n")
    
    # Track statistics
    total_evals = 0
    successful_evals = 0
    improvements = 0
    failed_extractions = 0
    failed_validations = 0
    
    hints = [
        "Increase the search grid resolution for better placement",
        "Try placing a large circle (r=0.29) in the center",
        "Use hexagonal packing arrangement",
        "Optimize radius selection with finer steps",
        "Add local search to fine-tune positions",
        "Try corner-first placement strategy",
        "Implement tangent-point placement",
        "Use physics-based settling simulation",
    ]
    
    start_time = time.time()
    
    for gen in range(generations):
        gen_start = time.time()
        best_before = database.best_program.score
        gen_successes = 0
        
        for c in range(candidates_per_gen):
            # Select parent
            sample = database.sample_for_prompt()
            parent = sample["best"] if gen % 2 == 0 else (
                sample["parents"][0] if sample["parents"] else sample["best"]
            )
            
            # Create prompt
            hint = hints[(gen * candidates_per_gen + c) % len(hints)]
            
            if gen < 5:
                # Detailed prompt for early generations
                prompt = create_qwen_prompt(
                    parent.code, 
                    parent.score, 
                    hint,
                    database.best_program.code if database.best_program.id != parent.id else None,
                    database.best_program.score if database.best_program.id != parent.id else None
                )
            else:
                # Shorter creative prompts later
                prompt = create_creative_prompt(parent.code, parent.score, gen)
            
            try:
                # Generate with timeout (60 seconds max per call - reduced from 120)
                temp = 0.6 + (c * 0.15)  # Vary temperature
                try:
                    response = await asyncio.wait_for(
                        llm.generate(prompt, temperature=temp, max_tokens=2000),
                        timeout=timeout_seconds  # Use configured timeout
                    )
                except asyncio.TimeoutError:
                    print(f"  [Gen {gen+1}.{c+1}] ⏱️ LLM timeout after {timeout_seconds}s - skipping")
                    continue
                
                # Check for empty response (Ollama timeout)
                if not response.content or len(response.content.strip()) < 10:
                    if debug:
                        print(f"  [Gen {gen+1}.{c+1}] Empty response - LLM may have timed out")
                    continue
                
                # Extract code
                code = extract_code(response.content, "pack_circles")
                
                if not code:
                    failed_extractions += 1
                    if debug and gen < 3:
                        print(f"  [Gen {gen+1}.{c+1}] Failed to extract code")
                        print(f"    Response preview: {response.content[:200]}...")
                    continue
                
                # Evaluate with timeout protection (sandbox already has timeout)
                total_evals += 1
                result = evaluator.evaluate(code)
                
                if not result.success:
                    failed_validations += 1
                    error_msg = result.error[:100] if result.error else 'Unknown'
                    if 'Timeout' in error_msg:
                        if debug:
                            print(f"  [Gen {gen+1}.{c+1}] Code execution timeout")
                    elif debug and gen < 3:
                        print(f"  [Gen {gen+1}.{c+1}] Validation failed: {error_msg}")
                    continue
                
                successful_evals += 1
                gen_successes += 1
                
                # Add to database
                prog = Program(
                    code=code,
                    score=result.score,
                    metrics=result.metrics,
                    parent_ids=[parent.id]
                )
                is_new_best = database.add_program(prog)
                
                if debug:
                    status = "🎉 NEW BEST!" if is_new_best and result.score > best_before else ""
                    print(f"  [Gen {gen+1}.{c+1}] Score: {result.score:.4f} {status}")
                
            except Exception as e:
                if debug:
                    print(f"  [Gen {gen+1}.{c+1}] Error: {e}")
                continue
        
        database.step_generation()
        
        # Check improvement
        if database.best_program.score > best_before + 0.001:
            improvements += 1
            print(f"\n🎉 Gen {gen+1}: NEW BEST = {database.best_program.score:.4f} (+{database.best_program.score - best_before:.4f})\n")
        elif gen % 5 == 0:
            gen_time = time.time() - gen_start
            print(f"Gen {gen+1}: best={database.best_program.score:.4f}, "
                  f"success={gen_successes}/{candidates_per_gen}, time={gen_time:.1f}s")
    
    # Final results
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("EVOLUTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/generations:.1f}s/gen)")
    print(f"  Total evaluations: {total_evals}")
    print(f"  Successful: {successful_evals} ({100*successful_evals/max(1,total_evals):.0f}%)")
    print(f"  Failed extractions: {failed_extractions}")
    print(f"  Failed validations: {failed_validations}")
    print(f"  Improvements: {improvements}")
    print(f"  Final best: {database.best_program.score:.4f}")
    
    print(f"\n{'='*60}")
    print("BEST SOLUTION")
    print(f"{'='*60}")
    print(database.best_program.code)
    
    # Verify
    try:
        exec_globals = {}
        exec(database.best_program.code, exec_globals)
        circles = exec_globals['pack_circles']()
        valid, msg = validate_circle_packing(circles)
        fitness = circle_packing_fitness(circles)
        
        print(f"\n{'='*60}")
        print("VERIFICATION")
        print(f"{'='*60}")
        print(f"  Valid: {valid}")
        print(f"  Circles: {len(circles)}")
        print(f"  Sum of radii: {fitness:.4f}")
        
        if valid:
            print("\n  Circle details:")
            for i, (x, y, r) in enumerate(circles):
                print(f"    {i+1}. center=({x:.3f}, {y:.3f}), radius={r:.3f}")
    except Exception as e:
        print(f"\n  Verification error: {e}")
    
    # Show top diverse solutions
    print(f"\n{'='*60}")
    print("TOP DIVERSE SOLUTIONS")
    print(f"{'='*60}")
    top = sorted(database.all_programs.values(), key=lambda p: p.score, reverse=True)[:5]
    for i, prog in enumerate(top):
        print(f"  {i+1}. Score: {prog.score:.4f} (gen {prog.generation})")
    
    # Show timeout stats if any
    if llm.stats.get("timeouts", 0) > 0:
        print(f"\n⚠️  LLM timeouts: {llm.stats['timeouts']}")
    
    return database.best_program


async def main():
    parser = argparse.ArgumentParser(description="Improved Local Demo")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60, help="LLM timeout in seconds (default: 60)")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    # Set a maximum total time (generations * candidates * timeout * 1.5 safety factor)
    max_total_time = args.generations * args.candidates * args.timeout * 1.5
    print(f"Max estimated time: {max_total_time/60:.0f} minutes")
    print("Press Ctrl+C to stop early and see results\n")
    
    try:
        await run_improved_evolution(
            model=args.model,
            generations=args.generations,
            candidates_per_gen=args.candidates,
            timeout_seconds=args.timeout,
            debug=not args.quiet
        )
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user - showing results so far...")


if __name__ == "__main__":
    asyncio.run(main())
