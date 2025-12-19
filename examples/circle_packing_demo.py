#!/usr/bin/env python3
"""
AlphaEvolve-Mini: Complete Example

This example demonstrates all components of AlphaEvolve working together
to solve the circle packing problem.

Run with:
    python examples/circle_packing_demo.py

Or with a real LLM:
    OPENAI_API_KEY=... python examples/circle_packing_demo.py --provider openai
"""

import asyncio
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    AlphaEvolve,
    EvolutionConfig,
    Program,
    LLMEnsemble,
    LLMConfig,
    create_circle_packing_evaluator,
    get_circle_packing_seeds,
    create_openai_ensemble,
    create_anthropic_ensemble,
)


def visualize_circles(circles, title="Circle Packing"):
    """Visualize the circle packing solution using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Draw unit square
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))
        
        # Draw circles
        colors = plt.cm.viridis([i/len(circles) for i in range(len(circles))])
        for i, (x, y, r) in enumerate(circles):
            circle = patches.Circle((x, y), r, fill=True, facecolor=colors[i], 
                                    edgecolor='black', alpha=0.7)
            ax.add_patch(circle)
        
        total_radius = sum(r for _, _, r in circles)
        ax.set_title(f"{title}\nSum of radii: {total_radius:.4f}")
        
        plt.savefig("circle_packing_result.png", dpi=150, bbox_inches='tight')
        print("Visualization saved to circle_packing_result.png")
        
    except ImportError:
        print("matplotlib not installed, skipping visualization")


def on_improvement(program: Program):
    """Callback when a new best solution is found."""
    print(f"\n{'='*50}")
    print(f"🎉 NEW BEST SOLUTION!")
    print(f"Score: {program.score:.6f}")
    print(f"Generation: {program.generation}")
    print(f"{'='*50}\n")


async def main():
    parser = argparse.ArgumentParser(description="AlphaEvolve Circle Packing Demo")
    parser.add_argument("--circles", type=int, default=10, 
                       help="Number of circles to pack")
    parser.add_argument("--generations", type=int, default=30,
                       help="Number of generations to run")
    parser.add_argument("--provider", type=str, default="mock",
                       choices=["mock", "openai", "anthropic"],
                       help="LLM provider to use")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize the best solution")
    
    args = parser.parse_args()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║           AlphaEvolve-Mini: Circle Packing Demo               ║
╠═══════════════════════════════════════════════════════════════╣
║  This demo shows evolutionary code optimization in action.    ║
║  We'll evolve algorithms to pack circles in a unit square.    ║
╚═══════════════════════════════════════════════════════════════╝

Configuration:
  - Circles: {args.circles}
  - Generations: {args.generations}
  - LLM Provider: {args.provider}
""")
    
    # Create evaluator and seeds
    evaluator = create_circle_packing_evaluator(n_circles=args.circles)
    seeds = get_circle_packing_seeds(n_circles=args.circles)
    
    # Create LLM ensemble
    if args.provider == "mock":
        print("Using mock LLM (for testing without API calls)")
        llm_ensemble = LLMEnsemble()  # Uses mock by default
    elif args.provider == "openai":
        print("Using OpenAI GPT-4")
        llm_ensemble = create_openai_ensemble()
    elif args.provider == "anthropic":
        print("Using Anthropic Claude")
        llm_ensemble = create_anthropic_ensemble()
    
    # Create configuration
    config = EvolutionConfig(
        num_generations=args.generations,
        population_per_generation=5,
        num_islands=3,
        batch_size=3,
        log_every_n=5,
        patience=15
    )
    
    # Create optimizer
    optimizer = AlphaEvolve(
        evaluator=evaluator,
        llm_ensemble=llm_ensemble,
        config=config,
        initial_programs=seeds
    )
    
    # Set callback
    optimizer.on_improvement = on_improvement
    
    # Run evolution
    print("\nStarting evolution...\n")
    best = await optimizer.evolve()
    
    # Results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Score: {best.score:.6f}")
    print(f"Total Evaluations: {optimizer.stats.total_evaluations}")
    print(f"Successful Evaluations: {optimizer.stats.successful_evaluations}")
    print(f"\nBest Solution Code:")
    print("-" * 40)
    print(best.code)
    print("-" * 40)
    
    # Try to run and visualize
    if args.visualize:
        try:
            # Execute the best solution to get circles
            exec_globals = {}
            exec(best.code, exec_globals)
            circles = exec_globals['pack_circles']()
            
            print(f"\nCircles packed: {len(circles)}")
            print(f"Sum of radii: {sum(r for _, _, r in circles):.6f}")
            
            visualize_circles(circles, f"Best Solution (Gen {best.generation})")
            
        except Exception as e:
            print(f"Could not visualize: {e}")
    
    # Show LLM statistics
    print(f"\nLLM Statistics:")
    stats = llm_ensemble.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return best


if __name__ == "__main__":
    best = asyncio.run(main())
