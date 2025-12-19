"""
AlphaEvolve Controller: Orchestrates the evolutionary optimization loop.

This is the main entry point that ties together:
1. Program Database (evolutionary population)
2. Prompt Sampler (context generation)
3. LLM Ensemble (code generation)
4. Evaluator (fitness assessment)

The evolutionary loop:
1. Sample parents from database
2. Generate prompts with rich context
3. LLM proposes mutations
4. Evaluate candidates
5. Add successful programs to database
6. Repeat

Reference: AlphaEvolve whitepaper Section 2
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
import json

from .database import ProgramDatabase, Program
from .prompt_sampler import PromptSampler, PromptConfig
from .llm_ensemble import (
    LLMEnsemble, LLMConfig, LLMResponse,
    extract_code_from_response, apply_diff, create_openai_ensemble
)
from .evaluator import Evaluator, EvaluationResult


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alphaevolve")


@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary run."""
    # Evolution parameters
    num_generations: int = 100
    population_per_generation: int = 10
    
    # Database parameters
    num_islands: int = 4
    island_capacity: int = 50
    migration_interval: int = 10
    
    # LLM parameters
    batch_size: int = 5
    max_concurrent_llm: int = 3
    force_powerful_every_n: int = 5  # Use powerful model every N generations
    llm_timeout_seconds: float = 120.0  # Timeout for each LLM call
    
    # Evaluation parameters
    max_eval_time_seconds: float = 30.0
    generation_timeout_seconds: float = 300.0  # Max time per generation (5 min)
    
    # Logging
    log_every_n: int = 5
    save_every_n: int = 10
    checkpoint_dir: Optional[str] = None
    
    # Early stopping
    patience: int = 20  # Stop if no improvement for this many generations
    min_improvement: float = 0.001  # Minimum improvement to reset patience


@dataclass
class EvolutionStats:
    """Statistics for the evolutionary run."""
    generation: int = 0
    total_evaluations: int = 0
    successful_evaluations: int = 0
    best_score: float = 0.0
    best_score_generation: int = 0
    improvements: List[Dict] = field(default_factory=list)
    generation_times: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "generation": self.generation,
            "total_evaluations": self.total_evaluations,
            "successful_evaluations": self.successful_evaluations,
            "best_score": self.best_score,
            "best_score_generation": self.best_score_generation,
            "num_improvements": len(self.improvements),
            "avg_generation_time": sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0
        }


class AlphaEvolve:
    """
    Main AlphaEvolve controller.
    
    This class implements the complete evolutionary optimization loop
    as described in the AlphaEvolve whitepaper.
    """
    
    def __init__(
        self,
        evaluator: Evaluator,
        llm_ensemble: Optional[LLMEnsemble] = None,
        config: Optional[EvolutionConfig] = None,
        initial_programs: Optional[List[str]] = None
    ):
        self.evaluator = evaluator
        self.llm_ensemble = llm_ensemble or LLMEnsemble()  # Uses mock by default
        self.config = config or EvolutionConfig()
        
        # Initialize database
        self.database = ProgramDatabase(
            num_islands=self.config.num_islands,
            island_capacity=self.config.island_capacity,
            migration_interval=self.config.migration_interval
        )
        
        # Initialize prompt sampler
        self.prompt_sampler = PromptSampler(
            database=self.database,
            problem_description=evaluator.get_problem_description(),
            function_name=evaluator.get_function_name()
        )
        
        # Statistics
        self.stats = EvolutionStats()
        
        # Callbacks
        self.on_improvement: Optional[Callable[[Program], None]] = None
        self.on_generation: Optional[Callable[[int, EvolutionStats], None]] = None
        
        # Initialize with seed programs
        if initial_programs:
            for code in initial_programs:
                self._evaluate_and_add(code)
    
    def _evaluate_and_add(self, code: str, parent_ids: Optional[List[str]] = None) -> Optional[Program]:
        """Evaluate code and add to database if successful."""
        self.stats.total_evaluations += 1
        
        result = self.evaluator.evaluate(code)
        
        if result.success:
            self.stats.successful_evaluations += 1
            
            program = Program(
                code=code,
                score=result.score,
                metrics=result.metrics,
                parent_ids=parent_ids or []
            )
            
            # Check if this is an improvement
            is_improvement = self.database.add_program(program)
            
            if is_improvement and result.score > self.stats.best_score:
                improvement = result.score - self.stats.best_score
                if improvement >= self.config.min_improvement:
                    self.stats.best_score = result.score
                    self.stats.best_score_generation = self.stats.generation
                    self.stats.improvements.append({
                        "generation": self.stats.generation,
                        "score": result.score,
                        "improvement": improvement
                    })
                    
                    logger.info(f"🎉 New best score: {result.score:.6f} (gen {self.stats.generation})")
                    
                    if self.on_improvement:
                        self.on_improvement(program)
            
            return program
        
        return None
    
    async def _generate_mutations(self, force_powerful: bool = False) -> List[Tuple[str, Optional[str]]]:
        """
        Generate candidate mutations using the LLM ensemble.
        
        Returns:
            List of (code, parent_id) tuples for lineage tracking.
        """
        # Generate prompts
        prompts_data = self.prompt_sampler.generate_batch_prompts(
            n=self.config.batch_size,
            diversity_mode=True
        )
        
        prompts = [p["prompt"] for p in prompts_data]
        
        # Generate responses (pass force_powerful through)
        responses = await self.llm_ensemble.generate_batch(
            prompts,
            max_concurrent=self.config.max_concurrent_llm,
            force_powerful=force_powerful
        )
        
        # Extract code from responses, handling both full code and diffs
        # Track parent_id for lineage
        candidates = []
        for response, prompt_data in zip(responses, prompts_data):
            if response is None:  # Handle failed requests
                continue
                
            extracted = extract_code_from_response(response.content)
            if extracted is None:
                continue
            
            code_type, content = extracted
            parent_id = prompt_data.get("parent_id")
            
            if code_type == "full":
                # Full code replacement - still track parent for lineage
                candidates.append((content, parent_id))
            elif code_type == "diff":
                # Apply diff to parent program
                if parent_id and parent_id in self.database.all_programs:
                    parent = self.database.all_programs[parent_id]
                    applied = apply_diff(parent.code, content)
                    if applied:
                        candidates.append((applied, parent_id))
        
        return candidates
    
    async def _run_generation_internal(self, force_powerful: bool) -> int:
        """Internal generation logic - runs without timeout wrapper."""
        # Generate candidates (now returns (code, parent_id) tuples)
        candidates = await self._generate_mutations(force_powerful)
        
        # Evaluate candidates with lineage tracking
        successes = 0
        for code, parent_id in candidates:
            parent_ids = [parent_id] if parent_id else None
            program = self._evaluate_and_add(code, parent_ids=parent_ids)
            if program:
                successes += 1
        
        return successes, len(candidates)
    
    async def run_generation(self) -> int:
        """
        Run one generation of evolution with timeout protection.
        
        Returns number of successful evaluations.
        """
        gen_start = time.time()
        
        # Determine if we should use powerful model
        force_powerful = (self.stats.generation % self.config.force_powerful_every_n == 0)
        
        try:
            # Run generation with timeout
            result = await asyncio.wait_for(
                self._run_generation_internal(force_powerful),
                timeout=self.config.generation_timeout_seconds
            )
            successes, total_candidates = result
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Generation {self.stats.generation + 1} timed out after {self.config.generation_timeout_seconds}s")
            successes, total_candidates = 0, 0
        except Exception as e:
            logger.error(f"Generation error: {e}")
            successes, total_candidates = 0, 0
        
        # Advance generation
        self.database.step_generation()
        self.stats.generation += 1
        
        gen_time = time.time() - gen_start
        self.stats.generation_times.append(gen_time)
        
        # Logging
        if self.stats.generation % self.config.log_every_n == 0:
            db_stats = self.database.get_statistics()
            logger.info(
                f"Gen {self.stats.generation}: "
                f"best={self.stats.best_score:.6f}, "
                f"successes={successes}/{total_candidates}, "
                f"programs={db_stats['total_programs']}, "
                f"time={gen_time:.2f}s"
            )
        
        # Callbacks
        if self.on_generation:
            self.on_generation(self.stats.generation, self.stats)
        
        return successes
    
    async def evolve(self) -> Program:
        """
        Run the complete evolutionary optimization.
        
        Returns the best program found.
        """
        logger.info(f"Starting AlphaEvolve optimization for {self.config.num_generations} generations")
        logger.info(f"Problem: {self.evaluator.get_function_name()}")
        
        no_improvement_count = 0
        last_best = self.stats.best_score
        
        for gen in range(self.config.num_generations):
            await self.run_generation()
            
            # Early stopping check
            if self.stats.best_score > last_best + self.config.min_improvement:
                no_improvement_count = 0
                last_best = self.stats.best_score
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= self.config.patience:
                logger.info(f"Early stopping: no improvement for {self.config.patience} generations")
                break
            
            # Checkpointing
            if self.config.checkpoint_dir and gen % self.config.save_every_n == 0:
                self.save_checkpoint(f"{self.config.checkpoint_dir}/gen_{gen}.json")
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Evolution complete!")
        logger.info(f"Best score: {self.stats.best_score:.6f}")
        logger.info(f"Total evaluations: {self.stats.total_evaluations}")
        logger.info(f"Successful evaluations: {self.stats.successful_evaluations}")
        logger.info(f"{'='*50}\n")
        
        return self.database.best_program
    
    def save_checkpoint(self, filepath: str):
        """Save current state to checkpoint file."""
        data = {
            "stats": self.stats.to_dict(),
            "config": {
                "num_generations": self.config.num_generations,
                "population_per_generation": self.config.population_per_generation
            },
            "best_program": self.database.best_program.to_dict() if self.database.best_program else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Also save database
        db_path = filepath.replace('.json', '_database.json')
        self.database.save(db_path)
    
    def get_top_programs(self, n: int = 10) -> List[Program]:
        """Get the top N programs by score."""
        all_programs = list(self.database.all_programs.values())
        sorted_programs = sorted(all_programs, key=lambda p: p.score, reverse=True)
        return sorted_programs[:n]
    
    def get_diverse_programs(self, n: int = 10) -> List[Program]:
        """Get a diverse set of top programs from MAP-Elites."""
        return self.database.map_elites.get_elites()[:n]


def run_evolution(
    evaluator: Evaluator,
    llm_ensemble: Optional[LLMEnsemble] = None,
    config: Optional[EvolutionConfig] = None,
    initial_programs: Optional[List[str]] = None
) -> Program:
    """
    Convenience function to run evolution.
    
    Example:
        evaluator = create_circle_packing_evaluator(n_circles=10)
        best = run_evolution(evaluator, num_generations=100)
        print(best.code)
    """
    controller = AlphaEvolve(
        evaluator=evaluator,
        llm_ensemble=llm_ensemble,
        config=config,
        initial_programs=initial_programs
    )
    
    return asyncio.run(controller.evolve())


# CLI interface
def main():
    """Command-line interface for AlphaEvolve."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AlphaEvolve: Evolutionary Code Optimization")
    parser.add_argument("--problem", type=str, default="test", 
                       help="Problem to solve (test, circle_packing, sorting)")
    parser.add_argument("--generations", type=int, default=50,
                       help="Number of generations")
    parser.add_argument("--provider", type=str, default="mock",
                       help="LLM provider (mock, openai, anthropic, google, ollama)")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name for the provider")
    parser.add_argument("--output", type=str, default="results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Import problem modules
    from . import problems
    
    # Get evaluator
    if args.problem == "test":
        from .evaluator import create_test_evaluator
        evaluator = create_test_evaluator()
        initial_programs = [
            "def sum_of_squares(nums):\n    return sum(x**2 for x in nums)"
        ]
    elif args.problem == "circle_packing":
        evaluator = problems.create_circle_packing_evaluator()
        initial_programs = problems.get_circle_packing_seeds()
    else:
        raise ValueError(f"Unknown problem: {args.problem}")
    
    # Create LLM ensemble
    if args.provider == "mock":
        llm_ensemble = LLMEnsemble()
    elif args.provider == "openai":
        llm_ensemble = create_openai_ensemble(
            fast_model=args.model or "gpt-4o-mini",
            powerful_model="gpt-4o"
        )
    else:
        llm_ensemble = LLMEnsemble()  # Default to mock
    
    # Create config
    config = EvolutionConfig(
        num_generations=args.generations,
        checkpoint_dir="checkpoints"
    )
    
    # Run evolution
    best = run_evolution(
        evaluator=evaluator,
        llm_ensemble=llm_ensemble,
        config=config,
        initial_programs=initial_programs
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            "problem": args.problem,
            "best_score": best.score,
            "best_code": best.code,
            "metrics": best.metrics
        }, f, indent=2)
    
    print(f"\nBest solution saved to {args.output}")
    print(f"Score: {best.score:.6f}")
    print(f"\nCode:\n{best.code}")


if __name__ == "__main__":
    main()
