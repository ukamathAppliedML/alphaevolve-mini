"""
AlphaEvolve-Mini: Educational Implementation of AlphaEvolve

A minimal but complete implementation of Google DeepMind's AlphaEvolve system
for evolutionary code optimization using LLMs.

Components:
- database: MAP-Elites + Island-based evolutionary population
- prompt_sampler: Rich context generation for LLM mutations
- compact_prompts: Optimized prompts for small local models
- llm_ensemble: Multi-model LLM interface (cloud APIs)
- local_llm: Local LLM support (Ollama, HuggingFace, llama.cpp)
- evaluator: Sandboxed code execution and fitness evaluation
- controller: Main evolutionary loop orchestration
- problems: Example optimization problems

Usage (Local - Recommended):
    from alphaevolve_mini import AlphaEvolve, create_problem
    from alphaevolve_mini.core.local_llm import LocalLLMEnsemble
    
    # Create a problem
    evaluator, seeds = create_problem("circle_packing", n_circles=10)
    
    # Create local LLM ensemble (requires Ollama)
    llm = LocalLLMEnsemble(provider="ollama", model_name="qwen2.5-coder:1.5b")
    
    # Create and run the optimizer
    optimizer = AlphaEvolve(evaluator, llm_ensemble=llm, initial_programs=seeds)
    best = asyncio.run(optimizer.evolve())
    
    print(f"Best score: {best.score}")
    print(best.code)

Reference:
    AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms
    https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
"""

__version__ = "0.1.3"
__author__ = "Educational Implementation"

from .database import Program, ProgramDatabase, Island, MAPElitesGrid
from .prompt_sampler import PromptSampler, PromptConfig
from .compact_prompts import CompactPromptSampler, UltraCompactPromptSampler
from .llm_ensemble import (
    LLMEnsemble, 
    LLMConfig, 
    LLMResponse,
    create_openai_ensemble,
    create_anthropic_ensemble,
    create_google_ensemble,
    create_ollama_ensemble
)
from .local_llm import (
    LocalLLMEnsemble,
    OllamaLocalProvider,
    HuggingFaceLocalProvider,
    create_local_ensemble,
    check_ollama_status,
    setup_ollama_model,
    HARDWARE_RECOMMENDATIONS
)
from .evaluator import (
    Evaluator,
    EvaluationResult,
    FunctionOptimizationEvaluator,
    MaximizationEvaluator,
    CascadeEvaluator,
    CodeSandbox
)
from .controller import AlphaEvolve, EvolutionConfig, EvolutionStats, run_evolution
from .problems import (
    create_problem,
    list_problems,
    create_circle_packing_evaluator,
    create_sorting_evaluator,
    create_function_discovery_evaluator
)

__all__ = [
    # Main classes
    "AlphaEvolve",
    "EvolutionConfig",
    "EvolutionStats",
    
    # Database
    "Program",
    "ProgramDatabase",
    "Island",
    "MAPElitesGrid",
    
    # Prompt sampling
    "PromptSampler",
    "PromptConfig",
    "CompactPromptSampler",
    "UltraCompactPromptSampler",
    
    # Cloud LLM
    "LLMEnsemble",
    "LLMConfig",
    "LLMResponse",
    "create_openai_ensemble",
    "create_anthropic_ensemble",
    "create_google_ensemble",
    "create_ollama_ensemble",
    
    # Local LLM
    "LocalLLMEnsemble",
    "OllamaLocalProvider",
    "HuggingFaceLocalProvider",
    "create_local_ensemble",
    "check_ollama_status",
    "setup_ollama_model",
    "HARDWARE_RECOMMENDATIONS",
    
    # Evaluation
    "Evaluator",
    "EvaluationResult",
    "FunctionOptimizationEvaluator",
    "MaximizationEvaluator",
    "CascadeEvaluator",
    "CodeSandbox",
    
    # Problems
    "create_problem",
    "list_problems",
    "create_circle_packing_evaluator",
    "create_sorting_evaluator",
    "create_function_discovery_evaluator",
    
    # Convenience
    "run_evolution",
]
