# AlphaEvolve-Mini

**A complete educational implementation of Google DeepMind's AlphaEvolve system for evolutionary code optimization using LLMs**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Runs Locally](https://img.shields.io/badge/runs-100%25%20local-green.svg)](#quick-start)

---

## 🚀 Quick Start

Run evolutionary code optimization locally in under 5 minutes:

```bash
# 1. Install Ollama (one-time setup)
curl -fsSL https://ollama.ai/install.sh | sh    # Linux
# or: brew install ollama                        # macOS
# or: Download from https://ollama.ai            # Windows

# 2. Start Ollama and pull a coding model
ollama serve &
ollama pull qwen2.5-coder:7b    # Best quality (8GB RAM)
# or: ollama pull qwen2.5-coder:1.5b  # Faster (4GB RAM)

# 3. Clone and run
git clone https://github.com/yourusername/alphaevolve-mini.git
cd alphaevolve-mini
pip install httpx
python examples/local_demo_v2.py --generations 30
```

Watch evolution discover better solutions:
```
Evaluating seeds...
  Seed 1: 1.2000
  Seed 2: 0.4210
  Seed 3: 1.2100

Initial best: 1.2100
============================================================
Starting evolution...
============================================================
  [Gen 1.1] Score: 1.2300 🎉 NEW BEST!
  [Gen 5.2] Score: 1.1900 
  [Gen 13.1] Score: 1.2500 🎉 NEW BEST!

============================================================
EVOLUTION COMPLETE
============================================================
  Final best: 1.2500 (+3.3% improvement)
```

---

## Overview

AlphaEvolve-Mini captures the essential architecture of [Google DeepMind's AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/). It combines:

- **Large Language Models** for intelligent code mutations
- **Evolutionary algorithms** (MAP-Elites + Island Model) for population diversity
- **Automated evaluation** with sandboxed code execution

```
┌─────────────────────────────────────────────────────────────────┐
│                      AlphaEvolve-Mini                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │   Program   │────▶│   Prompt    │────▶│     LLM     │      │
│   │  Database   │     │  Sampler    │     │  Ensemble   │      │
│   │             │     │             │     │             │      │
│   │ MAP-Elites  │     │ Rich context│     │ Local/Cloud │      │
│   │ + Islands   │     │ + Lineage   │     │ Models      │      │
│   └─────────────┘     └─────────────┘     └──────┬──────┘      │
│         ▲                                        │              │
│         │                                        ▼              │
│   ┌─────┴─────┐                          ┌─────────────┐       │
│   │           │◀─────────────────────────│  Evaluator  │       │
│   │Controller │                          │             │       │
│   │           │                          │  Sandboxed  │       │
│   └───────────┘                          │  Execution  │       │
│                                          └─────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Matters

AlphaEvolve achieved remarkable results at Google:
- **0.7% recovery of worldwide compute** through better scheduling
- **23% faster matrix multiplication kernels** for Gemini training
- **New mathematical discoveries** improving on 50-year-old algorithms

This implementation lets you understand and experiment with the core techniques.

---

## Installation

### Option A: Ollama (Recommended)

Ollama is the simplest way to run local LLMs. It handles model management and optimization automatically.

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh    # Linux
brew install ollama                              # macOS

# Start the service
ollama serve

# Pull a model (choose based on your hardware)
ollama pull qwen2.5-coder:7b      # 8GB RAM - Best quality
ollama pull qwen2.5-coder:1.5b    # 4GB RAM - Good balance
ollama pull deepseek-coder:1.3b   # 2GB RAM - Fastest

# Install Python dependencies
git clone https://github.com/yourusername/alphaevolve-mini.git
cd alphaevolve-mini
pip install -r requirements.txt
```

### Option B: Hugging Face Transformers

Run models directly without a separate service. Better for GPU users who want more control.

```bash
git clone https://github.com/yourusername/alphaevolve-mini.git
cd alphaevolve-mini
pip install -r requirements-local.txt

# Run with Hugging Face
python examples/local_demo_v2.py --provider huggingface \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct
```

### Option C: Cloud APIs (Optional)

For access to larger models:

```bash
pip install openai anthropic google-generativeai

# Set API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### Model Recommendations

| Hardware | RAM | Recommended Model | Command |
|----------|-----|-------------------|---------|
| **Low-end** | 4GB | deepseek-coder:1.3b | `ollama pull deepseek-coder:1.3b` |
| **Mid-range** | 8GB | qwen2.5-coder:1.5b | `ollama pull qwen2.5-coder:1.5b` |
| **High-end** | 16GB+ | qwen2.5-coder:7b | `ollama pull qwen2.5-coder:7b` |
| **GPU (8GB+)** | - | qwen2.5-coder:7b | `ollama pull qwen2.5-coder:7b` |
| **Apple Silicon** | 16GB+ | qwen2.5-coder:7b | `ollama pull qwen2.5-coder:7b` |

### Verify Installation

```bash
python setup_local.py --check
```

This displays:
```
============================================================
  System Check
============================================================

  ✓ Python: 3.11.5
  
  Python packages:
    ✓ httpx: installed
  
  Local LLM:
    ✓ Ollama: Running with 1 models
    Available models:
      - qwen2.5-coder:7b
  
  Hardware:
    ✓ Apple MPS: Available

============================================================
  Summary
============================================================
     Ready to run! Use:
     python examples/local_demo_v2.py
```

---

## Usage

### Command Line

```bash
# Basic run with defaults
python examples/local_demo_v2.py

# Specify model and generations
python examples/local_demo_v2.py --model qwen2.5-coder:7b --generations 50

# Different problem
python examples/local_demo_v2.py --problem sorting

# More candidates per generation (slower but more thorough)
python examples/local_demo_v2.py --candidates 6

# Quiet mode (less output)
python examples/local_demo_v2.py --quiet
```

### Programmatic Usage

```python
import asyncio
from alphaevolve_mini import (
    AlphaEvolve,
    EvolutionConfig,
    create_circle_packing_evaluator,
    get_circle_packing_seeds
)
from alphaevolve_mini.core.local_llm import LocalLLMEnsemble

async def main():
    # Create the optimization problem
    evaluator = create_circle_packing_evaluator(n_circles=10)
    seeds = get_circle_packing_seeds(n_circles=10)
    
    # Create local LLM ensemble
    llm = LocalLLMEnsemble(
        provider="ollama",
        model_name="qwen2.5-coder:7b"
    )
    
    # Configure evolution
    config = EvolutionConfig(
        num_generations=50,
        population_per_generation=6,
        num_islands=4,
        patience=20  # Early stopping if no improvement
    )
    
    # Run evolution
    optimizer = AlphaEvolve(
        evaluator=evaluator,
        llm_ensemble=llm,
        config=config,
        initial_programs=seeds
    )
    
    # Add callbacks
    optimizer.on_improvement = lambda p, s: print(f"🎉 New best: {s:.4f}")
    
    best = await optimizer.evolve()
    print(f"\nFinal best score: {best.score}")
    print(best.code)

asyncio.run(main())
```

### Using Cloud APIs

```python
from alphaevolve_mini import create_openai_ensemble, create_anthropic_ensemble

# OpenAI (GPT-4)
llm = create_openai_ensemble()

# Anthropic (Claude)
llm = create_anthropic_ensemble()

# Google (Gemini)
llm = create_google_ensemble()

# Mixed ensemble (80% fast, 20% powerful)
llm = LLMEnsemble(configs=[
    LLMConfig(provider="openai", model="gpt-4o-mini", weight=0.8),
    LLMConfig(provider="openai", model="gpt-4o", weight=0.2),
])
```

---

## Architecture

### Core Components

#### 1. Program Database (`core/database.py`)

Manages the population using two complementary strategies:

**MAP-Elites Grid**: Quality-diversity algorithm that maintains elite solutions across behavioral dimensions (code complexity, runtime, etc.). This prevents convergence to a single local optimum.

**Island Model**: Four independent populations evolving in parallel with periodic migration. Balances exploration (island diversity) with exploitation (sharing good solutions).

```python
from alphaevolve_mini import ProgramDatabase, Program

# Create database with 4 islands
db = ProgramDatabase(num_islands=4, island_capacity=50)

# Add a program
program = Program(code="def solve(): ...", score=0.85)
db.add_program(program)

# Sample parents for mutation
sample = db.sample_for_prompt()
# Returns: {"best": Program, "parents": [Program, ...], "diverse": [...]}

# Trigger migration between islands
db.step_generation()
```

#### 2. Prompt Sampler (`core/prompt_sampler.py`, `core/compact_prompts.py`)

Constructs rich context for LLM mutations:

- Problem description and objectives
- Parent program(s) to mutate
- Evolutionary lineage showing successful patterns
- Diverse examples from different score tiers
- Specific mutation strategies

**Standard prompts** for large models (7B+):
```python
from alphaevolve_mini import PromptSampler

sampler = PromptSampler(
    database=db,
    problem_description="Pack circles to maximize area",
    function_name="pack_circles"
)
prompt = sampler.generate_prompt(strategy="algorithmic_rewrite")
```

**Compact prompts** for smaller models (1.5B-7B):
```python
from alphaevolve_mini.core.compact_prompts import CompactPromptSampler

sampler = CompactPromptSampler(
    database=db,
    problem_description="Pack circles to maximize area",
    function_name="pack_circles"
)
# Generates shorter prompts optimized for limited context windows
```

#### 3. LLM Ensemble (`core/llm_ensemble.py`, `core/local_llm.py`)

Multi-provider interface supporting both local and cloud models:

**Local Providers:**
- **Ollama** - Easiest setup, great model selection
- **Hugging Face Transformers** - Direct model loading, GPU support
- **llama.cpp** - Most efficient CPU inference

**Cloud Providers:**
- OpenAI (GPT-4, GPT-4o-mini)
- Anthropic (Claude Sonnet, Haiku)
- Google (Gemini Pro, Flash)

```python
from alphaevolve_mini.core.local_llm import LocalLLMEnsemble

# Local with Ollama
llm = LocalLLMEnsemble(provider="ollama", model_name="qwen2.5-coder:7b")

# Generate with temperature variation for diversity
responses = await llm.generate_diverse(prompt, n=4, temperature_range=(0.5, 1.0))
```

#### 4. Evaluator (`core/evaluator.py`)

Safe sandboxed code execution with:

- **Restricted environment**: Only whitelisted modules (math, random, itertools, etc.)
- **Security**: Blocks file I/O, network, dangerous imports
- **AST validation**: Checks for dangerous patterns before execution
- **Timeout enforcement**: Configurable time limits (default 5s)

```python
from alphaevolve_mini import MaximizationEvaluator

evaluator = MaximizationEvaluator(
    fitness_function=lambda code: evaluate_packing(code),
    timeout=5.0,
    allowed_modules=["math", "random"]
)

result = evaluator.evaluate(code)
# Returns: EvaluationResult(success=True, score=0.85, metrics={...})
```

#### 5. Controller (`core/controller.py`)

Orchestrates the evolutionary loop:

1. Sample parents from database
2. Generate prompts with rich context
3. LLM proposes mutations (batch async)
4. Evaluate candidates in sandbox
5. Add successful programs to database
6. Step generation (triggers migration)
7. Repeat until convergence

```python
from alphaevolve_mini import AlphaEvolve, EvolutionConfig

config = EvolutionConfig(
    num_generations=100,
    population_per_generation=8,
    num_islands=4,
    batch_size=4,
    patience=25
)

optimizer = AlphaEvolve(evaluator, llm, config, initial_programs=seeds)
best = await optimizer.evolve()
```

---

## Example Problems

### Circle Packing (Flagship)

Pack N non-overlapping circles in a unit square to maximize the sum of radii. This is the problem featured in the AlphaEvolve paper.

```python
from alphaevolve_mini import create_problem

evaluator, seeds = create_problem("circle_packing", n_circles=10)
```

**Scoring:**
- Sum of all radii (higher is better)
- Validation: no overlaps, all circles within [0,1]×[0,1]

**Expected results:**
| Circles | Seed Score | Good | Excellent |
|---------|------------|------|-----------|
| 8 | ~1.21 | 1.25+ | 1.35+ |
| 10 | ~1.35 | 1.45+ | 1.55+ |
| 26 | ~2.40 | 2.55+ | 2.63+ |

### Sorting Optimization

Evolve efficient sorting algorithms:

```python
evaluator, seeds = create_problem("sorting")
```

**Scoring:**
- Correctness (must sort correctly)
- Comparison count (fewer is better)
- Runtime efficiency

### Function Discovery

Discover mathematical functions that fit target data:

```python
evaluator, seeds = create_problem("function_discovery", 
    target_function=lambda x: x**2 + 2*x + 1)
```

### Custom Problems

Create your own optimization problems:

```python
from alphaevolve_mini import MaximizationEvaluator

def evaluate_my_problem(code: str) -> float:
    """Execute code and return fitness score."""
    exec_globals = {}
    exec(code, exec_globals)
    result = exec_globals['solve']()
    return compute_fitness(result)

evaluator = MaximizationEvaluator(
    fitness_function=evaluate_my_problem,
    timeout=10.0
)

seeds = [
    "def solve(): return baseline_solution()",
    "def solve(): return alternative_approach()",
]
```

---

## Project Structure

```
alphaevolve-mini/
├── core/
│   ├── __init__.py          # Public API exports
│   ├── database.py          # MAP-Elites + Island population (13KB)
│   ├── prompt_sampler.py    # Rich context generation (10KB)
│   ├── compact_prompts.py   # Optimized prompts for small models (8KB)
│   ├── llm_ensemble.py      # Cloud LLM providers (14KB)
│   ├── local_llm.py         # Local LLM providers (19KB)
│   ├── evaluator.py         # Sandboxed execution (17KB)
│   ├── controller.py        # Evolution loop (14KB)
│   └── problems.py          # Example problems (13KB)
├── examples/
│   ├── local_demo.py        # Basic local demo
│   ├── local_demo_v2.py     # Improved demo with debugging
│   ├── test_llm.py          # LLM diagnostic tool
│   └── circle_packing_demo.py
├── tests/
│   └── test_core.py         # Comprehensive test suite
├── setup.py                 # Package installation
├── setup_local.py           # Local environment setup script
├── requirements.txt         # Minimal dependencies
├── requirements-local.txt   # Local LLM dependencies
├── Dockerfile               # Container setup
└── README.md
```

---

## Configuration Reference

### EvolutionConfig

```python
EvolutionConfig(
    num_generations=100,        # Maximum generations to run
    population_per_generation=8, # New candidates per generation
    num_islands=4,              # Parallel populations
    batch_size=4,               # Concurrent LLM calls
    patience=25,                # Early stop after N gens without improvement
    migration_interval=5,       # Generations between island migrations
    migration_rate=0.1,         # Fraction of population to migrate
)
```

### LocalLLMEnsemble

```python
LocalLLMEnsemble(
    provider="ollama",          # "ollama", "huggingface", "llamacpp"
    model_name="qwen2.5-coder:7b",
    device="auto",              # "auto", "cpu", "cuda", "mps"
    quantization=None,          # "4bit", "8bit", None
)
```

### Evaluator Options

```python
MaximizationEvaluator(
    fitness_function=fn,
    timeout=5.0,                # Seconds before timeout
    allowed_modules=["math", "random", "itertools"],
)
```

---


## How It Compares to Full AlphaEvolve

The core algorithms are the same:
- ✅ MAP-Elites quality-diversity
- ✅ Island-based parallel evolution  
- ✅ Rich prompt context with lineage
- ✅ Sandboxed evaluation
- ✅ Evolutionary population management


---

## Extending the System

### Adding New LLM Providers

```python
from alphaevolve_mini.core.llm_ensemble import LLMProvider, LLMConfig, LLMResponse

class MyCustomProvider(LLMProvider):
    async def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        # Your implementation
        response = await my_api_call(prompt, config.temperature)
        return LLMResponse(
            content=response.text,
            model=config.model,
            usage={"prompt_tokens": ..., "completion_tokens": ...},
            latency_ms=response.latency
        )
```

### Adding New Problems

```python
from alphaevolve_mini import Evaluator, EvaluationResult

class MyProblemEvaluator(Evaluator):
    def evaluate(self, code: str) -> EvaluationResult:
        try:
            # Run in sandbox
            result = self.sandbox.execute(code, timeout=5.0)
            
            # Compute fitness
            score = self.compute_fitness(result)
            
            return EvaluationResult(
                success=True,
                score=score,
                metrics={"custom_metric": value}
            )
        except Exception as e:
            return EvaluationResult(success=False, error=str(e))
```

### Custom Prompt Strategies

```python
from alphaevolve_mini import PromptSampler

class MyPromptSampler(PromptSampler):
    def generate_prompt(self, parent, strategy="default"):
        if strategy == "my_strategy":
            return self._my_custom_prompt(parent)
        return super().generate_prompt(parent, strategy)
    
    def _my_custom_prompt(self, parent):
        return f"""
        My custom prompt format...
        Current code: {parent.code}
        """
```

---

## References

- [AlphaEvolve Blog Post](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) - DeepMind's announcement
- [AlphaEvolve Paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) - Technical details
- [MAP-Elites Paper](https://arxiv.org/abs/1504.04909) - Quality-diversity algorithm
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) - Recommended coding model

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Areas of interest:

- Additional example problems
- New LLM provider integrations
- Performance optimizations
- Documentation improvements
- Test coverage

Please open an issue to discuss major changes before submitting a PR.
