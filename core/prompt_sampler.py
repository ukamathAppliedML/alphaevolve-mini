"""
Prompt Sampler: Constructs rich prompts for the LLM ensemble.

The prompt sampler is critical for AlphaEvolve's success. It provides:
1. Problem context and objectives
2. Parent programs to mutate
3. Evolutionary history (lineage) showing what worked
4. Diverse examples for inspiration
5. Specific mutation instructions

Reference: Section 2.2 of AlphaEvolve whitepaper
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import random
from .database import Program, ProgramDatabase


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    include_problem_description: bool = True
    include_parent_code: bool = True
    include_lineage: bool = True
    include_diverse_examples: bool = True
    include_scores: bool = True
    include_mutation_hints: bool = True
    max_examples: int = 5
    lineage_depth: int = 3
    output_format: str = "full"  # "diff" or "full" - default to full for better local model support


class PromptSampler:
    """
    Generates prompts for the LLM to propose code mutations.
    
    The key insight from AlphaEvolve is that rich context dramatically
    improves the quality of LLM-generated mutations. The prompt includes:
    - What we're trying to optimize
    - Examples of good (and bad) solutions
    - The evolutionary history showing successful mutation patterns
    """
    
    def __init__(
        self, 
        database: ProgramDatabase,
        problem_description: str,
        function_name: str,
        config: Optional[PromptConfig] = None
    ):
        self.database = database
        self.problem_description = problem_description
        self.function_name = function_name
        self.config = config or PromptConfig()
        
        # Mutation strategies to suggest to the LLM
        self.mutation_strategies = [
            "Make a small incremental change to improve efficiency",
            "Try a fundamentally different algorithmic approach",
            "Optimize the inner loop or hot path",
            "Simplify the code while maintaining correctness",
            "Add a clever optimization or pruning strategy",
            "Combine ideas from multiple parent solutions",
            "Fix a bug or edge case you notice",
            "Improve numerical stability or precision",
            "Reduce memory usage",
            "Parallelize or vectorize operations",
        ]
    
    def _format_program(self, program: Program, include_score: bool = True) -> str:
        """Format a program for inclusion in the prompt."""
        result = f"```python\n{program.code}\n```"
        if include_score and self.config.include_scores:
            result += f"\n# Score: {program.score:.6f}"
            if program.metrics:
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in program.metrics.items())
                result += f" | Metrics: {metrics_str}"
        return result
    
    def _format_lineage(self, programs: List[Program]) -> str:
        """Format evolutionary lineage to show progression."""
        if not programs:
            return ""
        
        result = ["## Evolutionary History (what mutations worked):"]
        for i, prog in enumerate(programs):
            result.append(f"\n### Generation {prog.generation} (Score: {prog.score:.6f})")
            result.append(f"```python\n{prog.code}\n```")
        
        return "\n".join(result)
    
    def _select_mutation_strategy(self) -> str:
        """Select a mutation strategy, biased by what's worked."""
        # Could be made smarter by tracking which strategies led to improvements
        return random.choice(self.mutation_strategies)
    
    def generate_prompt(
        self, 
        parent: Optional[Program] = None,
        strategy_hint: Optional[str] = None
    ) -> str:
        """
        Generate a complete prompt for the LLM.
        
        This is the core function that assembles all context needed
        for the LLM to propose intelligent mutations.
        """
        parts = []
        
        # 1. System context and role
        parts.append("""You are an expert algorithm designer participating in an evolutionary optimization process. Your task is to propose improvements to code that will achieve higher scores on the given objective.

Be creative but precise. Small, targeted changes often work better than large rewrites.
""")
        
        # 2. Problem description
        if self.config.include_problem_description:
            parts.append(f"""## Problem Description
{self.problem_description}

You must implement a function named `{self.function_name}` that will be evaluated automatically.
""")
        
        # 3. Sample from database
        sample = self.database.sample_for_prompt()
        
        # 4. Current best solution
        if sample["best"]:
            parts.append(f"""## Current Best Solution (Score: {sample['best'].score:.6f})
{self._format_program(sample['best'], include_score=False)}
""")
        
        # 5. Parent to mutate (primary focus)
        if parent is None and sample["parents"]:
            parent = sample["parents"][0]
        
        if parent and self.config.include_parent_code:
            parts.append(f"""## Parent Solution to Improve
{self._format_program(parent)}
""")
        
        # 6. Evolutionary lineage
        if parent and self.config.include_lineage:
            lineage = self.database.get_lineage(parent.id, self.config.lineage_depth)
            if lineage:
                parts.append(self._format_lineage(lineage))
        
        # 7. Diverse examples for inspiration
        if self.config.include_diverse_examples and sample["diverse"]:
            parts.append("\n## Other Solutions for Inspiration")
            for prog in sample["diverse"][:self.config.max_examples]:
                if prog.id != (parent.id if parent else None):
                    parts.append(self._format_program(prog))
        
        # 8. Mutation instructions
        strategy = strategy_hint or self._select_mutation_strategy()
        
        if self.config.include_mutation_hints:
            parts.append(f"""
## Your Task
{strategy}

Propose an improved version of the parent solution. Consider:
- What patterns appear in high-scoring solutions?
- What might be limiting the current approach?
- Are there edge cases being handled inefficiently?
""")
        
        # 9. Output format instructions
        if self.config.output_format == "diff":
            parts.append("""
## Output Format
Provide your changes as a unified diff that can be applied to the parent code.
Start your diff with ```diff and end with ```.

Example:
```diff
--- a/solution.py
+++ b/solution.py
@@ -1,5 +1,6 @@
 def solve(x):
-    return x * 2
+    # More efficient approach
+    return x << 1
```

If you prefer, you can also provide the complete new solution:
```python
# Your complete solution here
```
""")
        else:
            parts.append(f"""
## Output Format
Provide the complete improved solution as a Python code block.
The function must be named `{self.function_name}`.

```python
# Your complete solution here
```
""")
        
        return "\n".join(parts)
    
    def generate_batch_prompts(
        self, 
        n: int = 5,
        diversity_mode: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple prompts for parallel LLM calls.
        
        In diversity mode, we vary:
        - Which parent to mutate
        - Which mutation strategy to suggest
        - How much context to include
        """
        prompts = []
        sample = self.database.sample_for_prompt()
        
        # Get diverse parents
        parents = sample["parents"] + sample["diverse"]
        if sample["best"] and sample["best"] not in parents:
            parents.insert(0, sample["best"])
        
        for i in range(n):
            # Select parent - mix of exploitation (best) and exploration (diverse)
            if i == 0 and sample["best"]:
                parent = sample["best"]
            elif parents:
                parent = random.choice(parents)
            else:
                parent = None
            
            # Vary the strategy
            strategy = random.choice(self.mutation_strategies)
            
            # Generate prompt
            prompt = self.generate_prompt(parent=parent, strategy_hint=strategy)
            
            prompts.append({
                "prompt": prompt,
                "parent_id": parent.id if parent else None,
                "strategy": strategy,
                "index": i
            })
        
        return prompts


class InsightGenerator:
    """
    Generates insights about the evolutionary process for meta-learning.
    
    AlphaEvolve uses a separate LLM call to analyze successful mutations
    and generate insights that can guide future evolution.
    """
    
    def __init__(self, database: ProgramDatabase):
        self.database = database
    
    def generate_insight_prompt(self, recent_improvements: List[Program]) -> str:
        """Generate a prompt asking the LLM to analyze what's working."""
        
        if not recent_improvements:
            return ""
        
        parts = ["""## Analyze Recent Improvements

Look at these recent successful mutations and identify patterns:
"""]
        
        for prog in recent_improvements[:5]:
            lineage = self.database.get_lineage(prog.id, depth=2)
            if len(lineage) >= 2:
                parent = lineage[1]
                parts.append(f"""
### Improvement: {parent.score:.4f} → {prog.score:.4f}

Before:
```python
{parent.code}
```

After:
```python
{prog.code}
```
""")
        
        parts.append("""
## Your Analysis

1. What patterns do you see in successful mutations?
2. What strategies seem to work well for this problem?
3. What should we try next?

Provide your analysis as JSON:
```json
{
    "successful_patterns": ["pattern1", "pattern2"],
    "recommended_strategies": ["strategy1", "strategy2"],
    "hypotheses": ["hypothesis1", "hypothesis2"]
}
```
""")
        
        return "\n".join(parts)
