"""
Compact Prompt Sampler for Small Local LLMs

Small models (1-7B parameters) have:
- Limited context windows (2K-8K tokens)
- Less ability to follow complex instructions
- Better performance with concise, focused prompts

This module provides optimized prompts for local models.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .database import Program, ProgramDatabase


@dataclass
class CompactPromptConfig:
    """Configuration for compact prompts."""
    max_prompt_tokens: int = 1500  # Leave room for generation
    include_examples: int = 1  # Number of examples to include
    include_best: bool = True
    concise_instructions: bool = True


class CompactPromptSampler:
    """
    Generates concise prompts optimized for small local LLMs.
    
    Key differences from full PromptSampler:
    - Much shorter system instructions
    - Fewer examples (1-2 instead of 5+)
    - Direct mutation requests without lengthy context
    - No diff format (full code only for simplicity)
    """
    
    def __init__(
        self,
        database: ProgramDatabase,
        problem_description: str,
        function_name: str,
        config: Optional[CompactPromptConfig] = None
    ):
        self.database = database
        self.problem_description = self._truncate(problem_description, 500)
        self.function_name = function_name
        self.config = config or CompactPromptConfig()
        
        # Compact mutation hints
        self.mutation_hints = [
            "optimize the algorithm",
            "improve efficiency",
            "try a different approach",
            "fix any bugs",
            "simplify the code",
            "add better edge case handling",
        ]
    
    def _truncate(self, text: str, max_chars: int) -> str:
        """Truncate text to max characters."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars-3] + "..."
    
    def _format_code(self, code: str, max_lines: int = 30) -> str:
        """Format code, truncating if too long."""
        lines = code.split('\n')
        if len(lines) > max_lines:
            return '\n'.join(lines[:max_lines]) + '\n# ... (truncated)'
        return code
    
    def generate_prompt(
        self, 
        parent: Optional[Program] = None,
        hint: Optional[str] = None
    ) -> str:
        """Generate a compact prompt for mutation."""
        import random
        
        parts = []
        
        # 1. Brief role (much shorter than full version)
        parts.append("You are a code optimizer. Improve the given code.\n")
        
        # 2. Problem (condensed)
        parts.append(f"Task: {self.problem_description}\n")
        parts.append(f"Function name: {self.function_name}\n")
        
        # 3. Get best/parent program
        if parent is None:
            sample = self.database.sample_for_prompt()
            parent = sample.get("best") or (sample["parents"][0] if sample["parents"] else None)
        
        # 4. Current code to improve
        if parent:
            parts.append(f"\nCurrent solution (score: {parent.score:.4f}):")
            parts.append(f"```python\n{self._format_code(parent.code)}\n```\n")
        
        # 5. One example of better code (if available and different)
        if self.config.include_examples > 0 and self.database.best_program:
            if parent is None or self.database.best_program.id != parent.id:
                parts.append(f"\nBest known (score: {self.database.best_program.score:.4f}):")
                parts.append(f"```python\n{self._format_code(self.database.best_program.code, 20)}\n```\n")
        
        # 6. Mutation instruction
        hint = hint or random.choice(self.mutation_hints)
        parts.append(f"\nImprove the code: {hint}")
        parts.append(f"\nRespond with only the improved Python code in ```python``` block.")
        
        return '\n'.join(parts)
    
    def generate_simple_prompt(self, code: str, score: float) -> str:
        """
        Generate an even simpler prompt for very small models.
        
        This is the minimal prompt that still works.
        """
        return f"""Improve this Python code to get a higher score.

Current code (score: {score:.4f}):
```python
{self._format_code(code, 25)}
```

Write improved code:
```python"""
    
    def generate_batch_prompts(
        self, 
        n: int = 3,
        vary_hints: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate multiple prompts with different mutation hints."""
        import random
        
        prompts = []
        sample = self.database.sample_for_prompt()
        
        # Get available parents
        parents = []
        if sample["best"]:
            parents.append(sample["best"])
        parents.extend(sample.get("parents", []))
        parents.extend(sample.get("diverse", []))
        
        # Deduplicate
        seen = set()
        unique_parents = []
        for p in parents:
            if p.id not in seen:
                seen.add(p.id)
                unique_parents.append(p)
        
        hints = self.mutation_hints.copy()
        random.shuffle(hints)
        
        for i in range(n):
            parent = unique_parents[i % len(unique_parents)] if unique_parents else None
            hint = hints[i % len(hints)] if vary_hints else None
            
            prompt = self.generate_prompt(parent=parent, hint=hint)
            
            prompts.append({
                "prompt": prompt,
                "parent_id": parent.id if parent else None,
                "hint": hint,
                "index": i
            })
        
        return prompts


class UltraCompactPromptSampler:
    """
    Ultra-minimal prompts for tiny models (<2B parameters).
    
    These models work best with:
    - Very short prompts (<500 tokens)
    - Direct instructions without explanation
    - Code-completion style rather than chat
    """
    
    def __init__(
        self,
        database: ProgramDatabase,
        function_name: str,
        objective: str = "maximize score"
    ):
        self.database = database
        self.function_name = function_name
        self.objective = objective
    
    def generate_completion_prompt(self, parent: Optional[Program] = None) -> str:
        """Generate a completion-style prompt."""
        if parent is None and self.database.best_program:
            parent = self.database.best_program
        
        if parent:
            return f"""# Improve this function to {self.objective}
# Current score: {parent.score:.4f}

{parent.code}

# Improved version:
def {self.function_name}"""
        else:
            return f"""# Write a function to {self.objective}

def {self.function_name}"""
    
    def generate_prompt(self, parent: Optional[Program] = None) -> str:
        """Generate a minimal chat-style prompt."""
        if parent is None and self.database.best_program:
            parent = self.database.best_program
        
        code = parent.code if parent else f"def {self.function_name}(): pass"
        score = parent.score if parent else 0.0
        
        return f"""Improve this code (score: {score:.3f}):
```python
{code}
```
Better version:
```python"""


def create_compact_sampler(
    database: ProgramDatabase,
    problem_description: str,
    function_name: str,
    model_size: str = "small"  # "tiny", "small", "medium"
) -> Any:
    """
    Create appropriate prompt sampler based on model size.
    
    Args:
        model_size: 
            - "tiny": <2B params, use UltraCompactPromptSampler
            - "small": 2-7B params, use CompactPromptSampler
            - "medium": 7B+ params, use full PromptSampler
    """
    if model_size == "tiny":
        return UltraCompactPromptSampler(
            database=database,
            function_name=function_name,
            objective=problem_description[:100]
        )
    elif model_size == "small":
        return CompactPromptSampler(
            database=database,
            problem_description=problem_description,
            function_name=function_name,
            config=CompactPromptConfig(max_prompt_tokens=1500, include_examples=1)
        )
    else:  # medium or larger
        from .prompt_sampler import PromptSampler
        return PromptSampler(
            database=database,
            problem_description=problem_description,
            function_name=function_name
        )
