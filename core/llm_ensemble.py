"""
LLM Ensemble: Interface for multiple language models.

AlphaEvolve uses an ensemble of models:
- Fast model (e.g., Gemini Flash): High throughput for exploration
- Powerful model (e.g., Gemini Pro): Occasional deep reasoning

This module provides a unified interface supporting:
- OpenAI API (GPT-4, GPT-4o)
- Anthropic API (Claude)
- Google AI (Gemini)
- Local models via Ollama

Reference: Section 2.3 of AlphaEvolve whitepaper
"""

import os
import re
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import random


@dataclass
class LLMConfig:
    """Configuration for an LLM endpoint."""
    provider: str  # "openai", "anthropic", "google", "ollama"
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    role: str = "fast"  # "fast" or "powerful"


@dataclass  
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    usage: Dict[str, int]
    latency_ms: float


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Generate a response from the LLM."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (also works with compatible APIs)."""
    
    async def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        try:
            import openai
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        import time
        start_time = time.time()
        
        client = AsyncOpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.base_url
        )
        
        response = await client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=config.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            },
            latency_ms=latency_ms
        )


class AnthropicProvider(LLMProvider):
    """Anthropic API provider for Claude models."""
    
    async def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        import time
        start_time = time.time()
        
        client = anthropic.AsyncAnthropic(
            api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        
        response = await client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.content[0].text,
            model=config.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens
            },
            latency_ms=latency_ms
        )


class GoogleProvider(LLMProvider):
    """Google AI provider for Gemini models."""
    
    async def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        
        import time
        start_time = time.time()
        
        genai.configure(api_key=config.api_key or os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(config.model)
        
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens
            )
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.text,
            model=config.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0},  # Gemini doesn't always report
            latency_ms=latency_ms
        )


class OllamaProvider(LLMProvider):
    """Ollama provider for local models."""
    
    async def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        try:
            import httpx
        except ImportError:
            raise ImportError("Please install httpx: pip install httpx")
        
        import time
        start_time = time.time()
        
        base_url = config.base_url or "http://localhost:11434"
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": config.temperature,
                        "num_predict": config.max_tokens
                    }
                }
            )
            result = response.json()
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=result.get("response", ""),
            model=config.model,
            usage={
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0)
            },
            latency_ms=latency_ms
        )


class MockProvider(LLMProvider):
    """Mock provider for testing without API calls."""
    
    async def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        # Generate a simple mutation for testing
        import time
        await asyncio.sleep(0.1)  # Simulate latency
        
        mock_code = '''def solve(x):
    # Mock evolved solution
    result = 0
    for i in range(len(x)):
        result += x[i] * (i + 1)
    return result
'''
        
        return LLMResponse(
            content=f"```python\n{mock_code}\n```",
            model="mock",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            latency_ms=100
        )


def get_provider(provider_name: str) -> LLMProvider:
    """Factory function to get the appropriate provider."""
    providers = {
        "openai": OpenAIProvider(),
        "anthropic": AnthropicProvider(),
        "google": GoogleProvider(),
        "ollama": OllamaProvider(),
        "mock": MockProvider()
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Choose from: {list(providers.keys())}")
    
    return providers[provider_name]


class LLMEnsemble:
    """
    Ensemble of LLMs for evolutionary code generation.
    
    The key insight is balancing:
    - Fast models: High throughput, more exploration
    - Powerful models: Better quality, breakthrough discoveries
    
    AlphaEvolve uses ~80% fast model calls, ~20% powerful model calls.
    """
    
    def __init__(
        self,
        fast_config: Optional[LLMConfig] = None,
        powerful_config: Optional[LLMConfig] = None,
        fast_ratio: float = 0.8
    ):
        self.fast_config = fast_config
        self.powerful_config = powerful_config
        self.fast_ratio = fast_ratio
        
        # Default configurations if not provided
        if self.fast_config is None:
            self.fast_config = LLMConfig(
                provider="mock",
                model="mock-fast",
                role="fast"
            )
        
        if self.powerful_config is None:
            self.powerful_config = LLMConfig(
                provider="mock", 
                model="mock-powerful",
                role="powerful"
            )
        
        self.providers = {
            "fast": get_provider(self.fast_config.provider),
            "powerful": get_provider(self.powerful_config.provider)
        }
        
        # Statistics
        self.stats = {
            "fast_calls": 0,
            "powerful_calls": 0,
            "total_tokens": 0,
            "total_latency_ms": 0
        }
    
    def select_model(self) -> Tuple[str, LLMConfig]:
        """Select which model to use based on the configured ratio."""
        if random.random() < self.fast_ratio:
            return "fast", self.fast_config
        else:
            return "powerful", self.powerful_config
    
    async def generate(self, prompt: str, force_powerful: bool = False) -> LLMResponse:
        """
        Generate a response using the ensemble.
        
        Args:
            prompt: The prompt to send
            force_powerful: If True, always use the powerful model
        """
        if force_powerful:
            role, config = "powerful", self.powerful_config
        else:
            role, config = self.select_model()
        
        provider = self.providers[role]
        response = await provider.generate(prompt, config)
        
        # Update statistics
        self.stats[f"{role}_calls"] += 1
        self.stats["total_tokens"] += sum(response.usage.values())
        self.stats["total_latency_ms"] += response.latency_ms
        
        return response
    
    async def generate_batch(
        self, 
        prompts: List[str],
        max_concurrent: int = 5,
        force_powerful: bool = False
    ) -> List[Optional[LLMResponse]]:
        """
        Generate responses for multiple prompts in parallel.
        
        Returns a list of LLMResponse objects (or None for failed requests).
        Failed requests don't cancel the entire batch.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_generate(prompt: str) -> Optional[LLMResponse]:
            async with semaphore:
                try:
                    return await self.generate(prompt, force_powerful=force_powerful)
                except Exception as e:
                    # Log but don't crash - return None for this prompt
                    import logging
                    logging.warning(f"LLM generation failed: {e}")
                    return None
        
        tasks = [limited_generate(prompt) for prompt in prompts]
        # Use return_exceptions=False since we handle exceptions in limited_generate
        results = await asyncio.gather(*tasks)
        return list(results)
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        total_calls = self.stats["fast_calls"] + self.stats["powerful_calls"]
        return {
            **self.stats,
            "total_calls": total_calls,
            "actual_fast_ratio": self.stats["fast_calls"] / total_calls if total_calls > 0 else 0,
            "avg_latency_ms": self.stats["total_latency_ms"] / total_calls if total_calls > 0 else 0
        }


def extract_code_from_response(response: str) -> Optional[Tuple[str, str]]:
    """
    Extract Python code from an LLM response.
    
    Returns:
        Tuple of (code_type, content) where code_type is "full" or "diff"
        Returns None if no code found.
    """
    # Try to find Python code block
    python_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
    if python_match:
        return ("full", python_match.group(1).strip())
    
    # Try to find diff block explicitly
    diff_match = re.search(r'```diff\n(.*?)```', response, re.DOTALL)
    if diff_match:
        return ("diff", diff_match.group(1).strip())
    
    # Try to find generic code block
    code_match = re.search(r'```\n(.*?)```', response, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        # Check if it looks like a diff
        if content.startswith('---') or content.startswith('@@') or content.startswith('diff'):
            return ("diff", content)
        return ("full", content)
    
    # Fallback: detect unfenced diff by looking for unified diff patterns
    # Look for @@ hunk markers with +/- lines
    if re.search(r'^@@\s+-\d+', response, re.MULTILINE):
        # Extract the diff portion (from first --- or @@ to end of diff content)
        diff_start = re.search(r'^(---|\+\+\+|@@)', response, re.MULTILINE)
        if diff_start:
            # Find where the diff ends (blank line followed by non-diff content, or end)
            diff_content = response[diff_start.start():]
            # Trim trailing non-diff content
            lines = diff_content.split('\n')
            diff_lines = []
            for line in lines:
                if (line.startswith('---') or line.startswith('+++') or 
                    line.startswith('@@') or line.startswith('+') or 
                    line.startswith('-') or line.startswith(' ') or line == ''):
                    diff_lines.append(line)
                elif diff_lines:  # We had diff content and hit non-diff
                    break
            if diff_lines:
                return ("diff", '\n'.join(diff_lines).strip())
    
    # Fallback: look for any code that looks like Python function definition
    func_match = re.search(r'(def\s+\w+\s*\([^)]*\).*?)(?=\n\n|\Z)', response, re.DOTALL)
    if func_match:
        return ("full", func_match.group(1).strip())
    
    return None


def apply_diff(original: str, diff: str) -> Optional[str]:
    """
    Apply a unified diff to original code.
    
    Pure Python implementation that doesn't depend on external tools.
    Handles standard unified diff format with @@ hunks.
    """
    try:
        original_lines = original.splitlines(keepends=True)
        # Ensure last line has newline for consistent handling
        if original_lines and not original_lines[-1].endswith('\n'):
            original_lines[-1] += '\n'
        
        result_lines = list(original_lines)
        
        # Parse diff hunks
        hunk_pattern = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
        
        lines = diff.splitlines()
        i = 0
        offset = 0  # Track line number shifts from previous hunks
        
        while i < len(lines):
            line = lines[i]
            
            # Skip diff headers
            if line.startswith('---') or line.startswith('+++') or line.startswith('diff '):
                i += 1
                continue
            
            # Parse hunk header
            match = hunk_pattern.match(line)
            if match:
                orig_start = int(match.group(1)) - 1  # Convert to 0-indexed
                orig_count = int(match.group(2)) if match.group(2) else 1
                
                i += 1
                
                # Process hunk lines
                current_pos = orig_start + offset
                deletions = 0
                insertions = []
                
                while i < len(lines):
                    if i >= len(lines):
                        break
                    hunk_line = lines[i]
                    
                    # Check for next hunk or end
                    if hunk_line.startswith('@@') or hunk_line.startswith('diff '):
                        break
                    
                    if hunk_line.startswith('-'):
                        # Delete line
                        if current_pos < len(result_lines):
                            del result_lines[current_pos]
                            deletions += 1
                    elif hunk_line.startswith('+'):
                        # Insert line
                        new_line = hunk_line[1:] + '\n'
                        result_lines.insert(current_pos, new_line)
                        current_pos += 1
                        insertions.append(new_line)
                    elif hunk_line.startswith(' ') or hunk_line == '':
                        # Context line - advance position
                        current_pos += 1
                    
                    i += 1
                
                # Update offset for next hunk
                offset += len(insertions) - deletions
            else:
                i += 1
        
        return ''.join(result_lines)
    
    except Exception as e:
        # Log for debugging but don't crash
        import logging
        logging.debug(f"apply_diff failed: {e}")
        return None


# Convenience function to create common configurations
def create_openai_ensemble(
    fast_model: str = "gpt-4o-mini",
    powerful_model: str = "gpt-4o"
) -> LLMEnsemble:
    """Create an ensemble using OpenAI models."""
    return LLMEnsemble(
        fast_config=LLMConfig(provider="openai", model=fast_model, role="fast"),
        powerful_config=LLMConfig(provider="openai", model=powerful_model, role="powerful")
    )


def create_anthropic_ensemble(
    fast_model: str = "claude-3-5-haiku-latest",
    powerful_model: str = "claude-sonnet-4-20250514"
) -> LLMEnsemble:
    """Create an ensemble using Anthropic models."""
    return LLMEnsemble(
        fast_config=LLMConfig(provider="anthropic", model=fast_model, role="fast"),
        powerful_config=LLMConfig(provider="anthropic", model=powerful_model, role="powerful")
    )


def create_google_ensemble(
    fast_model: str = "gemini-2.0-flash",
    powerful_model: str = "gemini-2.0-pro"
) -> LLMEnsemble:
    """Create an ensemble using Google Gemini models."""
    return LLMEnsemble(
        fast_config=LLMConfig(provider="google", model=fast_model, role="fast"),
        powerful_config=LLMConfig(provider="google", model=powerful_model, role="powerful")
    )


def create_ollama_ensemble(
    fast_model: str = "llama3.2",
    powerful_model: str = "llama3.1:70b"
) -> LLMEnsemble:
    """Create an ensemble using local Ollama models."""
    return LLMEnsemble(
        fast_config=LLMConfig(provider="ollama", model=fast_model, role="fast"),
        powerful_config=LLMConfig(provider="ollama", model=powerful_model, role="powerful")
    )
