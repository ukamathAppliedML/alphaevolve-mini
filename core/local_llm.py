"""
Local LLM Providers for AlphaEvolve-Mini

Supports running completely offline with:
1. Ollama (recommended) - Easiest setup, great model selection
2. Hugging Face Transformers - Direct model loading
3. llama-cpp-python - Efficient CPU/GPU inference

Recommended models for code evolution (in order of quality/speed tradeoff):
- Qwen2.5-Coder-7B-Instruct (best quality, needs ~16GB RAM)
- Qwen2.5-Coder-1.5B-Instruct (good balance, ~4GB RAM)
- DeepSeek-Coder-1.3B-Instruct (fast, ~3GB RAM)  
- CodeGemma-2B (good for simple tasks, ~5GB RAM)
- Phi-3-mini-4k-instruct (general purpose, ~8GB RAM)
"""

import os
import asyncio
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Import base classes from llm_ensemble
from .llm_ensemble import LLMProvider, LLMConfig, LLMResponse


@dataclass
class LocalModelConfig:
    """Configuration for local models."""
    model_name: str
    model_path: Optional[str] = None  # For HF or GGUF path
    context_length: int = 4096
    temperature: float = 0.7
    max_tokens: int = 2048
    device: str = "auto"  # "cpu", "cuda", "mps", "auto"
    quantization: Optional[str] = None  # "4bit", "8bit", None
    

# =============================================================================
# OLLAMA PROVIDER (Recommended for ease of use)
# =============================================================================

class OllamaLocalProvider(LLMProvider):
    """
    Ollama provider for local LLM inference.
    
    Setup:
        1. Install Ollama: https://ollama.ai
        2. Pull a model: ollama pull qwen2.5-coder:1.5b
        3. Ollama runs as a service on localhost:11434
    
    Recommended models:
        - qwen2.5-coder:7b (best quality)
        - qwen2.5-coder:1.5b (good balance)
        - deepseek-coder:1.3b (fastest)
        - codellama:7b (good for Python)
        - phi3:mini (general purpose)
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout_seconds: float = 60.0):
        """
        Initialize Ollama provider.
        
        Args:
            base_url: Ollama server URL
            timeout_seconds: Max time for single generation (default 60s, was 120s)
        """
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._available_models = None
    
    async def check_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            import httpx
            print(f"Pulling model {model_name}... (this may take a while)")
            async with httpx.AsyncClient(timeout=3600.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name},
                    timeout=3600.0
                )
                return response.status_code == 200
        except Exception as e:
            print(f"Failed to pull model: {e}")
            return False
    
    async def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Generate response using Ollama with timeout protection."""
        try:
            import httpx
        except ImportError:
            raise ImportError("Please install httpx: pip install httpx")
        
        start_time = time.time()
        
        # Use explicit timeout config for better control
        timeout_config = httpx.Timeout(
            connect=10.0,  # Connection timeout
            read=self.timeout_seconds,  # Read timeout (main generation time)
            write=10.0,  # Write timeout
            pool=10.0  # Pool timeout
        )
        
        try:
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": config.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": config.temperature,
                            "num_predict": config.max_tokens,
                            "num_ctx": 4096,
                        }
                    }
                )
                result = response.json()
        except httpx.TimeoutException:
            # Return empty response on timeout rather than crashing
            return LLMResponse(
                content="",
                model=config.model,
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                latency_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            # Log and return empty on other errors
            import logging
            logging.warning(f"Ollama generation error: {e}")
            return LLMResponse(
                content="",
                model=config.model,
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                latency_ms=(time.time() - start_time) * 1000
            )
        
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


# =============================================================================
# HUGGING FACE TRANSFORMERS PROVIDER
# =============================================================================

class HuggingFaceLocalProvider(LLMProvider):
    """
    Hugging Face Transformers provider for local inference.
    
    Loads models directly - no external service needed.
    Supports quantization for memory efficiency.
    
    Setup:
        pip install transformers torch accelerate bitsandbytes
    
    Recommended models:
        - Qwen/Qwen2.5-Coder-1.5B-Instruct (best for code)
        - deepseek-ai/deepseek-coder-1.3b-instruct
        - google/codegemma-2b
        - microsoft/phi-3-mini-4k-instruct
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.device = None
    
    def _get_device(self, device_preference: str = "auto") -> str:
        """Determine the best available device."""
        import torch
        
        if device_preference != "auto":
            return device_preference
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self, model_name: str, config: LocalModelConfig):
        """Load a model from Hugging Face."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers torch accelerate"
            )
        
        if self.current_model_name == model_name:
            return  # Already loaded
        
        print(f"Loading model {model_name}...")
        self.device = self._get_device(config.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
        }
        
        # Add quantization if requested
        if config.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError:
                print("Warning: bitsandbytes not installed, skipping quantization")
        elif config.quantization == "8bit":
            model_kwargs["load_in_8bit"] = True
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device != "cpu" else None,
            **model_kwargs
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.current_model_name = model_name
        print(f"Model loaded on {self.device}")
    
    async def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Generate response using loaded model."""
        import torch
        
        # Load model if needed
        local_config = LocalModelConfig(
            model_name=config.model,
            device=getattr(config, 'device', 'auto'),
            quantization=getattr(config, 'quantization', None)
        )
        self.load_model(config.model, local_config)
        
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode (only new tokens)
        generated_tokens = outputs[0][prompt_tokens:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response_text,
            model=config.model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(generated_tokens)
            },
            latency_ms=latency_ms
        )


# =============================================================================
# LLAMA.CPP PROVIDER (Most efficient for CPU)
# =============================================================================

class LlamaCppProvider(LLMProvider):
    """
    llama-cpp-python provider for efficient CPU/GPU inference.
    
    Uses GGUF quantized models for memory efficiency.
    
    Setup:
        pip install llama-cpp-python
        
        # For GPU support:
        CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
    
    Download GGUF models from:
        https://huggingface.co/TheBloke (many quantized models)
    """
    
    def __init__(self):
        self.llm = None
        self.current_model_path = None
    
    def load_model(self, model_path: str, config: LocalModelConfig):
        """Load a GGUF model."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("Please install llama-cpp-python: pip install llama-cpp-python")
        
        if self.current_model_path == model_path:
            return
        
        print(f"Loading GGUF model from {model_path}...")
        
        # Determine GPU layers
        n_gpu_layers = 0
        if config.device in ["cuda", "auto"]:
            try:
                import torch
                if torch.cuda.is_available():
                    n_gpu_layers = -1  # All layers on GPU
            except ImportError:
                pass
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=config.context_length,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        
        self.current_model_path = model_path
        print("Model loaded")
    
    async def generate(self, prompt: str, config: LLMConfig) -> LLMResponse:
        """Generate response using llama.cpp."""
        start_time = time.time()
        
        # model path should be in config.model for GGUF
        local_config = LocalModelConfig(model_name=config.model)
        self.load_model(config.model, local_config)
        
        output = self.llm(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            echo=False
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=output["choices"][0]["text"],
            model=config.model,
            usage={
                "prompt_tokens": output["usage"]["prompt_tokens"],
                "completion_tokens": output["usage"]["completion_tokens"]
            },
            latency_ms=latency_ms
        )


# =============================================================================
# SMART LOCAL ENSEMBLE
# =============================================================================

class LocalLLMEnsemble:
    """
    Ensemble optimized for local models.
    
    Unlike cloud APIs, local models benefit from:
    - Using the same model for fast/powerful (avoids loading multiple)
    - Adjusting temperature instead of model for diversity
    - Caching model in memory
    """
    
    def __init__(
        self,
        provider: str = "ollama",  # "ollama", "huggingface", "llamacpp"
        model_name: str = "qwen2.5-coder:1.5b",
        model_path: Optional[str] = None,  # For GGUF models
        device: str = "auto",
        quantization: Optional[str] = None,
        timeout_seconds: float = 60.0,  # Timeout for LLM calls
    ):
        self.provider_name = provider
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.quantization = quantization
        self.timeout_seconds = timeout_seconds
        
        # Create provider
        if provider == "ollama":
            self.provider = OllamaLocalProvider(timeout_seconds=timeout_seconds)
        elif provider == "huggingface":
            self.provider = HuggingFaceLocalProvider()
        elif provider == "llamacpp":
            self.provider = LlamaCppProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Statistics
        self.stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_latency_ms": 0,
            "timeouts": 0
        }
    
    async def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> LLMResponse:
        """Generate a response."""
        config = LLMConfig(
            provider=self.provider_name,
            model=self.model_path or self.model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response = await self.provider.generate(prompt, config)
        
        # Update stats
        self.stats["total_calls"] += 1
        self.stats["total_tokens"] += sum(response.usage.values())
        self.stats["total_latency_ms"] += response.latency_ms
        
        # Track timeouts (empty content usually indicates timeout)
        if not response.content:
            self.stats["timeouts"] += 1
        
        return response
    
    async def generate_diverse(
        self,
        prompt: str,
        n: int = 3,
        temperature_range: tuple = (0.5, 1.0)
    ) -> List[LLMResponse]:
        """Generate diverse responses by varying temperature."""
        import random
        
        responses = []
        for i in range(n):
            # Vary temperature for diversity
            temp = temperature_range[0] + (temperature_range[1] - temperature_range[0]) * (i / max(1, n-1))
            temp += random.uniform(-0.1, 0.1)
            temp = max(0.1, min(1.5, temp))
            
            response = await self.generate(prompt, temperature=temp)
            responses.append(response)
        
        return responses
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            **self.stats,
            "avg_latency_ms": self.stats["total_latency_ms"] / max(1, self.stats["total_calls"])
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def check_ollama_status() -> Dict[str, Any]:
    """Check Ollama installation and available models."""
    provider = OllamaLocalProvider()
    
    available = await provider.check_available()
    models = await provider.list_models() if available else []
    
    return {
        "installed": available,
        "running": available,
        "models": models,
        "recommended": [
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:7b", 
            "deepseek-coder:1.3b",
            "codellama:7b"
        ]
    }


async def setup_ollama_model(model_name: str = "qwen2.5-coder:1.5b") -> bool:
    """Ensure an Ollama model is available, pulling if necessary."""
    provider = OllamaLocalProvider()
    
    if not await provider.check_available():
        print("ERROR: Ollama is not running!")
        print("Please install and start Ollama: https://ollama.ai")
        return False
    
    models = await provider.list_models()
    
    if model_name in models or any(model_name in m for m in models):
        print(f"Model {model_name} is ready")
        return True
    
    print(f"Model {model_name} not found, pulling...")
    return await provider.pull_model(model_name)


def create_local_ensemble(
    provider: str = "ollama",
    model: str = "qwen2.5-coder:1.5b",
    **kwargs
) -> LocalLLMEnsemble:
    """
    Create a local LLM ensemble.
    
    Args:
        provider: "ollama", "huggingface", or "llamacpp"
        model: Model name or path
        **kwargs: Additional config (device, quantization, etc.)
    
    Examples:
        # Ollama (easiest)
        ensemble = create_local_ensemble("ollama", "qwen2.5-coder:1.5b")
        
        # Hugging Face
        ensemble = create_local_ensemble(
            "huggingface", 
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            quantization="4bit"
        )
        
        # llama.cpp with GGUF
        ensemble = create_local_ensemble(
            "llamacpp",
            model_path="/path/to/model.gguf"
        )
    """
    return LocalLLMEnsemble(
        provider=provider,
        model_name=model,
        **kwargs
    )


# Model recommendations based on hardware
HARDWARE_RECOMMENDATIONS = {
    "low_end": {
        "description": "4-8GB RAM, CPU only",
        "ollama": "deepseek-coder:1.3b",
        "huggingface": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "quantization": "4bit"
    },
    "mid_range": {
        "description": "8-16GB RAM, optional GPU",
        "ollama": "qwen2.5-coder:1.5b",
        "huggingface": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "quantization": "8bit"
    },
    "high_end": {
        "description": "16GB+ RAM, GPU with 8GB+ VRAM",
        "ollama": "qwen2.5-coder:7b",
        "huggingface": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "quantization": None
    }
}


def get_recommended_model(hardware_tier: str = "mid_range") -> Dict[str, str]:
    """Get recommended model for your hardware."""
    return HARDWARE_RECOMMENDATIONS.get(hardware_tier, HARDWARE_RECOMMENDATIONS["mid_range"])
