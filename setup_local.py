#!/usr/bin/env python3
"""
AlphaEvolve-Mini Setup Script

This script helps you set up everything needed to run AlphaEvolve locally.

Usage:
    python setup_local.py --check           # Check what's installed
    python setup_local.py --install         # Install Python dependencies
    python setup_local.py --setup-ollama    # Setup Ollama with a model
    python setup_local.py --all             # Do everything
"""

import subprocess
import sys
import os
import shutil
import platform
from pathlib import Path


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_status(name, status, details=""):
    symbol = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{symbol}{reset} {name}: {details}")


def check_python():
    """Check Python version."""
    version = sys.version_info
    ok = version >= (3, 8)
    print_status(
        "Python", 
        ok, 
        f"{version.major}.{version.minor}.{version.micro}" + 
        (" (3.8+ required)" if not ok else "")
    )
    return ok


def check_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    import_name = import_name or package_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def check_ollama():
    """Check if Ollama is installed and running."""
    # Check if ollama command exists
    ollama_path = shutil.which("ollama")
    installed = ollama_path is not None
    
    if not installed:
        print_status("Ollama", False, "Not installed")
        return False, False, []
    
    # Check if running
    try:
        import urllib.request
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        import json
        data = json.loads(req.read())
        models = [m["name"] for m in data.get("models", [])]
        print_status("Ollama", True, f"Running with {len(models)} models")
        return True, True, models
    except Exception:
        print_status("Ollama", True, "Installed but not running")
        return True, False, []


def check_gpu():
    """Check for GPU availability."""
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_status("CUDA GPU", True, gpu_name)
            return "cuda"
    except ImportError:
        pass
    
    # Check MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print_status("Apple MPS", True, "Available")
            return "mps"
    except ImportError:
        pass
    
    print_status("GPU", False, "CPU only (slower but works)")
    return "cpu"


def run_checks():
    """Run all system checks."""
    print_header("System Check")
    
    results = {}
    
    # Python
    results["python"] = check_python()
    
    # Required packages
    print("\n  Python packages:")
    packages = [
        ("httpx", "httpx"),  # For Ollama
    ]
    for pkg, imp in packages:
        installed = check_package(imp)
        print_status(f"  {pkg}", installed, "installed" if installed else "missing")
        results[pkg] = installed
    
    # Optional packages
    print("\n  Optional packages:")
    optional = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("matplotlib", "matplotlib"),
    ]
    for pkg, imp in optional:
        installed = check_package(imp)
        print_status(f"  {pkg}", installed, "installed" if installed else "not installed")
    
    # Ollama
    print("\n  Local LLM:")
    ollama_installed, ollama_running, models = check_ollama()
    results["ollama"] = ollama_running
    
    if models:
        print("    Available models:")
        for m in models[:5]:
            print(f"      - {m}")
        if len(models) > 5:
            print(f"      ... and {len(models)-5} more")
    
    # GPU
    print("\n  Hardware:")
    gpu = check_gpu()
    results["gpu"] = gpu
    
    # Summary
    print_header("Summary")
    
    if results["python"] and results.get("httpx"):
        if ollama_running and models:
            print("  🎉 Ready to run! Use:")
            print(f"     python examples/local_demo.py")
        elif ollama_installed:
            print("  ⚠️  Ollama installed but not running. Start it with:")
            print("     ollama serve")
            print("  Then pull a model:")
            print("     ollama pull qwen2.5-coder:1.5b")
        else:
            print("  ⚠️  Ollama not installed. Install from: https://ollama.ai")
    else:
        print("  ❌ Missing dependencies. Run:")
        print("     python setup_local.py --install")
    
    return results


def install_dependencies(gpu_support=True):
    """Install Python dependencies."""
    print_header("Installing Dependencies")
    
    # Base dependencies
    packages = ["httpx"]
    
    # Optional but recommended
    optional = ["matplotlib"]
    
    print("Installing base packages...")
    for pkg in packages:
        print(f"  Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"])
    
    # GPU support
    if gpu_support:
        print("\nInstalling PyTorch (for GPU support)...")
        # Detect platform for correct PyTorch install
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "-q"])
        else:
            # Try CUDA version first, fall back to CPU
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "torch", "--index-url", "https://download.pytorch.org/whl/cu118", "-q"
                ], check=True)
            except subprocess.CalledProcessError:
                subprocess.run([sys.executable, "-m", "pip", "install", "torch", "-q"])
    
    print("\nInstalling optional packages...")
    for pkg in optional:
        print(f"  Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"])
    
    print("\n✓ Dependencies installed!")


def setup_ollama(model="qwen2.5-coder:1.5b"):
    """Setup Ollama with a recommended model."""
    print_header(f"Setting up Ollama with {model}")
    
    # Check if installed
    if not shutil.which("ollama"):
        print("Ollama not installed!")
        print("\nInstallation instructions:")
        system = platform.system()
        if system == "Darwin":
            print("  brew install ollama")
            print("  or download from https://ollama.ai")
        elif system == "Linux":
            print("  curl -fsSL https://ollama.ai/install.sh | sh")
        else:
            print("  Download from https://ollama.ai")
        return False
    
    # Check if running
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
    except Exception:
        print("Ollama not running. Starting...")
        # Try to start (this may not work in all environments)
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        import time
        time.sleep(3)
    
    # Pull model
    print(f"Pulling {model}... (this may take a few minutes)")
    result = subprocess.run(["ollama", "pull", model])
    
    if result.returncode == 0:
        print(f"\n✓ Model {model} ready!")
        return True
    else:
        print(f"\n✗ Failed to pull model")
        return False


def create_config():
    """Create a default configuration file."""
    config_content = """# AlphaEvolve-Mini Local Configuration

# LLM Settings
provider: ollama
model: qwen2.5-coder:1.5b

# Evolution Settings  
num_generations: 30
population_per_generation: 5
num_islands: 3

# Hardware
device: auto  # auto, cpu, cuda, mps

# Paths
checkpoint_dir: ./checkpoints
output_dir: ./outputs
"""
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        config_path.write_text(config_content)
        print(f"Created {config_path}")
    else:
        print(f"{config_path} already exists")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AlphaEvolve-Mini Setup")
    parser.add_argument("--check", action="store_true", help="Check system status")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--setup-ollama", action="store_true", help="Setup Ollama")
    parser.add_argument("--model", default="qwen2.5-coder:1.5b", help="Ollama model to install")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU packages")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    parser.add_argument("--config", action="store_true", help="Create config file")
    
    args = parser.parse_args()
    
    # Default to check if no args
    if not any([args.check, args.install, args.setup_ollama, args.all, args.config]):
        args.check = True
    
    if args.all:
        args.install = True
        args.setup_ollama = True
        args.config = True
    
    if args.install:
        install_dependencies(gpu_support=not args.no_gpu)
    
    if args.setup_ollama:
        setup_ollama(args.model)
    
    if args.config:
        create_config()
    
    if args.check or args.all:
        run_checks()


if __name__ == "__main__":
    main()
