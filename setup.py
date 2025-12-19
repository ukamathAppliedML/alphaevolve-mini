"""
AlphaEvolve-Mini: Educational Implementation of AlphaEvolve

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="alphaevolve-mini",
    version="0.1.0",
    description="Educational implementation of Google DeepMind's AlphaEvolve",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Educational Implementation",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        # Core dependencies (minimal)
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.20.0"],
        "google": ["google-generativeai>=0.3.0"],
        "viz": ["matplotlib>=3.5.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.20.0",
            "google-generativeai>=0.3.0",
            "matplotlib>=3.5.0",
            "httpx>=0.24.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "alphaevolve=core.controller:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="alphaevolve, evolutionary-algorithm, llm, code-optimization, deepmind",
)
