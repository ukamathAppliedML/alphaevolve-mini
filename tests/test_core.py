"""
Tests for AlphaEvolve-Mini

Run with: pytest tests/test_core.py -v
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import Program, ProgramDatabase, Island, MAPElitesGrid
from core.prompt_sampler import PromptSampler, PromptConfig
from core.llm_ensemble import LLMEnsemble, extract_code_from_response
from core.evaluator import CodeSandbox, FunctionOptimizationEvaluator, MaximizationEvaluator
from core.problems import validate_circle_packing, circle_packing_fitness


class TestProgram:
    def test_program_creation(self):
        prog = Program(code="def f(x): return x", score=0.5)
        assert prog.code == "def f(x): return x"
        assert prog.score == 0.5
        assert len(prog.id) == 16  # SHA256 hash prefix
    
    def test_program_id_uniqueness(self):
        prog1 = Program(code="def f(x): return x", score=0.5)
        prog2 = Program(code="def f(x): return x*2", score=0.5)
        assert prog1.id != prog2.id


class TestIsland:
    def test_island_add_program(self):
        island = Island(island_id=0, capacity=10)
        prog = Program(code="def f(): pass", score=0.5)
        assert island.add_program(prog)
        assert len(island.programs) == 1
        assert island.elite.id == prog.id
    
    def test_island_capacity(self):
        island = Island(island_id=0, capacity=2)
        prog1 = Program(code="def f1(): pass", score=0.3)
        prog2 = Program(code="def f2(): pass", score=0.5)
        prog3 = Program(code="def f3(): pass", score=0.7)
        
        island.add_program(prog1)
        island.add_program(prog2)
        island.add_program(prog3)  # Should replace prog1
        
        assert len(island.programs) == 2
        assert prog1.id not in island.programs
        assert prog3.id in island.programs
    
    def test_sample_parents(self):
        island = Island(island_id=0, capacity=10)
        for i in range(5):
            prog = Program(code=f"def f{i}(): pass", score=i/10)
            island.add_program(prog)
        
        parents = island.sample_parents(n=2)
        assert len(parents) == 2


class TestMAPElites:
    def test_map_elites_basic(self):
        grid = MAPElitesGrid(feature_dims=["x", "y"], bins_per_dim=5)
        
        prog1 = Program(code="def f1(): pass", score=0.5, metrics={"x": 0.2, "y": 0.3})
        prog2 = Program(code="def f2(): pass", score=0.7, metrics={"x": 0.2, "y": 0.3})
        
        assert grid.add_program(prog1)
        assert grid.add_program(prog2)  # Should replace prog1
        
        elites = grid.get_elites()
        assert len(elites) == 1
        assert elites[0].id == prog2.id


class TestProgramDatabase:
    def test_database_creation(self):
        db = ProgramDatabase(num_islands=4, island_capacity=10)
        assert len(db.islands) == 4
        assert db.generation == 0
    
    def test_add_and_retrieve(self):
        db = ProgramDatabase()
        prog = Program(code="def f(): return 42", score=0.8)
        
        db.add_program(prog)
        
        assert db.best_program.id == prog.id
        assert len(db.all_programs) == 1
    
    def test_sample_for_prompt(self):
        db = ProgramDatabase()
        for i in range(10):
            prog = Program(code=f"def f{i}(): return {i}", score=i/10)
            db.add_program(prog)
        
        sample = db.sample_for_prompt()
        assert "parents" in sample
        assert "diverse" in sample
        assert "best" in sample


class TestCodeSandbox:
    def test_safe_execution(self):
        sandbox = CodeSandbox(timeout_seconds=2.0)
        code = "def add(a, b): return a + b"
        
        results = sandbox.execute(code, "add", [(2, 3), (10, 20)])
        
        assert len(results) == 2
        assert results[0] == (True, 5, pytest.approx(results[0][2], abs=100))
        assert results[1][1] == 30
    
    def test_dangerous_code_blocked(self):
        sandbox = CodeSandbox()
        
        # Try to import os
        code = "import os\ndef f(): return os.getcwd()"
        results = sandbox.execute(code, "f", [()])
        assert not results[0][0]
    
    def test_timeout(self):
        sandbox = CodeSandbox(timeout_seconds=0.1)
        code = """
def slow():
    x = 0
    for i in range(10**9):
        x += 1
    return x
"""
        results = sandbox.execute(code, "slow", [()])
        assert not results[0][0]
        assert "Timeout" in str(results[0][1])


class TestEvaluator:
    def test_function_optimization_evaluator(self):
        evaluator = FunctionOptimizationEvaluator(
            problem_description="Test",
            function_name="square",
            test_cases=[
                (2, 4),
                (3, 9),
                (0, 0),
            ],
            objective_function=lambda outputs: 1.0 if all(o is not None for o in outputs) else 0.0
        )
        
        correct_code = "def square(x): return x * x"
        result = evaluator.evaluate(correct_code)
        
        assert result.success
        assert result.metrics["correctness"] == 1.0
    
    def test_incorrect_solution(self):
        evaluator = FunctionOptimizationEvaluator(
            problem_description="Test",
            function_name="square",
            test_cases=[(2, 4), (3, 9)],
            objective_function=lambda outputs: 1.0
        )
        
        wrong_code = "def square(x): return x + x"
        result = evaluator.evaluate(wrong_code)
        
        assert result.metrics["correctness"] < 1.0


class TestCirclePacking:
    def test_valid_packing(self):
        circles = [
            (0.25, 0.25, 0.2),
            (0.75, 0.75, 0.2),
        ]
        valid, msg = validate_circle_packing(circles)
        assert valid, msg
    
    def test_overlapping_circles(self):
        circles = [
            (0.5, 0.5, 0.3),
            (0.6, 0.5, 0.3),  # Overlaps with first
        ]
        valid, msg = validate_circle_packing(circles)
        assert not valid
        assert "overlap" in msg.lower()
    
    def test_out_of_bounds(self):
        circles = [
            (0.1, 0.5, 0.2),  # Left edge goes outside
        ]
        valid, msg = validate_circle_packing(circles)
        assert not valid
    
    def test_fitness_calculation(self):
        circles = [
            (0.25, 0.25, 0.1),
            (0.75, 0.75, 0.2),
        ]
        fitness = circle_packing_fitness(circles)
        assert fitness == pytest.approx(0.3)


class TestLLMEnsemble:
    @pytest.mark.asyncio
    async def test_mock_generation(self):
        ensemble = LLMEnsemble()  # Uses mock by default
        response = await ensemble.generate("Test prompt")
        
        assert response.content is not None
        assert response.model == "mock"
        assert response.latency_ms > 0
    
    def test_extract_code_from_response(self):
        response = """
Here's the solution:

```python
def solve(x):
    return x * 2
```

This doubles the input.
"""
        code = extract_code_from_response(response)
        assert code is not None
        assert "def solve(x):" in code
        assert "return x * 2" in code


class TestPromptSampler:
    def test_prompt_generation(self):
        db = ProgramDatabase()
        prog = Program(code="def f(): return 1", score=0.5)
        db.add_program(prog)
        
        sampler = PromptSampler(
            database=db,
            problem_description="Test problem",
            function_name="f"
        )
        
        prompt = sampler.generate_prompt()
        
        assert "Test problem" in prompt
        assert "def f():" in prompt
    
    def test_batch_prompts(self):
        db = ProgramDatabase()
        for i in range(5):
            prog = Program(code=f"def f(): return {i}", score=i/10)
            db.add_program(prog)
        
        sampler = PromptSampler(
            database=db,
            problem_description="Test",
            function_name="f"
        )
        
        prompts = sampler.generate_batch_prompts(n=3)
        
        assert len(prompts) == 3
        assert all("prompt" in p for p in prompts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
