"""
Program Database: The evolutionary heart of AlphaEvolve.

Implements MAP-Elites (quality-diversity) combined with Island-based evolution.
This is the key innovation that enables AlphaEvolve to maintain diversity while
converging on high-quality solutions.

Reference: Section 2.5 of AlphaEvolve whitepaper
"""

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json


@dataclass
class Program:
    """A program in the evolutionary population."""
    code: str
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    island_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Unique identifier based on code hash."""
        return hashlib.sha256(self.code.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "code": self.code,
            "score": self.score,
            "metrics": self.metrics,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "timestamp": self.timestamp.isoformat(),
            "island_id": self.island_id,
            "metadata": self.metadata
        }


class Island:
    """
    An island in the evolutionary archipelago.
    
    Each island maintains its own population and evolves semi-independently.
    Periodic migration allows good solutions to spread across islands.
    """
    
    def __init__(self, island_id: int, capacity: int = 100):
        self.island_id = island_id
        self.capacity = capacity
        self.programs: Dict[str, Program] = {}  # id -> Program
        self.elite: Optional[Program] = None
        
    def add_program(self, program: Program) -> bool:
        """Add a program to the island, maintaining capacity limits."""
        program.island_id = self.island_id
        
        # Update elite if this is the best program
        if self.elite is None or program.score > self.elite.score:
            self.elite = program
        
        # If under capacity, just add
        if len(self.programs) < self.capacity:
            self.programs[program.id] = program
            return True
        
        # Otherwise, replace the worst program if new one is better
        worst_id = min(self.programs, key=lambda k: self.programs[k].score)
        if program.score > self.programs[worst_id].score:
            del self.programs[worst_id]
            self.programs[program.id] = program
            return True
        
        return False
    
    def sample_parents(self, n: int = 2, elite_prob: float = 0.3) -> List[Program]:
        """
        Sample parent programs for mutation.
        
        Uses tournament selection with elite bias - this is key for
        balancing exploration (diversity) with exploitation (quality).
        """
        if not self.programs:
            return []
        
        parents = []
        programs_list = list(self.programs.values())
        
        for _ in range(n):
            # With some probability, include the elite
            if self.elite and random.random() < elite_prob:
                parents.append(self.elite)
            else:
                # Tournament selection (size 3)
                tournament = random.sample(
                    programs_list, 
                    min(3, len(programs_list))
                )
                winner = max(tournament, key=lambda p: p.score)
                parents.append(winner)
        
        return parents
    
    def get_diverse_sample(self, n: int = 5) -> List[Program]:
        """Get a diverse sample across score ranges for prompt context."""
        if not self.programs:
            return []
        
        sorted_programs = sorted(
            self.programs.values(), 
            key=lambda p: p.score, 
            reverse=True
        )
        
        # Sample from different score tiers
        result = []
        step = max(1, len(sorted_programs) // n)
        for i in range(0, len(sorted_programs), step):
            if len(result) < n:
                result.append(sorted_programs[i])
        
        return result


class MAPElitesGrid:
    """
    MAP-Elites: Maintain an archive of elite solutions across feature dimensions.
    
    This ensures we don't just find one good solution, but a diverse set of
    high-quality solutions with different characteristics.
    
    Reference: Mouret & Clune (2015) "Illuminating search spaces"
    """
    
    def __init__(self, feature_dims: List[str], bins_per_dim: int = 10):
        self.feature_dims = feature_dims
        self.bins_per_dim = bins_per_dim
        self.grid: Dict[Tuple, Program] = {}
        self.feature_ranges: Dict[str, Tuple[float, float]] = {
            dim: (0.0, 1.0) for dim in feature_dims
        }
    
    def _get_cell(self, program: Program) -> Tuple:
        """Map a program to its grid cell based on features."""
        cell = []
        for dim in self.feature_dims:
            value = program.metrics.get(dim, 0.5)
            low, high = self.feature_ranges[dim]
            # Normalize to [0, 1] then bin
            if high > low:
                normalized = (value - low) / (high - low)
            else:
                normalized = 0.5
            bin_idx = min(int(normalized * self.bins_per_dim), self.bins_per_dim - 1)
            cell.append(bin_idx)
        return tuple(cell)
    
    def update_ranges(self, program: Program):
        """Dynamically update feature ranges as we see more programs."""
        for dim in self.feature_dims:
            if dim in program.metrics:
                value = program.metrics[dim]
                low, high = self.feature_ranges[dim]
                self.feature_ranges[dim] = (min(low, value), max(high, value))
    
    def add_program(self, program: Program) -> bool:
        """Add program to grid if it's elite for its cell."""
        self.update_ranges(program)
        cell = self._get_cell(program)
        
        if cell not in self.grid or program.score > self.grid[cell].score:
            self.grid[cell] = program
            return True
        return False
    
    def get_elites(self) -> List[Program]:
        """Get all elite programs from the grid."""
        return list(self.grid.values())


class ProgramDatabase:
    """
    The complete evolutionary database combining Islands + MAP-Elites.
    
    This is the core data structure that drives AlphaEvolve's evolution.
    It maintains:
    1. Multiple islands for parallel evolution with migration
    2. A MAP-Elites grid for quality-diversity
    3. A global archive of all evaluated programs
    """
    
    def __init__(
        self, 
        num_islands: int = 4,
        island_capacity: int = 50,
        feature_dims: Optional[List[str]] = None,
        migration_interval: int = 10,
        migration_size: int = 2
    ):
        self.num_islands = num_islands
        self.islands = [Island(i, island_capacity) for i in range(num_islands)]
        self.feature_dims = feature_dims or ["complexity", "runtime"]
        self.map_elites = MAPElitesGrid(self.feature_dims)
        self.all_programs: Dict[str, Program] = {}  # Complete archive
        self.generation = 0
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.best_program: Optional[Program] = None
        
    def add_program(self, program: Program, island_id: Optional[int] = None) -> bool:
        """
        Add a program to the database.
        
        The program is added to:
        1. A specific island (or round-robin if not specified)
        2. The MAP-Elites grid (if it's elite for its cell)
        3. The global archive
        """
        program.generation = self.generation
        
        # Assign to island
        if island_id is None:
            island_id = random.randint(0, self.num_islands - 1)
        
        # Add to island
        self.islands[island_id].add_program(program)
        
        # Add to MAP-Elites
        self.map_elites.add_program(program)
        
        # Add to global archive
        self.all_programs[program.id] = program
        
        # Update global best
        if self.best_program is None or program.score > self.best_program.score:
            self.best_program = program
            return True
        
        return False
    
    def migrate(self):
        """
        Migrate top programs between islands.
        
        This is crucial for sharing good solutions while maintaining
        diversity through island isolation.
        """
        for i, island in enumerate(self.islands):
            if island.elite:
                # Send elite to neighboring islands
                neighbors = [(i - 1) % self.num_islands, (i + 1) % self.num_islands]
                for neighbor_id in neighbors:
                    if random.random() < 0.5:  # 50% migration chance
                        self.islands[neighbor_id].add_program(island.elite)
    
    def step_generation(self):
        """Advance to the next generation, potentially triggering migration."""
        self.generation += 1
        if self.generation % self.migration_interval == 0:
            self.migrate()
    
    def sample_for_prompt(
        self, 
        n_parents: int = 2,
        n_diverse: int = 3,
        include_elites: bool = True
    ) -> Dict[str, List[Program]]:
        """
        Sample programs for constructing an LLM prompt.
        
        Returns a structured sample including:
        - Parents for direct mutation
        - Diverse programs for context
        - MAP-Elites for inspiration
        """
        # Pick a random island for this sample
        island = random.choice(self.islands)
        
        result = {
            "parents": island.sample_parents(n_parents),
            "diverse": island.get_diverse_sample(n_diverse),
            "elites": self.map_elites.get_elites()[:5] if include_elites else [],
            "best": self.best_program
        }
        
        return result
    
    def get_lineage(self, program_id: str, depth: int = 3) -> List[Program]:
        """
        Get the evolutionary lineage of a program.
        
        This is used to provide the LLM with context about what
        mutations led to successful programs.
        """
        lineage = []
        visited = set()
        queue = [program_id]
        
        while queue and len(lineage) < depth * 2:
            current_id = queue.pop(0)
            if current_id in visited or current_id not in self.all_programs:
                continue
            
            visited.add(current_id)
            program = self.all_programs[current_id]
            lineage.append(program)
            queue.extend(program.parent_ids)
        
        return lineage
    
    def get_statistics(self) -> Dict:
        """Get current database statistics."""
        scores = [p.score for p in self.all_programs.values()]
        return {
            "generation": self.generation,
            "total_programs": len(self.all_programs),
            "best_score": self.best_program.score if self.best_program else 0,
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "num_map_elites": len(self.map_elites.grid),
            "island_sizes": [len(island.programs) for island in self.islands]
        }
    
    def save(self, filepath: str):
        """Save database to JSON."""
        data = {
            "generation": self.generation,
            "programs": [p.to_dict() for p in self.all_programs.values()],
            "feature_dims": self.feature_dims
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ProgramDatabase':
        """Load database from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        db = cls(feature_dims=data.get("feature_dims", ["complexity", "runtime"]))
        db.generation = data["generation"]
        
        for p_data in data["programs"]:
            program = Program(
                code=p_data["code"],
                score=p_data["score"],
                metrics=p_data.get("metrics", {}),
                generation=p_data.get("generation", 0),
                parent_ids=p_data.get("parent_ids", [])
            )
            db.add_program(program)
        
        return db
