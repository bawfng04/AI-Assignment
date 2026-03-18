from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaBetaConfig:
    max_depth: int = 4
    quiescence_depth: int = 3
    use_transposition_table: bool = True
    move_ordering: bool = True


@dataclass(frozen=True)
class MCTSConfig:
    iterations: int = 1500
    exploration_constant: float = 1.41421356237
    rollout_depth: int = 40


@dataclass(frozen=True)
class MatchConfig:
    max_plies: int = 300
    random_opening_plies: int = 0
    seed: int | None = 42
