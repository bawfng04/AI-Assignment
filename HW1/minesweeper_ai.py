"""minesweeper_ai.py — AI solver for Minesweeper.

Tách từ minesweeper.py để tách biệt thuật toán AI khỏi UI.
"""

from __future__ import annotations

import random


class MinesweeperAI:
    """AI solver dùng propositional logic + backtracking + educated guess."""

    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def __init__(self, model):
        self.model = model

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.model.rows and 0 <= c < self.model.cols

    def _get_neighbors(self, r: int, c: int):
        neighbors = []
        for dr, dc in self.DIRECTIONS:
            nr, nc = r + dr, c + dc
            if self._in_bounds(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def _get_cell_info(self, r: int, c: int):
        hidden_neighbors = []
        flagged_count = 0

        for nr, nc in self._get_neighbors(r, c):
            if self.model.state[nr][nc] == "hidden":
                hidden_neighbors.append((nr, nc))
            elif self.model.state[nr][nc] == "flagged":
                flagged_count += 1

        return hidden_neighbors, flagged_count, len(hidden_neighbors)

    def get_next_move(self):
        model = self.model

        if model.game_over or model.won:
            return None

        if model.first_click:
            center_r = model.rows // 2
            center_c = model.cols // 2
            return ("reveal", [(center_r, center_c)])

        safe_cells = set()
        mine_cells = set()

        for r in range(model.rows):
            for c in range(model.cols):
                if model.state[r][c] != "revealed" or model.board[r][c] <= 0:
                    continue

                value = model.board[r][c]
                hidden_neighbors, flagged_count, hidden_count = self._get_cell_info(r, c)

                if hidden_count == 0:
                    continue

                if value == flagged_count:
                    for nr, nc in hidden_neighbors:
                        safe_cells.add((nr, nc))

                remaining_mines = value - flagged_count
                if remaining_mines == hidden_count:
                    for nr, nc in hidden_neighbors:
                        mine_cells.add((nr, nc))

        conflict = safe_cells & mine_cells
        safe_cells -= conflict
        mine_cells -= conflict

        if mine_cells:
            return ("flag", list(mine_cells))

        if safe_cells:
            return ("reveal", list(safe_cells))

        backtrack_result = self._backtracking_solve()
        if backtrack_result:
            return backtrack_result

        return self._educated_guess()

    def _get_frontier_cells(self):
        frontier = set()
        model = self.model

        for r in range(model.rows):
            for c in range(model.cols):
                if model.state[r][c] != "hidden":
                    continue

                for nr, nc in self._get_neighbors(r, c):
                    if model.state[nr][nc] == "revealed" and model.board[nr][nc] > 0:
                        frontier.add((r, c))
                        break

        return frontier

    def _get_constraints(self):
        constraints = []
        model = self.model

        for r in range(model.rows):
            for c in range(model.cols):
                if model.state[r][c] != "revealed" or model.board[r][c] <= 0:
                    continue

                hidden_neighbors, flagged_count, hidden_count = self._get_cell_info(r, c)

                if hidden_count == 0:
                    continue

                remaining = model.board[r][c] - flagged_count
                if 0 <= remaining <= hidden_count:
                    constraints.append((set(hidden_neighbors), remaining))

        return constraints

    def _is_consistent(self, assignment, constraints) -> bool:
        for cells, remaining in constraints:
            assigned_mines = 0
            unassigned = 0

            for cell in cells:
                if cell in assignment:
                    if assignment[cell]:
                        assigned_mines += 1
                else:
                    unassigned += 1

            if assigned_mines > remaining:
                return False

            if assigned_mines + unassigned < remaining:
                return False

        return True

    def _backtracking_solve(self):
        frontier = self._get_frontier_cells()
        if not frontier:
            return None

        constraints = self._get_constraints()
        if not constraints:
            return None

        safe_cells = set()
        mine_cells = set()

        for cell in frontier:
            test_mine = {cell: True}
            mine_ok = self._is_consistent(test_mine, constraints)

            test_safe = {cell: False}
            safe_ok = self._is_consistent(test_safe, constraints)

            if mine_ok and not safe_ok:
                mine_cells.add(cell)
            elif safe_ok and not mine_ok:
                safe_cells.add(cell)

        if mine_cells:
            return ("flag", list(mine_cells))
        if safe_cells:
            return ("reveal", list(safe_cells))

        clusters = self._build_clusters(frontier, constraints)

        for cluster_cells, cluster_constraints in clusters:
            if len(cluster_cells) > 15:
                continue

            result = self._enumerate_cluster(cluster_cells, cluster_constraints)
            if result:
                return result

        return None

    def _build_clusters(self, frontier, constraints):
        parent = {cell: cell for cell in frontier}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for cells, _ in constraints:
            frontier_in_constraint = [c for c in cells if c in frontier]
            for i in range(1, len(frontier_in_constraint)):
                union(frontier_in_constraint[0], frontier_in_constraint[i])

        from collections import defaultdict

        groups = defaultdict(set)
        for cell in frontier:
            groups[find(cell)].add(cell)

        result = []
        for cluster_cells in groups.values():
            cluster_constraints = []
            for cells, remaining in constraints:
                if cells & cluster_cells:
                    cluster_constraints.append((cells & cluster_cells, remaining))
            result.append((list(cluster_cells), cluster_constraints))

        return result

    def _enumerate_cluster(self, cluster_cells, constraints):
        n = len(cluster_cells)
        mine_count = {cell: 0 for cell in cluster_cells}
        total_valid = 0

        stack = [(0, {})]

        while stack:
            idx, assignment = stack.pop()

            if idx == n:
                valid = True
                for cells, remaining in constraints:
                    mine_in_constraint = sum(1 for c in cells if assignment.get(c, False))
                    if mine_in_constraint != remaining:
                        valid = False
                        break

                if valid:
                    total_valid += 1
                    for cell in cluster_cells:
                        if assignment.get(cell, False):
                            mine_count[cell] += 1
                continue

            cell = cluster_cells[idx]

            new_assignment_safe = dict(assignment)
            new_assignment_safe[cell] = False
            if self._is_consistent(new_assignment_safe, constraints):
                stack.append((idx + 1, new_assignment_safe))

            new_assignment_mine = dict(assignment)
            new_assignment_mine[cell] = True
            if self._is_consistent(new_assignment_mine, constraints):
                stack.append((idx + 1, new_assignment_mine))

        if total_valid == 0:
            return None

        safe_cells = []
        mine_cells_result = []

        for cell in cluster_cells:
            if mine_count[cell] == 0:
                safe_cells.append(cell)
            elif mine_count[cell] == total_valid:
                mine_cells_result.append(cell)

        if mine_cells_result:
            return ("flag", mine_cells_result)
        if safe_cells:
            return ("reveal", safe_cells)

        return None

    def _educated_guess(self):
        model = self.model
        frontier = self._get_frontier_cells()

        all_hidden = []
        for r in range(model.rows):
            for c in range(model.cols):
                if model.state[r][c] == "hidden":
                    all_hidden.append((r, c))

        if not all_hidden:
            return None

        non_frontier = [c for c in all_hidden if c not in frontier]

        if non_frontier:
            cell = random.choice(non_frontier)
            return ("reveal", [cell])

        if frontier:
            best_cell = None
            best_prob = 2.0

            for cell in frontier:
                r, c = cell
                max_prob = 0.0
                count = 0

                for nr, nc in self._get_neighbors(r, c):
                    if model.state[nr][nc] == "revealed" and model.board[nr][nc] > 0:
                        _, fc, hc = self._get_cell_info(nr, nc)
                        if hc > 0:
                            prob = (model.board[nr][nc] - fc) / hc
                            max_prob = max(max_prob, prob)
                            count += 1

                if count > 0 and max_prob < best_prob:
                    best_prob = max_prob
                    best_cell = cell

            if best_cell:
                return ("reveal", [best_cell])

        cell = random.choice(all_hidden)
        return ("reveal", [cell])
