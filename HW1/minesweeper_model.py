"""minesweeper_model.py — Model (game logic) for Minesweeper.

Tách từ minesweeper.py để dễ maintain/test.
Không phụ thuộc GUI.
"""

from __future__ import annotations

import random


class MinesweeperModel:
    """Logic game minesweeper (board, mine, reveal, flag, win/loss)."""

    # 8 hướng xung quanh 1 ô (trên, dưới, trái, phải, 4 đường chéo)
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def __init__(self, rows: int, cols: int, num_mines: int):
        self.rows = rows
        self.cols = cols

        # clamp số mine lại cho an toàn, không cho nhiều hơn tổng ô trừ 9
        # (9 ô = ô click + 8 ô xung quanh, để first click luôn safe)
        max_mines = rows * cols - 9
        if max_mines < 1:
            max_mines = 1
        self.num_mines = min(num_mines, max_mines)

        # board[r][c] = -1 nếu là mine, 0-8 nếu là số đếm mine xung quanh
        self.board = [[0] * cols for _ in range(rows)]

        # state[r][c] = "hidden" | "revealed" | "flagged"
        self.state = [["hidden"] * cols for _ in range(rows)]

        self.first_click = True
        self.game_over = False
        self.won = False

        self.revealed_count = 0
        self.flag_count = 0

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def generate_mines(self, safe_row: int, safe_col: int) -> None:
        """Sinh mine, đảm bảo vùng 3x3 quanh first click không có mine."""
        safe_zone = set()
        for dr, dc in self.DIRECTIONS:
            nr, nc = safe_row + dr, safe_col + dc
            if self._in_bounds(nr, nc):
                safe_zone.add((nr, nc))
        safe_zone.add((safe_row, safe_col))

        all_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in safe_zone:
                    all_cells.append((r, c))

        mine_count = min(self.num_mines, len(all_cells))
        mine_positions = random.sample(all_cells, mine_count)

        for r, c in mine_positions:
            self.board[r][c] = -1

        self._count_adjacent_mines()

    def _count_adjacent_mines(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == -1:
                    continue

                count = 0
                for dr, dc in self.DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if self._in_bounds(nr, nc) and self.board[nr][nc] == -1:
                        count += 1

                self.board[r][c] = count

    def reveal(self, row: int, col: int):
        """Mở ô (row, col). Trả về list ô đã mở hoặc list mine khi thua."""
        if self.game_over or self.won:
            return []

        if self.first_click:
            self.generate_mines(row, col)
            self.first_click = False

        if self.state[row][col] != "hidden":
            return []

        if self.board[row][col] == -1:
            self.game_over = True
            return self._reveal_all_mines()

        if self.board[row][col] > 0:
            self.state[row][col] = "revealed"
            self.revealed_count += 1
            self._check_win()
            return [(row, col, self.board[row][col])]

        stack = [(row, col)]
        visited = set()
        revealed_cells = []

        while stack:
            r, c = stack.pop()

            if (r, c) in visited:
                continue
            visited.add((r, c))

            if self.state[r][c] != "hidden":
                continue

            self.state[r][c] = "revealed"
            self.revealed_count += 1
            revealed_cells.append((r, c, self.board[r][c]))

            if self.board[r][c] == 0:
                for dr, dc in self.DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if (
                        self._in_bounds(nr, nc)
                        and (nr, nc) not in visited
                        and self.board[nr][nc] != -1
                    ):
                        stack.append((nr, nc))

        self._check_win()
        return revealed_cells

    def chord_reveal(self, row: int, col: int):
        if self.game_over or self.won:
            return []

        if self.state[row][col] != "revealed" or self.board[row][col] <= 0:
            return []

        flag_count = 0
        hidden_neighbors = []
        for dr, dc in self.DIRECTIONS:
            nr, nc = row + dr, col + dc
            if self._in_bounds(nr, nc):
                if self.state[nr][nc] == "flagged":
                    flag_count += 1
                elif self.state[nr][nc] == "hidden":
                    hidden_neighbors.append((nr, nc))

        if flag_count != self.board[row][col]:
            return []

        all_revealed = []
        for nr, nc in hidden_neighbors:
            result = self.reveal(nr, nc)
            if isinstance(result, list):
                all_revealed.extend(result)

        return all_revealed

    def toggle_flag(self, row: int, col: int):
        if self.game_over or self.won:
            return None

        if self.state[row][col] == "hidden":
            self.state[row][col] = "flagged"
            self.flag_count += 1
            return "flagged"
        if self.state[row][col] == "flagged":
            self.state[row][col] = "hidden"
            self.flag_count -= 1
            return "hidden"

        return None

    def _check_win(self) -> None:
        total_safe = self.rows * self.cols - self.num_mines
        if self.revealed_count >= total_safe:
            self.won = True

    def _reveal_all_mines(self):
        result = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == -1:
                    result.append((r, c, -1))
                elif self.state[r][c] == "flagged":
                    result.append((r, c, -2))
        return result

    def get_mines_remaining(self) -> int:
        return self.num_mines - self.flag_count
