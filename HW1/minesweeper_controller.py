"""minesweeper_controller.py — Controller (glue) for Minesweeper MVC."""

from __future__ import annotations

import time
from tkinter import messagebox

from minesweeper_ai import MinesweeperAI
from minesweeper_model import MinesweeperModel
from minesweeper_view import MinesweeperView


class MinesweeperController:
    """Nhận event từ View, gọi Model/AI và cập nhật lại View."""

    def __init__(self, root):
        self.root = root
        self.view = MinesweeperView(root)

        self.rows = 9
        self.cols = 9
        self.num_mines = 10

        self.model = None

        self.timer_running = False
        self.start_time = 0
        self.timer_id = None

        self.ai = None

        # AI autoplay (giải tự động toàn bộ)
        self.ai_autoplay_running = False

        self.view.on_left_click = self.on_left_click
        self.view.on_right_click = self.on_right_click
        self.view.on_double_click = self.on_double_click
        self.view.on_new_game = self.new_game
        self.view.on_reset = self.reset_game
        # 🤖 Solve (autoplay) + 💡 Hint (1 bước)
        self.view.on_ai_move = self.on_ai_solve
        self.view.on_ai_hint = self.on_ai_hint

        self.new_game(self.rows, self.cols, self.num_mines)

    def new_game(self, rows: int, cols: int, mines: int) -> None:
        self._stop_timer()
        self.ai_autoplay_running = False

        self.rows = rows
        self.cols = cols
        self.num_mines = mines

        self.model = MinesweeperModel(rows, cols, mines)
        self.ai = MinesweeperAI(self.model)

        self.view.create_grid(rows, cols)
        self.view.update_mine_counter(mines)
        self.view.update_timer(0)
        self.view.set_smiley("happy")

        self._center_window()

    def reset_game(self) -> None:
        self.new_game(self.rows, self.cols, self.num_mines)

    def on_left_click(self, row: int, col: int) -> None:
        if not self.model or self.model.game_over or self.model.won:
            return

        if self.model.first_click:
            self._start_timer()

        result = self.model.reveal(row, col)

        if not result:
            return

        if self.model.game_over:
            self._stop_timer()
            self.view.highlight_mine(row, col)
            for r, c, val in result:
                self.view.update_cell(r, c, val)
            self.view.show_game_over()
            return

        for r, c, val in result:
            self.view.update_cell(r, c, val)

        if self.model.won:
            self._stop_timer()
            self._auto_flag_mines()
            self.view.show_win()

    def on_right_click(self, row: int, col: int) -> None:
        if not self.model or self.model.game_over or self.model.won:
            return

        result = self.model.toggle_flag(row, col)
        if result == "flagged":
            self.view.update_flag(row, col, True)
        elif result == "hidden":
            self.view.update_flag(row, col, False)

        self.view.update_mine_counter(self.model.get_mines_remaining())

    def on_double_click(self, row: int, col: int) -> None:
        if not self.model or self.model.game_over or self.model.won:
            return

        result = self.model.chord_reveal(row, col)

        if not result:
            return

        if self.model.game_over:
            self._stop_timer()
            for r, c, val in result:
                if val == -1:
                    self.view.highlight_mine(r, c)

            all_mines = self.model._reveal_all_mines()
            for r, c, val in all_mines:
                self.view.update_cell(r, c, val)
            self.view.show_game_over()
            return

        for r, c, val in result:
            self.view.update_cell(r, c, val)

        if self.model.won:
            self._stop_timer()
            self._auto_flag_mines()
            self.view.show_win()

    def on_ai_hint(self) -> None:
        """AI đi 1 bước (step-by-step) như logic ban đầu."""
        if not self.model or self.model.game_over or self.model.won:
            return

        # không chen ngang khi autoplay đang chạy
        if self.ai_autoplay_running:
            return

        if not self.ai:
            self.ai = MinesweeperAI(self.model)

        move = self.ai.get_next_move()
        if not move:
            messagebox.showinfo(
                "🤖 AI",
                "AI không tìm được nước đi nào!\nCó thể phải đoán hoặc game đã kết thúc.",
            )
            return

        action, cells = move

        if action == "reveal":
            for r, c in cells:
                if self.model.game_over or self.model.won:
                    break
                self.view.highlight_ai_move(r, c, "reveal")
                self.on_left_click(r, c)

        elif action == "flag":
            for r, c in cells:
                if self.model.game_over or self.model.won:
                    break
                if self.model.state[r][c] == "hidden":
                    self.view.highlight_ai_move(r, c, "flag")
                    self.on_right_click(r, c)

    def on_ai_solve(self) -> None:
        if not self.model or self.model.game_over or self.model.won:
            return

        if not self.ai:
            self.ai = MinesweeperAI(self.model)

        # bấm 1 lần: AI tự chơi cho tới khi kết thúc
        if self.ai_autoplay_running:
            return
        self.ai_autoplay_running = True
        self._ai_autoplay_step()

    # backward-compat: các chỗ cũ gọi on_ai_move sẽ chạy Solve
    def on_ai_move(self) -> None:
        self.on_ai_solve()

    def _ai_autoplay_step(self) -> None:
        """Chạy AI theo từng bước bằng after() để không làm đơ UI."""
        if not self.ai_autoplay_running:
            return

        if not self.model or self.model.game_over or self.model.won:
            self.ai_autoplay_running = False
            return

        if not self.ai:
            self.ai = MinesweeperAI(self.model)

        move = self.ai.get_next_move()
        if not move:
            self.ai_autoplay_running = False
            messagebox.showinfo(
                "🤖 AI",
                "AI không tìm được nước đi nào!\nGame có thể đã kết thúc.",
            )
            return

        action, cells = move

        if action == "reveal":
            for r, c in cells:
                if self.model.game_over or self.model.won:
                    break
                self.view.highlight_ai_move(r, c, "reveal")
                self.on_left_click(r, c)

        elif action == "flag":
            for r, c in cells:
                if self.model.game_over or self.model.won:
                    break
                if self.model.state[r][c] == "hidden":
                    self.view.highlight_ai_move(r, c, "flag")
                    self.on_right_click(r, c)

        if self.model.game_over or self.model.won:
            self.ai_autoplay_running = False
            return

        # delay nhỏ để UI kịp vẽ lại và người dùng quan sát
        self.root.after(20, self._ai_autoplay_step)

    def _auto_flag_mines(self) -> None:
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.board[r][c] == -1 and self.model.state[r][c] != "flagged":
                    self.view.update_flag(r, c, True)
        self.view.update_mine_counter(0)

    def _start_timer(self) -> None:
        self.start_time = time.time()
        self.timer_running = True
        self._update_timer()

    def _stop_timer(self) -> None:
        self.timer_running = False
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None

    def _update_timer(self) -> None:
        if self.timer_running:
            elapsed = int(time.time() - self.start_time)
            self.view.update_timer(elapsed)
            self.timer_id = self.root.after(500, self._update_timer)

    def _center_window(self) -> None:
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        self.root.geometry(f"+{x}+{y}")
