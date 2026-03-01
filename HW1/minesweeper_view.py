"""minesweeper_view.py — Tkinter UI (View) for Minesweeper.

Tách từ minesweeper.py để UI độc lập với logic/AI.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox


class MinesweeperView:
    """GUI layer: render grid buttons, menu, status bar, dialogs."""

    NUMBER_COLORS = {
        1: "#0000FF",
        2: "#008000",
        3: "#FF0000",
        4: "#000080",
        5: "#800000",
        6: "#008080",
        7: "#000000",
        8: "#808080",
    }

    CELL_SIZE = 30

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Minesweeper — AI Assignment")
        self.root.resizable(False, False)
        self.root.configure(bg="#C0C0C0")

        self.on_left_click = None
        self.on_right_click = None
        self.on_double_click = None
        self.on_new_game = None
        self.on_reset = None
        self.on_ai_move = None

        self.buttons = []

        self.top_frame = tk.Frame(root, bg="#C0C0C0", bd=3, relief=tk.SUNKEN)
        self.top_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        # dùng grid cho top bar để không bị giãn nút khi cửa sổ rộng (16×16, 16×30)
        self.top_frame.grid_columnconfigure(0, weight=0)
        self.top_frame.grid_columnconfigure(1, weight=1)
        self.top_frame.grid_columnconfigure(2, weight=0)

        self.mine_label = tk.Label(
            self.top_frame,
            text="000",
            font=("Consolas", 18, "bold"),
            fg="#FF0000",
            bg="#000000",
            width=4,
            anchor="center",
            relief=tk.SUNKEN,
            bd=2,
        )
        self.mine_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # nhóm nút ở giữa (reset + AI) để luôn nằm giữa top bar
        self.center_frame = tk.Frame(self.top_frame, bg="#C0C0C0")
        self.center_frame.grid(row=0, column=1, padx=5, pady=5)

        self.reset_button = tk.Button(
            self.center_frame,
            text="🙂",
            font=("Segoe UI Emoji", 16),
            width=2,
            height=1,
            relief=tk.RAISED,
            bd=2,
            bg="#C0C0C0",
            command=self._on_reset_click,
        )
        self.reset_button.pack(side=tk.LEFT, padx=(0, 6))

        self.ai_button = tk.Button(
            self.center_frame,
            text="🤖",
            font=("Segoe UI Emoji", 14),
            width=2,
            height=1,
            relief=tk.RAISED,
            bd=2,
            bg="#90EE90",
            activebackground="#7CCD7C",
            command=self._on_ai_click,
        )
        self.ai_button.pack(side=tk.LEFT)

        self.timer_label = tk.Label(
            self.top_frame,
            text="000",
            font=("Consolas", 18, "bold"),
            fg="#FF0000",
            bg="#000000",
            width=4,
            anchor="center",
            relief=tk.SUNKEN,
            bd=2,
        )
        self.timer_label.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        self.grid_frame = tk.Frame(root, bg="#C0C0C0", bd=3, relief=tk.SUNKEN)
        self.grid_frame.pack(padx=5, pady=5)

        self._create_menu()

    def _create_menu(self) -> None:
        menubar = tk.Menu(self.root)

        game_menu = tk.Menu(menubar, tearoff=0)
        game_menu.add_command(
            label="Dễ (9×9, 10 mìn)",
            command=lambda: self._trigger_new_game(9, 9, 10),
        )
        game_menu.add_command(
            label="Trung bình (16×16, 40 mìn)",
            command=lambda: self._trigger_new_game(16, 16, 40),
        )
        game_menu.add_command(
            label="Khó (16×30, 99 mìn)",
            command=lambda: self._trigger_new_game(16, 30, 99),
        )
        game_menu.add_separator()
        game_menu.add_command(label="Tùy chỉnh...", command=self._show_custom_dialog)
        game_menu.add_separator()
        game_menu.add_command(label="Thoát", command=self.root.quit)
        menubar.add_cascade(label="Game", menu=game_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Cách chơi", command=self._show_help)
        help_menu.add_command(label="Thuật toán DFS", command=self._show_dfs_info)
        help_menu.add_command(label="AI Solver", command=self._show_ai_info)
        help_menu.add_command(label="Thông tin", command=self._show_about)
        menubar.add_cascade(label="Hướng dẫn", menu=help_menu)

        self.root.config(menu=menubar)

    def _trigger_new_game(self, rows: int, cols: int, mines: int) -> None:
        if self.on_new_game:
            self.on_new_game(rows, cols, mines)

    def _on_reset_click(self) -> None:
        if self.on_reset:
            self.on_reset()

    def _on_ai_click(self) -> None:
        if self.on_ai_move:
            self.on_ai_move()

    def _show_custom_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Tùy chỉnh kích thước")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        dialog.geometry(
            "+%d+%d" % (self.root.winfo_x() + 50, self.root.winfo_y() + 50)
        )

        frame = tk.Frame(dialog, padx=15, pady=15)
        frame.pack()

        tk.Label(frame, text="Số hàng (5-30):", font=("Segoe UI", 10)).grid(
            row=0, column=0, sticky="w", pady=3
        )
        row_var = tk.StringVar(value="9")
        tk.Entry(frame, textvariable=row_var, width=8, font=("Segoe UI", 10)).grid(
            row=0, column=1, padx=(10, 0), pady=3
        )

        tk.Label(frame, text="Số cột (5-30):", font=("Segoe UI", 10)).grid(
            row=1, column=0, sticky="w", pady=3
        )
        col_var = tk.StringVar(value="9")
        tk.Entry(frame, textvariable=col_var, width=8, font=("Segoe UI", 10)).grid(
            row=1, column=1, padx=(10, 0), pady=3
        )

        tk.Label(frame, text="Số mìn:", font=("Segoe UI", 10)).grid(
            row=2, column=0, sticky="w", pady=3
        )
        mine_var = tk.StringVar(value="10")
        tk.Entry(frame, textvariable=mine_var, width=8, font=("Segoe UI", 10)).grid(
            row=2, column=1, padx=(10, 0), pady=3
        )

        def on_ok():
            try:
                rows = int(row_var.get())
                cols = int(col_var.get())
                mines = int(mine_var.get())

                rows = max(5, min(30, rows))
                cols = max(5, min(30, cols))
                mines = max(1, min(rows * cols - 9, mines))

                dialog.destroy()
                self._trigger_new_game(rows, cols, mines)
            except ValueError:
                messagebox.showwarning(
                    "Lỗi",
                    "Vui lòng nhập số nguyên hợp lệ!",
                    parent=dialog,
                )

        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))

        tk.Button(btn_frame, text="OK", width=8, command=on_ok).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(btn_frame, text="Hủy", width=8, command=dialog.destroy).pack(
            side=tk.LEFT, padx=5
        )

        dialog.bind("<Return>", lambda e: on_ok())

    def _show_help(self) -> None:
        help_text = (
            "🎮 CÁCH CHƠI MINESWEEPER\n\n"
            "🖱️ Click trái: Mở ô\n"
            "🖱️ Click phải: Cắm/bỏ cờ 🚩\n"
            "🖱️ Double click trái: Chord reveal\n"
            "   (mở nhanh các ô xung quanh ô số\n"
            "    khi đã cắm đủ cờ)\n\n"
            "📋 LUẬT CHƠI:\n"
            "• Mở tất cả ô không có mìn để thắng\n"
            "• Số trên ô = số mìn xung quanh (8 hướng)\n"
            "• Ô trống (0) sẽ tự mở các ô trống liền kề\n"
            "• Click đầu tiên luôn an toàn\n\n"
            "⚙️ THUẬT TOÁN:\n"
            "• Flood Fill dùng DFS (Depth-First Search)\n"
            "• Cài đặt iterative bằng Stack (LIFO)"
        )
        messagebox.showinfo("Hướng dẫn", help_text)

    def _show_dfs_info(self) -> None:
        dfs_text = (
            "🔍 THUẬT TOÁN DFS FLOOD FILL\n\n"
            "Depth-First Search (DFS) là thuật toán\n"
            "duyệt đồ thị đi sâu trước.\n\n"
            "📐 CÀI ĐẶT TRONG MINESWEEPER:\n"
            "• Dùng Stack (LIFO) — iterative\n"
            "• Không dùng đệ quy (tránh stack overflow)\n"
            "• Mỗi ô là 1 node trong đồ thị\n"
            "• 8 ô xung quanh là các cạnh kề\n\n"
            "📝 PSEUDOCODE:\n"
            "  stack = [(start_row, start_col)]\n"
            "  while stack not empty:\n"
            "      (r, c) = stack.pop()  ← LIFO\n"
            "      if đã visit → skip\n"
            "      mark revealed\n"
            "      if value == 0:  ← ô trống\n"
            "          push 8 neighbors vào stack\n\n"
            "⏱️ ĐỘ PHỨC TẠP:\n"
            "• Thời gian: O(V + E) — V = số ô, E = số cạnh\n"
            "• Không gian: O(V) — stack + visited set"
        )
        messagebox.showinfo("Thuật toán DFS", dfs_text)

    def _show_ai_info(self) -> None:
        ai_text = (
            "🤖 AI SOLVER — AUTO MINESWEEPER\n\n"
            "AI sử dụng 3 tầng suy luận:\n\n"
            "📐 TẦNG 1: Rule-Based (Propositional Logic)\n"
            "• Rule 1 (All Safe): Nếu số = flag count\n"
            "  → Tất cả hidden neighbors an toàn\n"
            "• Rule 2 (All Mines): Nếu số - flags = hidden\n"
            "  → Tất cả hidden neighbors là mìn\n\n"
            "🔍 TẦNG 2: DFS Backtracking\n"
            "• Thử gán mine/safe cho frontier cells\n"
            "• Kiểm tra tính nhất quán (consistency)\n"
            "• Enumerate tổ hợp cho cluster nhỏ\n\n"
            "🎲 TẦNG 3: Educated Guess\n"
            "• Chọn ô có xác suất mìn thấp nhất\n"
            "• Ưu tiên ô không giáp ô đã mở\n\n"
            "💡 CÁCH DÙNG:\n"
            "• Click 🤖 để AI đi 1 nước\n"
            "• AI sẽ reveal hoặc flag tùy logic"
        )
        messagebox.showinfo("AI Solver", ai_text)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "Thông tin",
            "💣 Minesweeper — AI Assignment\n\n"
            "Môn: Nhập môn Trí tuệ Nhân tạo (CO3061)\n\n"
            "🛠️ Công nghệ:\n"
            "• Python + Tkinter\n"
            "• DFS Iterative Flood Fill\n"
            "• AI Solver (Logic + Backtracking)\n"
            "• Kiến trúc MVC\n\n"
            "📚 Thuật toán:\n"
            "• DFS dùng Stack (LIFO) cho Flood Fill\n"
            "• Propositional Logic cho AI Solver\n"
            "• DFS Backtracking cho constraint solving",
        )

    def create_grid(self, rows: int, cols: int) -> None:
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.buttons = []

        for r in range(rows):
            row_buttons = []
            for c in range(cols):
                btn = tk.Button(
                    self.grid_frame,
                    width=2,
                    height=1,
                    font=("Consolas", 10, "bold"),
                    relief=tk.RAISED,
                    bd=2,
                    bg="#C0C0C0",
                    activebackground="#D0D0D0",
                )
                btn.grid(row=r, column=c, padx=0, pady=0)

                btn.bind(
                    "<Button-1>",
                    lambda e, r=r, c=c: self._handle_left_click(r, c),
                )
                btn.bind(
                    "<Button-3>",
                    lambda e, r=r, c=c: self._handle_right_click(r, c),
                )
                btn.bind(
                    "<Double-Button-1>",
                    lambda e, r=r, c=c: self._handle_double_click(r, c),
                )

                row_buttons.append(btn)
            self.buttons.append(row_buttons)

        self.root.update_idletasks()

    def _handle_left_click(self, row: int, col: int) -> None:
        if self.on_left_click:
            self.on_left_click(row, col)

    def _handle_right_click(self, row: int, col: int) -> None:
        if self.on_right_click:
            self.on_right_click(row, col)

    def _handle_double_click(self, row: int, col: int) -> None:
        if self.on_double_click:
            self.on_double_click(row, col)

    def update_cell(self, row: int, col: int, value: int) -> None:
        btn = self.buttons[row][col]

        if value == -1:
            btn.configure(
                text="💣",
                font=("Segoe UI Emoji", 10),
                relief=tk.SUNKEN,
                bg="#FF0000",
                disabledforeground="#000000",
                state=tk.DISABLED,
            )
        elif value == -2:
            btn.configure(
                text="❌",
                font=("Segoe UI Emoji", 10),
                relief=tk.SUNKEN,
                bg="#FFA500",
                disabledforeground="#000000",
                state=tk.DISABLED,
            )
        elif value == 0:
            btn.configure(
                text="",
                relief=tk.SUNKEN,
                bg="#D9D9D9",
                state=tk.DISABLED,
            )
        else:
            color = self.NUMBER_COLORS.get(value, "#000000")
            btn.configure(
                text=str(value),
                font=("Consolas", 10, "bold"),
                relief=tk.SUNKEN,
                bg="#D9D9D9",
                disabledforeground=color,
                state=tk.DISABLED,
            )

    def update_flag(self, row: int, col: int, flagged: bool) -> None:
        btn = self.buttons[row][col]
        if flagged:
            btn.configure(text="🚩", font=("Segoe UI Emoji", 10), fg="#FF0000")
        else:
            btn.configure(text="", font=("Consolas", 10, "bold"), fg="#000000")

    def update_mine_counter(self, count: int) -> None:
        if count < 0:
            text = "-" + str(abs(count)).zfill(2)
        else:
            text = str(count).zfill(3)
        self.mine_label.configure(text=text)

    def update_timer(self, seconds: int) -> None:
        seconds = min(seconds, 999)
        self.timer_label.configure(text=str(seconds).zfill(3))

    def set_smiley(self, face: str) -> None:
        faces = {
            "happy": "🙂",
            "dead": "😵",
            "cool": "😎",
            "surprised": "😮",
        }
        self.reset_button.configure(text=faces.get(face, "🙂"))

    def highlight_ai_move(self, row: int, col: int, action: str) -> None:
        btn = self.buttons[row][col]
        if action == "reveal":
            btn.configure(bg="#90EE90")
        elif action == "flag":
            btn.configure(bg="#FFD700")

    def highlight_mine(self, row: int, col: int) -> None:
        btn = self.buttons[row][col]
        btn.configure(bg="#CC0000")

    def show_win(self) -> None:
        self.set_smiley("cool")
        messagebox.showinfo(
            "🎉 Thắng rồi!",
            "Chúc mừng! Bạn đã dò hết mìn!\n\n🏆 Bạn thắng game Minesweeper!",
        )

    def show_game_over(self) -> None:
        self.set_smiley("dead")
        messagebox.showinfo(
            "💥 Thua rồi!",
            "Bạn đã dẫm trúng mìn!\n\n💀 Game Over — thử lại nhé!",
        )
