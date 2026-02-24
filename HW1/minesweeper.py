"""
minesweeper.py — minesweeper dùng tkinter + DFS flood fill

kiến trúc MVC:
    - MinesweeperModel: xử lý logic game (board, mine, reveal, flag, win/loss)
    - MinesweeperView: giao diện tkinter (grid nút bấm, menu, status bar, timer)
    - MinesweeperController: kết nối Model với View, xử lý event click

thuật toán chính: DFS iterative dùng stack (LIFO) cho flood fill
    - không dùng recursion vì python giới hạn đệ quy
    - stack tự quản lý nên không lo stack overflow với grid lớn

"""

import tkinter as tk
from tkinter import messagebox, simpledialog
import random
import time


# ===========================================================================
#  PHẦN 1: MODEL — logic game thuần, không liên quan gì tới GUI
# ===========================================================================

class MinesweeperModel:
    """
    class chứa toàn bộ logic của game minesweeper.
    tách riêng ra khỏi GUI để dễ test và maintain.
    kiểu MVC đó, thầy thích cái này lắm.
    """

    # 8 hướng xung quanh 1 ô (trên, dưới, trái, phải, 4 đường chéo)
    # dùng tuple
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    def __init__(self, rows, cols, num_mines):
        """
        khởi tạo model với kích thước grid và số mine.
        chưa generate mine ở đây vì phải đợi first click đã (first click safe).

        params:
            rows: số hàng
            cols: số cột
            num_mines: số mìn cần đặt
        """
        self.rows = rows
        self.cols = cols
        # clamp số mine lại cho an toàn, không cho nhiều hơn tổng ô trừ 9
        # (9 ô = ô click + 8 ô xung quanh, để first click luôn safe)
        max_mines = rows * cols - 9
        if max_mines < 1:
            max_mines = 1
        self.num_mines = min(num_mines, max_mines)

        # board[r][c] = -1 nếu là mine, 0-8 nếu là số đếm mine xung quanh
        # ban đầu tất cả bằng 0, sẽ generate mine sau first click
        self.board = [[0] * cols for _ in range(rows)]

        # state[r][c] = trạng thái hiển thị của ô:
        #   "hidden"   — chưa mở
        #   "revealed" — đã mở
        #   "flagged"  — đã cắm cờ
        self.state = [["hidden"] * cols for _ in range(rows)]

        # flag theo dõi first click (để biết khi nào generate mine)
        self.first_click = True

        # trạng thái game
        self.game_over = False
        self.won = False

        # đếm số ô đã reveal (dùng check win nhanh hơn)
        self.revealed_count = 0

        # đếm số flag đã đặt (hiển thị mine counter)
        self.flag_count = 0

    def _in_bounds(self, r, c):
        """
        check xem (r, c) có nằm trong grid không.
        chỗ này quan trọng lắm, không check là out of range ngay.
        """
        return 0 <= r < self.rows and 0 <= c < self.cols

    def generate_mines(self, safe_row, safe_col):
        """
        sinh mine ngẫu nhiên trên board, ĐẢM BẢO ô (safe_row, safe_col)
        và 8 ô xung quanh nó KHÔNG có mine.

        logic:
            1. tạo danh sách tất cả ô trên board
            2. loại bỏ ô safe và 8 neighbor của nó
            3. random chọn num_mines ô từ danh sách còn lại
            4. đặt mine (-1) vào board
            5. tính số mine xung quanh cho từng ô không phải mine

        tại sao phải safe zone 3x3?
            - vì nếu chỉ safe 1 ô thì first click có thể ra số 8
            - safe cả vùng 3x3 thì first click luôn trigger flood fill
        """
        # tạo safe zone: ô được click + 8 ô xung quanh
        safe_zone = set()
        for dr, dc in self.DIRECTIONS:
            nr, nc = safe_row + dr, safe_col + dc
            if self._in_bounds(nr, nc):
                safe_zone.add((nr, nc))
        safe_zone.add((safe_row, safe_col))

        # tạo list tất cả ô có thể đặt mine (trừ safe zone)
        all_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in safe_zone:
                    all_cells.append((r, c))

        # random chọn vị trí mine
        # nếu không đủ ô thì lấy hết (edge case grid quá nhỏ)
        mine_count = min(self.num_mines, len(all_cells))
        mine_positions = random.sample(all_cells, mine_count)

        # đặt mine lên board
        for r, c in mine_positions:
            self.board[r][c] = -1

        # tính số mine xung quanh cho từng ô không phải mine
        # scan toàn bộ board, đếm neighbor có mine
        self._count_adjacent_mines()

    def _count_adjacent_mines(self):
        """
        duyệt từng ô trên board, nếu ô đó không phải mine thì
        đếm xem xung quanh nó có bao nhiêu mine (8 hướng).

        complexity: O(rows * cols * 8) = O(rows * cols) — fast enough
        """
        for r in range(self.rows):
            for c in range(self.cols):
                # mine thì skip, không cần đếm
                if self.board[r][c] == -1:
                    continue

                count = 0
                for dr, dc in self.DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    # check boundary trước khi access, tránh index out of range
                    if self._in_bounds(nr, nc) and self.board[nr][nc] == -1:
                        count += 1

                self.board[r][c] = count

    def reveal(self, row, col):
        """
        mở ô (row, col). đây là hàm quan trọng nhất.

        trường hợp:
            1. ô đã revealed hoặc flagged → bỏ qua
            2. ô là mine → game over, trả về danh sách tất cả mine
            3. ô có số > 0 → chỉ mở ô đó
            4. ô trống (0) → DFS FLOOD FILL mở tất cả ô trống liền kề

        returns:
            list of (row, col, value) — các ô đã được mở
            hoặc "mine" nếu dính mine
        """
        # không cho click khi game đã kết thúc
        if self.game_over or self.won:
            return []

        # first click → generate mine trước
        if self.first_click:
            self.generate_mines(row, col)
            self.first_click = False

        # ô đã mở hoặc đang cắm cờ → bỏ qua
        if self.state[row][col] != "hidden":
            return []

        # dính mine → game over
        if self.board[row][col] == -1:
            self.game_over = True
            return self._reveal_all_mines()

        # bắt đầu reveal
        # nếu ô có số > 0 → chỉ mở ô đó, không flood fill
        if self.board[row][col] > 0:
            self.state[row][col] = "revealed"
            self.revealed_count += 1
            self._check_win()
            return [(row, col, self.board[row][col])]

        # =====================================================
        #  DFS FLOOD FILL — dùng stack iterative
        #  đây là phần QUAN TRỌNG NHẤT của bài
        #
        #  tại sao dùng stack (LIFO) thay vì recursion?
        #    - python có recursion limit mặc định ~1000
        #    - grid 30x16 = 480 ô, recursion có thể vượt limit
        #    - stack tự quản lý → không bao giờ bị stack overflow
        #    - vẫn đúng thứ tự DFS vì pop từ cuối (LIFO)
        #
        #  thuật toán:
        #    1. push ô bắt đầu vào stack
        #    2. pop 1 ô ra, mark revealed
        #    3. nếu ô đó có giá trị 0 (trống):
        #       → push tất cả neighbor chưa visit vào stack
        #    4. nếu ô đó có giá trị > 0 (số):
        #       → chỉ reveal ô đó, KHÔNG push neighbor
        #    5. lặp lại bước 2 cho đến khi stack rỗng
        # =====================================================

        stack = [(row, col)]  # stack dùng list, append = push, pop = pop
        visited = set()       # set để track ô đã xử lý, tránh visit lại
        revealed_cells = []   # danh sách ô đã mở, trả về cho view update

        while stack:
            # pop từ cuối → LIFO → DFS
            r, c = stack.pop()

            # đã visit rồi thì skip, tránh xử lý lại
            if (r, c) in visited:
                continue
            visited.add((r, c))

            # chỉ xử lý ô hidden, ô đã revealed hoặc flagged thì bỏ qua
            if self.state[r][c] != "hidden":
                continue

            # mark ô là revealed
            self.state[r][c] = "revealed"
            self.revealed_count += 1
            revealed_cells.append((r, c, self.board[r][c]))

            # NẾU ô trống (giá trị 0) → push 8 neighbor vào stack
            # NẾU ô có số (1-8) → dừng lại, không expand thêm
            # đây là điểm khác biệt giữa flood fill và BFS/DFS thông thường:
            # ô biên (có số) vẫn được reveal nhưng KHÔNG tiếp tục expand
            if self.board[r][c] == 0:
                for dr, dc in self.DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    # check boundary + chưa visit + không phải mine
                    if (self._in_bounds(nr, nc) and
                            (nr, nc) not in visited and
                            self.board[nr][nc] != -1):
                        stack.append((nr, nc))

        # check win sau khi flood fill xong
        self._check_win()
        return revealed_cells

    def chord_reveal(self, row, col):
        """
        chord reveal: khi click vào ô số đã revealed mà xung quanh đủ flag
        → tự động mở các ô hidden xung quanh.

        đây là feature nâng cao, nhiều người chơi minesweeper pro hay dùng.
        giúp chơi nhanh hơn nhiều.

        logic:
            1. đếm số flag xung quanh ô
            2. nếu = giá trị ô → reveal tất cả ô hidden xung quanh
        """
        if self.game_over or self.won:
            return []

        # chỉ chord ô đã revealed và có số > 0
        if self.state[row][col] != "revealed" or self.board[row][col] <= 0:
            return []

        # đếm flag xung quanh
        flag_count = 0
        hidden_neighbors = []
        for dr, dc in self.DIRECTIONS:
            nr, nc = row + dr, col + dc
            if self._in_bounds(nr, nc):
                if self.state[nr][nc] == "flagged":
                    flag_count += 1
                elif self.state[nr][nc] == "hidden":
                    hidden_neighbors.append((nr, nc))

        # nếu số flag != giá trị ô → không chord
        if flag_count != self.board[row][col]:
            return []

        # reveal tất cả ô hidden xung quanh
        all_revealed = []
        for nr, nc in hidden_neighbors:
            result = self.reveal(nr, nc)
            if isinstance(result, list):
                all_revealed.extend(result)

        return all_revealed

    def toggle_flag(self, row, col):
        """
        toggle cờ trên ô (row, col).
        - hidden → flagged (cắm cờ)
        - flagged → hidden (bỏ cờ)
        - revealed → bỏ qua (không flag ô đã mở)

        returns: trạng thái mới của ô ("flagged", "hidden", hoặc None)
        """
        if self.game_over or self.won:
            return None

        if self.state[row][col] == "hidden":
            self.state[row][col] = "flagged"
            self.flag_count += 1
            return "flagged"
        elif self.state[row][col] == "flagged":
            self.state[row][col] = "hidden"
            self.flag_count -= 1
            return "hidden"

        return None

    def _check_win(self):
        """
        check điều kiện thắng: tất cả ô KHÔNG PHẢI mine đều đã revealed.
        tức là: revealed_count == tổng ô - số mine

        dùng counter thay vì duyệt toàn bộ board mỗi lần check
        → O(1) thay vì O(rows * cols), hiệu quả hơn nhiều
        """
        total_safe = self.rows * self.cols - self.num_mines
        if self.revealed_count >= total_safe:
            self.won = True

    def _reveal_all_mines(self):
        """
        khi thua → mở tất cả mine để người chơi thấy.
        cũng đánh dấu những ô flag sai (flag mà không phải mine).

        returns: list of (row, col, value) cho tất cả mine
                 value = -1 cho mine, -2 cho flag sai (wrong flag)
        """
        result = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == -1:
                    # mine: hiển thị mine
                    result.append((r, c, -1))
                elif self.state[r][c] == "flagged":
                    # flag sai: ô được flag nhưng không phải mine
                    result.append((r, c, -2))
        return result

    def get_mines_remaining(self):
        """
        tính số mine còn lại = tổng mine - số flag đã đặt.
        có thể âm nếu flag nhiều hơn mine (người chơi flag sai).
        """
        return self.num_mines - self.flag_count


# ===========================================================================
#  PHẦN 2: VIEW — giao diện tkinter, chỉ lo hiển thị, không xử lý logic
# ===========================================================================

class MinesweeperView:
    """
    class quản lý toàn bộ giao diện GUI.
    dùng tkinter grid layout cho bàn cờ.
    mỗi ô là 1 tk.Button, click thì gửi event cho controller xử lý.

    tại sao dùng Button thay vì Canvas?
        - Button có sẵn event binding, dễ handle click
        - không cần vẽ border, highlight tự có
        - code đơn giản hơn Canvas rất nhiều
        - trade-off: hơi chậm với grid rất lớn, nhưng 30x16 thì ok
    """

    # bảng màu cho các số trên ô (1-8)
    # classic minesweeper colors, nhìn quen thuộc
    NUMBER_COLORS = {
        1: "#0000FF",   # xanh dương — 1 mine xung quanh
        2: "#008000",   # xanh lá — 2
        3: "#FF0000",   # đỏ — 3
        4: "#000080",   # xanh navy — 4
        5: "#800000",   # đỏ đậm — 5
        6: "#008080",   # teal — 6
        7: "#000000",   # đen — 7
        8: "#808080",   # xám — 8
    }

    # kích thước ô (pixel), font size sẽ scale theo
    CELL_SIZE = 30

    def __init__(self, root):
        """
        khởi tạo view với root window.
        chưa tạo grid ở đây, đợi controller gọi create_grid().

        params:
            root: tk.Tk() — cửa sổ chính
        """
        self.root = root
        self.root.title("Minesweeper — AI Assignment")
        self.root.resizable(False, False)

        # configure style cho đẹp hơn default tkinter
        self.root.configure(bg="#C0C0C0")

        # callback functions — controller sẽ set sau
        self.on_left_click = None
        self.on_right_click = None
        self.on_double_click = None
        self.on_new_game = None
        self.on_reset = None

        # grid buttons — 2D list, tạo khi create_grid()
        self.buttons = []

        # frame cho status bar (mine counter, smiley, timer)
        self.top_frame = tk.Frame(root, bg="#C0C0C0", bd=3, relief=tk.SUNKEN)
        self.top_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        # mine counter bên trái
        self.mine_label = tk.Label(
            self.top_frame,
            text="000",
            font=("Consolas", 18, "bold"),
            fg="#FF0000",
            bg="#000000",
            width=4,
            anchor="center",
            relief=tk.SUNKEN,
            bd=2
        )
        self.mine_label.pack(side=tk.LEFT, padx=5, pady=5)

        # nút reset ở giữa (mặt cười)
        self.reset_button = tk.Button(
            self.top_frame,
            text="🙂",
            font=("Segoe UI Emoji", 16),
            width=2,
            height=1,
            relief=tk.RAISED,
            bd=2,
            bg="#C0C0C0",
            command=self._on_reset_click
        )
        self.reset_button.pack(side=tk.LEFT, expand=True, padx=5, pady=5)

        # timer bên phải
        self.timer_label = tk.Label(
            self.top_frame,
            text="000",
            font=("Consolas", 18, "bold"),
            fg="#FF0000",
            bg="#000000",
            width=4,
            anchor="center",
            relief=tk.SUNKEN,
            bd=2
        )
        self.timer_label.pack(side=tk.RIGHT, padx=5, pady=5)

        # frame cho grid bàn cờ
        self.grid_frame = tk.Frame(root, bg="#C0C0C0", bd=3, relief=tk.SUNKEN)
        self.grid_frame.pack(padx=5, pady=5)

        # tạo menu bar
        self._create_menu()

    def _create_menu(self):
        """
        tạo menu bar với các preset difficulty và custom option.
        menu Game: New Game (các difficulty), Custom, Exit
        """
        menubar = tk.Menu(self.root)

        # menu Game
        game_menu = tk.Menu(menubar, tearoff=0)
        game_menu.add_command(
            label="Dễ (9×9, 10 mìn)",
            command=lambda: self._trigger_new_game(9, 9, 10)
        )
        game_menu.add_command(
            label="Trung bình (16×16, 40 mìn)",
            command=lambda: self._trigger_new_game(16, 16, 40)
        )
        game_menu.add_command(
            label="Khó (16×30, 99 mìn)",
            command=lambda: self._trigger_new_game(16, 30, 99)
        )
        game_menu.add_separator()
        game_menu.add_command(
            label="Tùy chỉnh...",
            command=self._show_custom_dialog
        )
        game_menu.add_separator()
        game_menu.add_command(
            label="Thoát",
            command=self.root.quit
        )
        menubar.add_cascade(label="Game", menu=game_menu)

        # menu Hướng dẫn
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="Cách chơi",
            command=self._show_help
        )
        help_menu.add_command(
            label="Thuật toán DFS",
            command=self._show_dfs_info
        )
        help_menu.add_command(
            label="Thông tin",
            command=self._show_about
        )
        menubar.add_cascade(label="Hướng dẫn", menu=help_menu)

        self.root.config(menu=menubar)

    def _trigger_new_game(self, rows, cols, mines):
        """
        gọi callback new game từ controller.
        tách ra function riêng cho menu command dễ bind lambda.
        """
        if self.on_new_game:
            self.on_new_game(rows, cols, mines)

    def _on_reset_click(self):
        """gọi callback reset game (chơi lại cùng difficulty)."""
        if self.on_reset:
            self.on_reset()

    def _show_custom_dialog(self):
        """
        hiện dialog để người chơi nhập kích thước grid tùy ý.
        dùng tkinter Toplevel thay vì simpledialog vì cần nhiều input.
        validate input: min 5, max 30 cho rows/cols, mine tối thiểu 1.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Tùy chỉnh kích thước")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # căn giữa dialog
        dialog.geometry("+%d+%d" % (
            self.root.winfo_x() + 50,
            self.root.winfo_y() + 50
        ))

        # frame chứa input
        frame = tk.Frame(dialog, padx=15, pady=15)
        frame.pack()

        # input hàng
        tk.Label(frame, text="Số hàng (5-30):", font=("Segoe UI", 10)).grid(
            row=0, column=0, sticky="w", pady=3
        )
        row_var = tk.StringVar(value="9")
        tk.Entry(frame, textvariable=row_var, width=8, font=("Segoe UI", 10)).grid(
            row=0, column=1, padx=(10, 0), pady=3
        )

        # input cột
        tk.Label(frame, text="Số cột (5-30):", font=("Segoe UI", 10)).grid(
            row=1, column=0, sticky="w", pady=3
        )
        col_var = tk.StringVar(value="9")
        tk.Entry(frame, textvariable=col_var, width=8, font=("Segoe UI", 10)).grid(
            row=1, column=1, padx=(10, 0), pady=3
        )

        # input mine
        tk.Label(frame, text="Số mìn:", font=("Segoe UI", 10)).grid(
            row=2, column=0, sticky="w", pady=3
        )
        mine_var = tk.StringVar(value="10")
        tk.Entry(frame, textvariable=mine_var, width=8, font=("Segoe UI", 10)).grid(
            row=2, column=1, padx=(10, 0), pady=3
        )

        def on_ok():
            """validate input rồi tạo game mới."""
            try:
                rows = int(row_var.get())
                cols = int(col_var.get())
                mines = int(mine_var.get())

                # clamp giá trị cho hợp lệ
                rows = max(5, min(30, rows))
                cols = max(5, min(30, cols))
                mines = max(1, min(rows * cols - 9, mines))

                dialog.destroy()
                self._trigger_new_game(rows, cols, mines)
            except ValueError:
                # nhập sai format → dùng default
                messagebox.showwarning(
                    "Lỗi",
                    "Vui lòng nhập số nguyên hợp lệ!",
                    parent=dialog
                )

        # nút OK và Cancel
        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))

        tk.Button(btn_frame, text="OK", width=8, command=on_ok).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(btn_frame, text="Hủy", width=8, command=dialog.destroy).pack(
            side=tk.LEFT, padx=5
        )

        # enter = OK
        dialog.bind("<Return>", lambda e: on_ok())

    def _show_help(self):
        """hiện hướng dẫn chơi."""
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

    def _show_dfs_info(self):
        """hiển thị thông tin về thuật toán DFS flood fill."""
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

    def _show_about(self):
        """thông tin về game."""
        messagebox.showinfo(
            "Thông tin",
            "💣 Minesweeper — AI Assignment\n\n"
            "Môn: Nhập môn Trí tuệ Nhân tạo (CO3061)\n\n"
            "🛠️ Công nghệ:\n"
            "• Python + Tkinter\n"
            "• DFS Iterative Flood Fill\n"
            "• Kiến trúc MVC\n\n"
            "📚 Thuật toán: DFS dùng Stack (LIFO)\n"
            "để tránh giới hạn đệ quy của Python."
        )

    def create_grid(self, rows, cols):
        """
        tạo grid nút bấm mới. xóa grid cũ nếu có.
        mỗi ô là 1 Button với bind click trái, phải, double click.

        dùng closure (lambda) để bind row, col vào callback.
        trick: phải dùng r=r, c=c trong lambda nếu không sẽ bị closure bug
        (tất cả button sẽ gửi cùng 1 giá trị row, col — bug kinh điển).
        """
        # xóa grid cũ
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
                    activebackground="#D0D0D0"
                )
                btn.grid(row=r, column=c, padx=0, pady=0)

                # bind events — dùng default argument trick để fix closure
                # nếu không có r=r, c=c thì tất cả button sẽ reference
                # cùng biến r, c cuối cùng → bug nổi tiếng với lambda
                btn.bind("<Button-1>", lambda e, r=r, c=c: self._handle_left_click(r, c))
                btn.bind("<Button-3>", lambda e, r=r, c=c: self._handle_right_click(r, c))
                btn.bind("<Double-Button-1>", lambda e, r=r, c=c: self._handle_double_click(r, c))

                row_buttons.append(btn)
            self.buttons.append(row_buttons)

        # update window size cho vừa grid
        self.root.update_idletasks()

    def _handle_left_click(self, row, col):
        """forward left click cho controller."""
        if self.on_left_click:
            self.on_left_click(row, col)

    def _handle_right_click(self, row, col):
        """forward right click cho controller."""
        if self.on_right_click:
            self.on_right_click(row, col)

    def _handle_double_click(self, row, col):
        """forward double click cho controller (chord reveal)."""
        if self.on_double_click:
            self.on_double_click(row, col)

    def update_cell(self, row, col, value):
        """
        cập nhật hiển thị 1 ô đã revealed.
        value = 0-8 (số mine xung quanh) hoặc -1 (mine) hoặc -2 (flag sai)

        styling:
            - ô revealed: nền sáng hơn, relief sunken (trông như đã ép xuống)
            - số: hiển thị con số với màu tương ứng
            - ô trống: không hiển thị gì
            - mine: hiển thị emoji 💣
            - flag sai: hiển thị ❌
        """
        btn = self.buttons[row][col]

        if value == -1:
            # mine — hiển thị bom
            btn.configure(
                text="💣",
                font=("Segoe UI Emoji", 10),
                relief=tk.SUNKEN,
                bg="#FF0000",
                disabledforeground="#000000",
                state=tk.DISABLED
            )
        elif value == -2:
            # flag sai — hiển thị X trên cờ
            btn.configure(
                text="❌",
                font=("Segoe UI Emoji", 10),
                relief=tk.SUNKEN,
                bg="#FFA500",
                disabledforeground="#000000",
                state=tk.DISABLED
            )
        elif value == 0:
            # ô trống — không hiển thị gì, chỉ đổi nền
            btn.configure(
                text="",
                relief=tk.SUNKEN,
                bg="#D9D9D9",
                state=tk.DISABLED
            )
        else:
            # ô có số 1-8 — hiển thị số với màu
            color = self.NUMBER_COLORS.get(value, "#000000")
            btn.configure(
                text=str(value),
                font=("Consolas", 10, "bold"),
                relief=tk.SUNKEN,
                bg="#D9D9D9",
                disabledforeground=color,
                state=tk.DISABLED
            )

    def update_flag(self, row, col, flagged):
        """
        toggle hiển thị cờ trên ô.
        flagged=True → hiện cờ 🚩
        flagged=False → xóa cờ (quay về ô hidden bình thường)
        """
        btn = self.buttons[row][col]
        if flagged:
            btn.configure(
                text="🚩",
                font=("Segoe UI Emoji", 10),
                fg="#FF0000"
            )
        else:
            btn.configure(
                text="",
                font=("Consolas", 10, "bold"),
                fg="#000000"
            )

    def update_mine_counter(self, count):
        """
        cập nhật mine counter ở góc trái.
        format 3 chữ số, có thể âm (nếu flag nhiều hơn mine).
        """
        if count < 0:
            text = "-" + str(abs(count)).zfill(2)
        else:
            text = str(count).zfill(3)
        self.mine_label.configure(text=text)

    def update_timer(self, seconds):
        """cập nhật timer ở góc phải. format 3 chữ số, max 999."""
        seconds = min(seconds, 999)
        self.timer_label.configure(text=str(seconds).zfill(3))

    def set_smiley(self, face):
        """
        đổi mặt nút reset theo trạng thái game.
        face: "happy" (đang chơi), "dead" (thua), "cool" (thắng),
              "surprised" (đang click — optional)
        """
        faces = {
            "happy": "🙂",
            "dead": "😵",
            "cool": "😎",
            "surprised": "😮"
        }
        self.reset_button.configure(text=faces.get(face, "🙂"))

    def highlight_mine(self, row, col):
        """
        highlight ô mine được click (ô gây thua).
        nền đỏ đậm hơn các mine khác.
        """
        btn = self.buttons[row][col]
        btn.configure(bg="#CC0000")

    def show_win(self):
        """hiện dialog thắng game."""
        self.set_smiley("cool")
        messagebox.showinfo(
            "🎉 Thắng rồi!",
            "Chúc mừng! Bạn đã dò hết mìn!\n\n"
            "🏆 Bạn thắng game Minesweeper!"
        )

    def show_game_over(self):
        """hiện dialog thua game."""
        self.set_smiley("dead")
        messagebox.showinfo(
            "💥 Thua rồi!",
            "Bạn đã dẫm trúng mìn!\n\n"
            "💀 Game Over — thử lại nhé!"
        )


# ===========================================================================
#  PHẦN 3: CONTROLLER — kết nối model và view, xử lý tất cả event
# ===========================================================================

class MinesweeperController:
    """
    controller: nhận event từ view, gọi model xử lý, update view.
    cũng quản lý timer, game flow (new game, reset, win/loss).

    kiểu MVC: view không biết model, model không biết view.
    controller là trung gian duy nhất.
    """

    def __init__(self, root):
        """
        khởi tạo controller, tạo view, set default game.

        params:
            root: tk.Tk() — cửa sổ chính
        """
        self.root = root
        self.view = MinesweeperView(root)

        # config ban đầu — easy difficulty (9x9, 10 mines)
        self.rows = 9
        self.cols = 9
        self.num_mines = 10

        # model sẽ tạo mới mỗi khi new game
        self.model = None

        # timer tracking
        self.timer_running = False
        self.start_time = 0
        self.timer_id = None

        # bind callbacks từ view
        self.view.on_left_click = self.on_left_click
        self.view.on_right_click = self.on_right_click
        self.view.on_double_click = self.on_double_click
        self.view.on_new_game = self.new_game
        self.view.on_reset = self.reset_game

        # bắt đầu game đầu tiên
        self.new_game(self.rows, self.cols, self.num_mines)

    def new_game(self, rows, cols, mines):
        """
        tạo game mới hoàn toàn.
        reset model, view, timer.
        gọi khi chọn difficulty từ menu hoặc custom.

        params:
            rows: số hàng mới
            cols: số cột mới
            mines: số mìn mới
        """
        # dừng timer cũ
        self._stop_timer()

        # lưu config mới (để reset dùng lại)
        self.rows = rows
        self.cols = cols
        self.num_mines = mines

        # tạo model mới
        self.model = MinesweeperModel(rows, cols, mines)

        # tạo grid mới trên view
        self.view.create_grid(rows, cols)
        self.view.update_mine_counter(mines)
        self.view.update_timer(0)
        self.view.set_smiley("happy")

        # căn giữa cửa sổ trên màn hình
        self._center_window()

    def reset_game(self):
        """reset game với cùng difficulty. gọi khi click nút smiley."""
        self.new_game(self.rows, self.cols, self.num_mines)

    def on_left_click(self, row, col):
        """
        xử lý left click: mở ô.
        gửi cho model reveal, nhận kết quả, update view.

        flow:
            1. model.reveal(row, col)
            2. nếu game over → hiện tất cả mine
            3. nếu win → hiện dialog thắng
            4. nếu bình thường → update các ô đã reveal
        """
        if not self.model or self.model.game_over or self.model.won:
            return

        # bắt đầu timer từ first click
        if self.model.first_click:
            self._start_timer()

        # gọi model reveal
        result = self.model.reveal(row, col)

        if not result:
            return

        # game over — dính mine
        if self.model.game_over:
            self._stop_timer()
            # highlight ô mine được click
            self.view.highlight_mine(row, col)
            # hiện tất cả mine và flag sai
            for r, c, val in result:
                self.view.update_cell(r, c, val)
            self.view.show_game_over()
            return

        # reveal thành công — update view
        for r, c, val in result:
            self.view.update_cell(r, c, val)

        # check win
        if self.model.won:
            self._stop_timer()
            # auto flag tất cả mine còn lại khi thắng
            self._auto_flag_mines()
            self.view.show_win()

    def on_right_click(self, row, col):
        """
        xử lý right click: toggle flag.
        cập nhật view và mine counter.
        """
        if not self.model or self.model.game_over or self.model.won:
            return

        result = self.model.toggle_flag(row, col)
        if result == "flagged":
            self.view.update_flag(row, col, True)
        elif result == "hidden":
            self.view.update_flag(row, col, False)

        # update mine counter
        self.view.update_mine_counter(self.model.get_mines_remaining())

    def on_double_click(self, row, col):
        """
        xử lý double click: chord reveal.
        chỉ hoạt động trên ô số đã revealed.
        phải đủ flag xung quanh mới reveal.
        """
        if not self.model or self.model.game_over or self.model.won:
            return

        result = self.model.chord_reveal(row, col)

        if not result:
            return

        # game over từ chord (flag sai vị trí)
        if self.model.game_over:
            self._stop_timer()
            # tìm ô mine bị click
            for r, c, val in result:
                if val == -1:
                    self.view.highlight_mine(r, c)
            # hiện tất cả mine
            all_mines = self.model._reveal_all_mines()
            for r, c, val in all_mines:
                self.view.update_cell(r, c, val)
            self.view.show_game_over()
            return

        # update view bình thường
        for r, c, val in result:
            self.view.update_cell(r, c, val)

        # check win
        if self.model.won:
            self._stop_timer()
            self._auto_flag_mines()
            self.view.show_win()

    def _auto_flag_mines(self):
        """
        khi thắng, tự động cắm cờ lên tất cả mine còn lại.
        để người chơi thấy đã tìm đúng hết vị trí mine.
        """
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if (self.model.board[r][c] == -1 and
                        self.model.state[r][c] != "flagged"):
                    self.view.update_flag(r, c, True)
        self.view.update_mine_counter(0)

    def _start_timer(self):
        """bắt đầu đếm timer. gọi khi first click."""
        self.start_time = time.time()
        self.timer_running = True
        self._update_timer()

    def _stop_timer(self):
        """dừng timer. gọi khi win hoặc game over."""
        self.timer_running = False
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None

    def _update_timer(self):
        """
        update timer mỗi giây.
        dùng root.after() thay vì threading — an toàn cho tkinter.
        tkinter không thread-safe, nên PHẢI dùng after().
        """
        if self.timer_running:
            elapsed = int(time.time() - self.start_time)
            self.view.update_timer(elapsed)
            # schedule update tiếp sau 1 giây
            # 500ms cho mượt hơn vì after() không chính xác tuyệt đối
            self.timer_id = self.root.after(500, self._update_timer)

    def _center_window(self):
        """
        căn giữa cửa sổ trên màn hình.
        phải update_idletasks trước để lấy đúng kích thước window.
        """
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        self.root.geometry(f"+{x}+{y}")


# ===========================================================================
#  PHẦN 4: MAIN — entry point, chạy game
# ===========================================================================

def main():
    """
    hàm main — tạo root window và khởi chạy game.
    controller sẽ tự tạo view và model bên trong.
    mainloop() để tkinter chạy event loop (chờ người dùng interact).
    """
    root = tk.Tk()

    # set icon title cho thanh taskbar
    root.title("Minesweeper — AI Assignment")

    # tạo controller
    controller = MinesweeperController(root)

    # bắt đầu event loop
    root.mainloop()


if __name__ == "__main__":
    main()
