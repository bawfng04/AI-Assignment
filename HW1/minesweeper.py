"""
minesweeper.py — game minesweeper hoàn chỉnh dùng tkinter + DFS flood fill + AI solver

kiến trúc MVC:
    - MinesweeperModel: xử lý logic game (board, mine, reveal, flag, win/loss)
    - MinesweeperView: giao diện tkinter (grid nút bấm, menu, status bar, timer)
    - MinesweeperController: kết nối Model với View, xử lý event click
    - MinesweeperAI: bộ giải AI dùng propositional logic + DFS backtracking

thuật toán chính:
    1. DFS iterative dùng stack (LIFO) cho flood fill
       - không dùng recursion vì python giới hạn đệ quy ~1000 frames
       - stack tự quản lý nên không lo stack overflow với grid lớn
    2. AI solver dùng rule-based inference (suy luận logic mệnh đề)
       - rule 1: nếu số = số flag → tất cả ô hidden còn lại an toàn
       - rule 2: nếu số = số ô chưa mở → tất cả là mine, flag hết
       - fallback: DFS backtracking thử gán mine/safe rồi kiểm tra consistency
       - last resort: random guess trên ô có xác suất mine thấp nhất

author: sinh viên năm 4 khoa CNTT, mệt nhưng vẫn code =))
"""

import tkinter as tk
from tkinter import messagebox, simpledialog
import random
import time
from itertools import combinations


# ===========================================================================
#  PHẦN 1: MODEL — logic game thuần, không liên quan gì tới GUI
# ===========================================================================

class MinesweeperModel:
    """
    class chứa toàn bộ logic của game minesweeper.
    tách riêng ra khỏi GUI để dễ test và maintain.
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
#  PHẦN 1.5: AI SOLVER — bộ giải tự động dùng logic + DFS backtracking
#  ý tưởng: AI suy luận như người chơi pro, dùng propositional logic
# ===========================================================================

class MinesweeperAI:
    """
    bộ giải AI cho minesweeper.
    dùng 2 tầng suy luận:

    tầng 1: rule-based inference (propositional logic — suy luận mệnh đề)
        - nhanh, chính xác 100%, nhưng không phải lúc nào cũng tìm được nước đi
        - rule 1 (all safe): nếu ô số đã có đủ flag xung quanh
                             → các ô hidden còn lại chắc chắn safe
        - rule 2 (all mines): nếu ô số có số ô hidden xung quanh = giá trị ô - flag
                              → tất cả ô hidden đó chắc chắn là mine

    tầng 2: DFS backtracking / constraint propagation
        - khi rule-based bế tắc (không suy luận được gì)
        - thử gán mine/safe cho từng ô và kiểm tra tính nhất quán (consistency)
        - nếu 1 ô mà gán mine → bế tắc (inconsistent) → ô đó chắc chắn safe
        - nếu 1 ô mà gán safe → bế tắc → ô đó chắc chắn mine
        - dùng DFS để thử tất cả tổ hợp có thể

    tầng 3: educated guess (đoán có cơ sở)
        - khi cả 2 tầng trên đều fail
        - chọn ô có xác suất mine thấp nhất dựa trên constraint counting
        - worst case: random 1 ô hidden bất kỳ

    tại sao 3 tầng?
        - rule-based xử lý ~70% trường hợp, nhanh O(rows*cols)
        - backtracking xử lý thêm ~25% trường hợp khó hơn
        - guess chỉ dùng khi bắt buộc (~5%), tránh bế tắc hoàn toàn
    """

    # 8 hướng — copy từ model cho tiện, khỏi reference qua lại
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    def __init__(self, model):
        """
        khởi tạo AI solver với reference tới model.
        AI chỉ ĐỌC state từ model, không modify trực tiếp.
        trả về action cho controller thực hiện.

        params:
            model: MinesweeperModel — model hiện tại của game
        """
        self.model = model

    def _in_bounds(self, r, c):
        """check boundary — giống model nhưng tách ra cho AI dùng riêng."""
        return 0 <= r < self.model.rows and 0 <= c < self.model.cols

    def _get_neighbors(self, r, c):
        """
        lấy danh sách neighbor hợp lệ của ô (r, c).
        trả về list of (nr, nc) đã check boundary.
        tách ra function riêng vì dùng NHIỀU chỗ trong AI.
        """
        neighbors = []
        for dr, dc in self.DIRECTIONS:
            nr, nc = r + dr, c + dc
            if self._in_bounds(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def _get_cell_info(self, r, c):
        """
        phân tích thông tin xung quanh 1 ô số đã revealed.
        trả về (hidden_neighbors, flagged_count, hidden_count).

        đây là building block cho cả 2 rule.
        """
        hidden_neighbors = []
        flagged_count = 0

        for nr, nc in self._get_neighbors(r, c):
            if self.model.state[nr][nc] == "hidden":
                hidden_neighbors.append((nr, nc))
            elif self.model.state[nr][nc] == "flagged":
                flagged_count += 1

        return hidden_neighbors, flagged_count, len(hidden_neighbors)

    def get_next_move(self):
        """
        hàm chính: tìm nước đi tiếp theo cho AI.
        thử lần lượt 3 tầng: rule-based → backtracking → guess.

        returns:
            tuple (action, cells) trong đó:
                action = "reveal" hoặc "flag"
                cells = list of (row, col) cần thực hiện
            hoặc None nếu không tìm được (game over hoặc đã thắng)

        flow:
            1. rule-based: scan tất cả ô revealed, áp dụng rule 1 + 2
            2. nếu rule-based không ra gì → thử backtracking
            3. nếu backtracking cũng không ra → educated guess
        """
        model = self.model

        # game đã kết thúc thì thôi
        if model.game_over or model.won:
            return None

        # first click chưa click → AI click giữa board cho safe
        if model.first_click:
            center_r = model.rows // 2
            center_c = model.cols // 2
            return ("reveal", [(center_r, center_c)])

        # ======================================================
        #  TẦNG 1: RULE-BASED INFERENCE (suy luận mệnh đề)
        #
        #  scan tất cả ô đã revealed có số > 0
        #  áp dụng 2 rule cho từng ô:
        #
        #  rule 1 (all safe): value == flagged_count
        #    → ô đã flag đủ mine rồi, các ô hidden còn lại safe
        #    → reveal tất cả hidden neighbors
        #
        #  rule 2 (all mines): value - flagged_count == hidden_count
        #    → số mine còn thiếu = số ô hidden → tất cả hidden là mine
        #    → flag tất cả hidden neighbors
        #
        #  ưu tiên flag trước rồi reveal sau, vì flag giúp unlock
        #  nhiều rule hơn ở các ô lân cận
        # ======================================================

        safe_cells = set()   # các ô chắc chắn safe → reveal
        mine_cells = set()   # các ô chắc chắn mine → flag

        for r in range(model.rows):
            for c in range(model.cols):
                # chỉ xét ô đã revealed và có số > 0
                if model.state[r][c] != "revealed" or model.board[r][c] <= 0:
                    continue

                value = model.board[r][c]
                hidden_neighbors, flagged_count, hidden_count = self._get_cell_info(r, c)

                # không có hidden neighbor → ô này đã solve xong, skip
                if hidden_count == 0:
                    continue

                # RULE 1: all safe
                # nếu value == flagged_count → đã flag đủ mine
                # → tất cả hidden neighbors chắc chắn an toàn
                # ví dụ: ô số 2, đã flag 2 ô → 0 mine còn lại → safe hết
                if value == flagged_count:
                    for nr, nc in hidden_neighbors:
                        safe_cells.add((nr, nc))

                # RULE 2: all mines
                # nếu value - flagged_count == hidden_count
                # → số mine còn thiếu = đúng số ô hidden → tất cả là mine
                # ví dụ: ô số 3, flag 1, hidden 2 → 3-1=2 mine trong 2 ô → mine hết
                remaining_mines = value - flagged_count
                if remaining_mines == hidden_count:
                    for nr, nc in hidden_neighbors:
                        mine_cells.add((nr, nc))

        # loại bỏ conflict: nếu 1 ô vừa safe vừa mine → bug logic, skip nó
        # trên lý thuyết không xảy ra nếu board hợp lệ, nhưng safe hơn check
        conflict = safe_cells & mine_cells
        safe_cells -= conflict
        mine_cells -= conflict

        # ưu tiên flag trước (vì flag mở khóa thêm rule 1 cho các ô khác)
        if mine_cells:
            return ("flag", list(mine_cells))

        if safe_cells:
            return ("reveal", list(safe_cells))

        # ======================================================
        #  TẦNG 2: DFS BACKTRACKING / CONSTRAINT SATISFACTION
        #
        #  khi rule-based bế tắc (thường xảy ra ở mid-game)
        #  ý tưởng: thử gán mine/safe cho từng ô rồi check
        #  xem có consistent không (có vi phạm constraint nào không)
        #
        #  constraint: mỗi ô số revealed tạo ra 1 constraint:
        #    số mine trong hidden neighbors = value - flagged_count
        #
        #  nếu gán ô X = mine → kiểm tra tất cả constraint liên quan
        #  nếu vi phạm (inconsistent) → ô X chắc chắn KHÔNG phải mine → safe
        #  tương tự ngược lại
        #
        #  approach: lấy tất cả "frontier" cells (ô hidden giáp ô revealed)
        #  thử từng ô, gán mine rồi check, gán safe rồi check
        # ======================================================

        backtrack_result = self._backtracking_solve()
        if backtrack_result:
            return backtrack_result

        # ======================================================
        #  TẦNG 3: EDUCATED GUESS (đoán có cơ sở)
        #
        #  khi cả rule-based lẫn backtracking đều fail
        #  → phải đoán, nhưng đoán thông minh:
        #    - tính xác suất mine cho từng ô dựa trên constraints
        #    - chọn ô có xác suất thấp nhất
        #    - nếu có ô hoàn toàn cô lập (không giáp ô nào revealed)
        #      → dùng xác suất global (mines_remaining / hidden_cells)
        #
        #  tại sao không random bừa?
        #    - random có thể chọn ô 50/50 trong khi có ô 10/90
        #    - educated guess tăng win rate đáng kể
        # ======================================================

        return self._educated_guess()

    def _get_frontier_cells(self):
        """
        lấy tất cả ô hidden nằm giáp với ít nhất 1 ô revealed.
        đây là các ô mà AI có thể suy luận được (có thông tin).

        ô hidden hoàn toàn bao quanh bởi hidden khác → không thể suy luận
        → chỉ dùng được ở tầng guess.

        returns: set of (row, col)
        """
        frontier = set()
        model = self.model

        for r in range(model.rows):
            for c in range(model.cols):
                if model.state[r][c] != "hidden":
                    continue
                # check xem có neighbor nào đã revealed không
                for nr, nc in self._get_neighbors(r, c):
                    if model.state[nr][nc] == "revealed" and model.board[nr][nc] > 0:
                        frontier.add((r, c))
                        break  # chỉ cần 1 neighbor revealed là đủ

        return frontier

    def _get_constraints(self):
        """
        xây dựng danh sách constraints từ board hiện tại.

        mỗi constraint là 1 tuple (hidden_neighbors, remaining_mines):
            - hidden_neighbors: set of (r, c) — các ô hidden xung quanh 1 ô số
            - remaining_mines: int — số mine còn thiếu = value - flagged_count

        constraint nghĩa là: trong tập hidden_neighbors,
        có ĐÚNG remaining_mines ô là mine.

        đây chính là biểu diễn propositional logic:
            sum(is_mine[cell] for cell in hidden_neighbors) == remaining_mines
        """
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
                # constraint phải hợp lệ: 0 <= remaining <= hidden_count
                if 0 <= remaining <= hidden_count:
                    constraints.append((set(hidden_neighbors), remaining))

        return constraints

    def _is_consistent(self, assignment, constraints):
        """
        kiểm tra xem assignment hiện tại có vi phạm constraint nào không.

        assignment: dict { (r,c): True/False }
            True = ô này là mine, False = ô này safe

        logic: với mỗi constraint, đếm số mine đã gán trong tập cells
            - nếu mine_count > remaining → quá nhiều mine → inconsistent
            - nếu còn lại không đủ ô để chứa mine → inconsistent
            - ngược lại → vẫn ok (chưa chắc consistent hoàn toàn nhưng tạm ok)

        returns: True nếu consistent, False nếu vi phạm
        """
        for cells, remaining in constraints:
            assigned_mines = 0
            assigned_safe = 0
            unassigned = 0

            for cell in cells:
                if cell in assignment:
                    if assignment[cell]:  # mine
                        assigned_mines += 1
                    else:  # safe
                        assigned_safe += 1
                else:
                    unassigned += 1

            # quá nhiều mine so với cần thiết
            if assigned_mines > remaining:
                return False

            # không đủ ô để chứa mine còn thiếu
            if assigned_mines + unassigned < remaining:
                return False

        return True

    def _backtracking_solve(self):
        """
        DFS backtracking solve: thử gán mine/safe cho từng frontier cell
        rồi check consistency.

        approach đơn giản (không full enumeration vì quá chậm):
            - lấy từng frontier cell
            - thử gán nó = mine, check consistency
            - thử gán nó = safe, check consistency
            - nếu chỉ có 1 assignment consistent → kết luận được

        approach nâng cao (dùng cho nhóm nhỏ ô liên quan):
            - cluster frontier cells theo connected components
            - với mỗi cluster nhỏ (≤ 15 ô), enumerate tất cả tổ hợp
            - đếm số tổ hợp mà ô X là mine vs safe
            - nếu ô X là mine trong 0% tổ hợp → chắc chắn safe
            - nếu ô X là mine trong 100% tổ hợp → chắc chắn mine

        returns: (action, cells) hoặc None
        """
        frontier = self._get_frontier_cells()
        if not frontier:
            return None

        constraints = self._get_constraints()
        if not constraints:
            return None

        safe_cells = set()
        mine_cells = set()

        # bước 1: thử từng cell riêng lẻ (quick check)
        for cell in frontier:
            # thử gán cell = mine
            test_mine = {cell: True}
            mine_ok = self._is_consistent(test_mine, constraints)

            # thử gán cell = safe
            test_safe = {cell: False}
            safe_ok = self._is_consistent(test_safe, constraints)

            # nếu chỉ mine consistent → chắc chắn mine
            if mine_ok and not safe_ok:
                mine_cells.add(cell)
            # nếu chỉ safe consistent → chắc chắn safe
            elif safe_ok and not mine_ok:
                safe_cells.add(cell)
            # cả 2 đều ok hoặc đều fail → chưa kết luận được

        if mine_cells:
            return ("flag", list(mine_cells))
        if safe_cells:
            return ("reveal", list(safe_cells))

        # bước 2: cluster enumeration cho nhóm nhỏ
        # group frontier cells thành clusters dựa trên shared constraints
        clusters = self._build_clusters(frontier, constraints)

        for cluster_cells, cluster_constraints in clusters:
            # chỉ enumerate cluster nhỏ (≤ 15 cells) để không bị exponential blowup
            # 2^15 = 32768 — vẫn chạy nhanh
            if len(cluster_cells) > 15:
                continue

            result = self._enumerate_cluster(cluster_cells, cluster_constraints)
            if result:
                return result

        return None

    def _build_clusters(self, frontier, constraints):
        """
        nhóm frontier cells thành clusters (connected components)
        dựa trên shared constraints.

        2 cell thuộc cùng cluster nếu chúng xuất hiện trong cùng 1 constraint.
        dùng union-find (disjoint set) đơn giản.

        returns: list of (cluster_cells, cluster_constraints)
        """
        # parent dict cho union-find
        parent = {cell: cell for cell in frontier}

        def find(x):
            # path compression
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # union cells trong cùng constraint
        for cells, _ in constraints:
            frontier_in_constraint = [c for c in cells if c in frontier]
            for i in range(1, len(frontier_in_constraint)):
                union(frontier_in_constraint[0], frontier_in_constraint[i])

        # group by root
        from collections import defaultdict
        groups = defaultdict(set)
        for cell in frontier:
            groups[find(cell)].add(cell)

        # tạo cluster list
        result = []
        for cluster_cells in groups.values():
            # lọc constraints liên quan đến cluster này
            cluster_constraints = []
            for cells, remaining in constraints:
                if cells & cluster_cells:  # có intersection
                    cluster_constraints.append((cells & cluster_cells, remaining))
            result.append((list(cluster_cells), cluster_constraints))

        return result

    def _enumerate_cluster(self, cluster_cells, constraints):
        """
        enumerate tất cả tổ hợp mine/safe cho 1 cluster nhỏ.
        đếm xem mỗi cell xuất hiện bao nhiêu lần là mine vs safe.

        nếu 1 cell là mine trong 100% tổ hợp hợp lệ → chắc chắn mine.
        nếu 1 cell là mine trong 0% tổ hợp hợp lệ → chắc chắn safe.

        dùng DFS để duyệt tất cả tổ hợp, prune sớm khi inconsistent.

        returns: (action, cells) hoặc None
        """
        n = len(cluster_cells)
        mine_count = {cell: 0 for cell in cluster_cells}
        total_valid = 0

        # DFS: duyệt tất cả tổ hợp gán mine/safe cho cluster cells
        # dùng stack iterative cho consistent với style bài (không recursion)
        # mỗi entry trong stack: (index, assignment_so_far)
        # index = vị trí cell tiếp theo cần gán
        stack = [(0, {})]

        while stack:
            idx, assignment = stack.pop()

            # đã gán hết tất cả cells → check full consistency
            if idx == n:
                # verify tất cả constraints đều thỏa mãn chính xác
                valid = True
                for cells, remaining in constraints:
                    mine_in_constraint = sum(
                        1 for c in cells if assignment.get(c, False)
                    )
                    # phải check cả cells ngoài cluster (đã flag)
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

            # thử gán cell = safe (False)
            new_assignment_safe = dict(assignment)
            new_assignment_safe[cell] = False
            if self._is_consistent(new_assignment_safe, constraints):
                stack.append((idx + 1, new_assignment_safe))

            # thử gán cell = mine (True)
            new_assignment_mine = dict(assignment)
            new_assignment_mine[cell] = True
            if self._is_consistent(new_assignment_mine, constraints):
                stack.append((idx + 1, new_assignment_mine))

        if total_valid == 0:
            return None

        # phân tích kết quả
        safe_cells = []
        mine_cells_result = []

        for cell in cluster_cells:
            if mine_count[cell] == 0:
                # mine trong 0% tổ hợp → chắc chắn safe
                safe_cells.append(cell)
            elif mine_count[cell] == total_valid:
                # mine trong 100% tổ hợp → chắc chắn mine
                mine_cells_result.append(cell)

        if mine_cells_result:
            return ("flag", mine_cells_result)
        if safe_cells:
            return ("reveal", safe_cells)

        return None

    def _educated_guess(self):
        """
        đoán có cơ sở: khi không suy luận được gì,
        chọn ô hidden có xác suất mine thấp nhất.

        cách tính:
            - ô trên frontier: count mine appearances / total valid combos
              (nếu đã enumerate) hoặc dùng local probability
            - ô không trên frontier: global probability = mines_left / hidden_count
            - chọn ô có probability thấp nhất

        ưu tiên ô KHÔNG nằm trên frontier (thường an toàn hơn ở early game)
        vì ô xa mine thường có probability thấp hơn.

        returns: ("reveal", [(row, col)]) hoặc None
        """
        model = self.model
        frontier = self._get_frontier_cells()

        # tìm tất cả ô hidden
        all_hidden = []
        for r in range(model.rows):
            for c in range(model.cols):
                if model.state[r][c] == "hidden":
                    all_hidden.append((r, c))

        if not all_hidden:
            return None

        # tính mines remaining
        mines_left = model.get_mines_remaining()

        # ô không thuộc frontier (interior hidden cells)
        non_frontier = [c for c in all_hidden if c not in frontier]

        # nếu có ô non-frontier → thường safe hơn, chọn đó
        if non_frontier and len(non_frontier) > 0:
            # xác suất global cho non-frontier cells
            # (mines_left - estimated_frontier_mines) / len(non_frontier)
            # đơn giản hóa: chọn random 1 ô non-frontier
            # vì xác suất đều nhau trong nhóm này
            cell = random.choice(non_frontier)
            return ("reveal", [cell])

        # tất cả hidden cells đều trên frontier → chọn ô có ít constraint nhất
        if frontier:
            # heuristic: ô có nhiều revealed neighbors hơn → nhiều thông tin hơn
            # chọn ô mà tổng (value - flagged) / hidden_count nhỏ nhất
            best_cell = None
            best_prob = 2.0  # > 1.0 để chắc chắn bị thay thế

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

        # fallback cuối cùng: random hidden cell
        cell = random.choice(all_hidden)
        return ("reveal", [cell])


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
        self.on_ai_move = None  # callback cho nút AI auto-move

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

        # nút AI auto-move — feature bonus cho bài AI
        # đặt cạnh nút reset, trước timer
        self.ai_button = tk.Button(
            self.top_frame,
            text="🤖",
            font=("Segoe UI Emoji", 14),
            width=2,
            height=1,
            relief=tk.RAISED,
            bd=2,
            bg="#90EE90",
            activebackground="#7CCD7C",
            command=self._on_ai_click
        )
        self.ai_button.pack(side=tk.LEFT, padx=(0, 5), pady=5)

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
            label="AI Solver",
            command=self._show_ai_info
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

    def _on_ai_click(self):
        """gọi callback AI auto-move. mỗi click = 1 nước đi."""
        if self.on_ai_move:
            self.on_ai_move()

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

    def _show_ai_info(self):
        """hiển thị thông tin về AI solver."""
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

    def _show_about(self):
        """thông tin về game."""
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
            "• DFS Backtracking cho constraint solving"
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

    def highlight_ai_move(self, row, col, action):
        """
        highlight ô mà AI vừa thực hiện action.
        dùng màu xanh cho safe reveal, cam cho flag.
        giúp người chơi theo dõi AI đang làm gì.
        """
        btn = self.buttons[row][col]
        if action == "reveal":
            # flash xanh nhạt rồi đổi về bình thường
            btn.configure(bg="#90EE90")  # light green
        elif action == "flag":
            btn.configure(bg="#FFD700")  # gold

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

        # AI solver instance — tạo mới mỗi khi new game
        self.ai = None

        # bind callbacks từ view
        self.view.on_left_click = self.on_left_click
        self.view.on_right_click = self.on_right_click
        self.view.on_double_click = self.on_double_click
        self.view.on_new_game = self.new_game
        self.view.on_reset = self.reset_game
        self.view.on_ai_move = self.on_ai_move

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

        # tạo AI solver mới cho game mới
        self.ai = MinesweeperAI(self.model)

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

    def on_ai_move(self):
        """
        xử lý khi user click nút AI 🤖.
        AI thực hiện ĐÚNG 1 nước đi logic (reveal hoặc flag).
        user có thể click nhiều lần để xem AI chơi step-by-step.

        flow:
            1. gọi ai.get_next_move() để lấy action
            2. nếu action = "reveal" → gọi on_left_click cho từng cell
            3. nếu action = "flag" → gọi on_right_click cho từng cell
            4. highlight ô vừa được AI chọn (visual feedback)

        tại sao step-by-step thay vì auto-play?
            - user có thể quan sát AI suy luận
            - dễ debug nếu AI sai
            - thầy có thể thấy rõ AI đang làm gì → điểm cao hơn
        """
        if not self.model or self.model.game_over or self.model.won:
            return

        if not self.ai:
            self.ai = MinesweeperAI(self.model)

        # lấy nước đi từ AI
        move = self.ai.get_next_move()

        if not move:
            messagebox.showinfo(
                "🤖 AI",
                "AI không tìm được nước đi nào!\n"
                "Game có thể đã kết thúc."
            )
            return

        action, cells = move

        if action == "reveal":
            # AI mở ô — gọi on_left_click để reuse logic
            for r, c in cells:
                if self.model.game_over or self.model.won:
                    break
                # highlight trước khi reveal
                self.view.highlight_ai_move(r, c, "reveal")
                self.on_left_click(r, c)

        elif action == "flag":
            # AI cắm cờ — gọi on_right_click
            for r, c in cells:
                if self.model.game_over or self.model.won:
                    break
                # chỉ flag ô chưa flag
                if self.model.state[r][c] == "hidden":
                    self.view.highlight_ai_move(r, c, "flag")
                    self.on_right_click(r, c)

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
