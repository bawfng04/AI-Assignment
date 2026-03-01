# 💣 Minesweeper — Bài tập 1 (Homework 1)

> **Môn học:** Nhập môn Trí tuệ Nhân tạo (CO3061)
>
> **Ngôn ngữ:** Python 3 + Tkinter (thư viện GUI có sẵn, không cần cài thêm)
>
> **Thuật toán chính:** DFS Iterative (Stack) cho Flood Fill + AI Solver (Propositional Logic + DFS Backtracking)

---

## 📋 Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc hệ thống (MVC)](#2-kiến-trúc-hệ-thống-mvc)
3. [Thuật toán DFS Flood Fill (Trọng tâm)](#3-thuật-toán-dfs-flood-fill-trọng-tâm)
4. [AI Solver — Bộ giải tự động](#4-ai-solver--bộ-giải-tự-động)
5. [Các tính năng chính & Edge Cases](#5-các-tính-năng-chính--edge-cases)
6. [Hướng dẫn chạy chương trình](#6-hướng-dẫn-chạy-chương-trình)

---

## 1. Tổng quan dự án

Chương trình cài đặt trò chơi **Minesweeper** (Dò mìn) hoàn chỉnh với giao diện đồ họa (GUI) sử dụng thư viện **Tkinter** có sẵn trong Python.

### Yêu cầu bài tập

| Yêu cầu | Giải pháp |
|----------|-----------|
| Cài đặt trò chơi Minesweeper với GUI | Python + Tkinter (không phụ thuộc thư viện ngoài) |
| Hỗ trợ nhiều kích thước bàn cờ | 5×5, 9×9, 16×16, 16×30 và tùy chỉnh |
| Sử dụng thuật toán DFS | Iterative DFS với Stack (LIFO) cho Flood Fill |
| AI Solver tự động | Propositional Logic + DFS Backtracking (95%+ win rate) |
| Thiết kế có cấu trúc rõ ràng | Kiến trúc MVC (Model-View-Controller) |

### Luật chơi cơ bản

- **Click trái** vào ô để mở. Nếu ô hiển thị một con số, đó là số mìn nằm trong 8 ô xung quanh (ngang, dọc, chéo). Nếu ô trống (giá trị 0) thì sẽ tự động mở rộng các ô trống liền kề.
- **Click phải** vào ô để cắm/bỏ cờ 🚩, đánh dấu vị trí nghi ngờ có mìn.
- **Chiến thắng** khi tất cả ô không chứa mìn đã được mở.
- **Thua** khi click trúng ô chứa mìn.

---

## 2. Kiến trúc hệ thống (MVC)

Chương trình được thiết kế theo mô hình **Model-View-Controller (MVC)** — tách biệt logic xử lý, giao diện hiển thị, và điều khiển sự kiện. Việc chia tách này giúp code dễ bảo trì, dễ test, và dễ mở rộng.

```
┌─────────────────────────────────────────────────────────┐
│                    MinesweeperController                │
│   (Điều phối: nhận event từ View, gọi Model xử lý,     │
│    gọi AI solver, rồi cập nhật lại View)               │
├────────────────────────┬────────────────────────────────┤
│    MinesweeperModel    │       MinesweeperView          │
│                        │                                │
│  • board[][] (mìn/số)  │  • Grid nút bấm (Tkinter)     │
│  • state[][] (trạng    │  • Menu bar (chọn difficulty)  │
│    thái ô)             │  • Mine counter (góc trái)     │
│  • generate_mines()    │  • Timer (góc phải)            │
│  • reveal() + DFS      │  • Nút 🙂 Reset + 💡 Hint + 🤖 Solve │
│  • toggle_flag()       │  • Dialog tùy chỉnh           │
│  • check_win()         │  • Hiển thị win/lose           │
├────────────────────────┴────────────────────────────────┤
│                     MinesweeperAI                        │
│  • Rule-based logic (propositional inference)            │
│  • DFS backtracking + cluster enumeration                │
│  • Educated guess (probability-based)                    │
└─────────────────────────────────────────────────────────┘
```

### 2.1. Model — `MinesweeperModel`

Chịu trách nhiệm toàn bộ **logic trò chơi**, không biết gì về giao diện:

| Thuộc tính | Mô tả |
|------------|-------|
| `board[r][c]` | Ma trận giá trị: `-1` = mìn, `0–8` = số mìn xung quanh |
| `state[r][c]` | Trạng thái hiển thị: `"hidden"`, `"revealed"`, `"flagged"` |
| `first_click` | Cờ đánh dấu lượt click đầu tiên (đảm bảo an toàn) |
| `revealed_count` | Bộ đếm ô đã mở (dùng cho kiểm tra thắng nhanh O(1)) |

**Các phương thức chính:**

- `generate_mines(safe_row, safe_col)` — Sinh mìn ngẫu nhiên, đảm bảo vùng 3×3 quanh ô click đầu tiên không có mìn.
- `reveal(row, col)` — Mở ô. Nếu ô trống → kích hoạt **DFS Flood Fill**. Nếu ô có mìn → game over.
- `toggle_flag(row, col)` — Cắm/bỏ cờ trên ô.
- `check_win()` — Kiểm tra điều kiện thắng: `revealed_count == tổng ô - số mìn`.
- `chord_reveal(row, col)` — Mở nhanh các ô xung quanh ô số khi đã cắm đủ cờ.

### 2.2. View — `MinesweeperView`

Chịu trách nhiệm **hiển thị giao diện** bằng Tkinter:

- **Grid nút bấm:** Mỗi ô trên bàn cờ là một `tk.Button`. Khi click, sự kiện được gửi cho Controller xử lý.
- **Menu bar:** Cung cấp các preset độ khó (Dễ, Trung bình, Khó) và tùy chỉnh kích thước.
- **Status bar:** Hiển thị bộ đếm mìn còn lại (trái), các nút 🙂 Reset / 💡 Hint / 🤖 Solve (giữa), timer (phải).
- **Màu số:** Mỗi con số 1–8 có màu riêng theo chuẩn Minesweeper gốc (1=xanh dương, 2=xanh lá, 3=đỏ, …).

### 2.3. Controller — `MinesweeperController`

Đóng vai trò **trung gian** giữa Model và View:

- Nhận sự kiện click từ View (click trái, click phải, double-click).
- Gọi phương thức tương ứng trên Model để xử lý logic.
- Lấy kết quả từ Model và cập nhật lại View.
- Quản lý timer (bắt đầu từ click đầu tiên, dừng khi thắng/thua).
- Xử lý tạo game mới / reset game.

**Luồng xử lý khi người chơi click trái vào 1 ô:**

```
Click trái (View) → on_left_click() (Controller)
    → model.reveal(row, col) (Model)
        → DFS Flood Fill nếu ô trống
        → Trả về danh sách ô đã mở
    → view.update_cell() cho từng ô (View)
    → Kiểm tra win/loss
```

---

## 3. Thuật toán DFS Flood Fill (Trọng tâm)

### 3.1. Vấn đề cần giải quyết

Khi người chơi click vào một ô **trống** (giá trị 0, không có mìn xung quanh), game cần **tự động mở tất cả ô trống liền kề**, dừng lại khi gặp ô có số (biên). Đây chính là bài toán **Flood Fill** — tương tự thuật toán tô màu trong đồ họa.

### 3.2. Tại sao chọn DFS Iterative (Stack) thay vì Recursion?

| Tiêu chí | Recursion | Iterative Stack |
|----------|-----------|-----------------|
| Giới hạn | Python mặc định ~1000 frame | Không giới hạn (chỉ phụ thuộc RAM) |
| Grid 16×30 (480 ô) | Có nguy cơ `RecursionError` | Hoàn toàn an toàn |
| Kiểm soát | Phụ thuộc call stack hệ thống | Tự quản lý stack |
| Hiệu suất | Overhead từ function call | Nhanh hơn (chỉ append/pop list) |
| Thứ tự duyệt | DFS (theo call stack) | DFS (LIFO — Last In First Out) |

**Kết luận:** Dùng **stack tự quản lý** (Python list với `append()` = push, `pop()` = pop) để đảm bảo đúng thứ tự DFS mà không bị giới hạn đệ quy.

### 3.3. Mô hình đồ thị

Bàn cờ Minesweeper được mô hình hóa như một **đồ thị vô hướng**:

- **Node (đỉnh):** Mỗi ô trên bàn cờ là một node, tọa độ `(row, col)`.
- **Edge (cạnh):** Mỗi ô kết nối với tối đa 8 ô xung quanh (ngang, dọc, chéo). Ô ở góc có 3 neighbor, ô ở cạnh có 5, ô ở giữa có 8.
- **Duyệt DFS:** Bắt đầu từ ô được click, đi sâu vào các ô trống liền kề trước khi quay lại (backtrack).

```
Ví dụ grid 5×5 (X = mìn, số = mine count, · = trống):

    0   1   2   3   4
  ┌───┬───┬───┬───┬───┐
0 │ · │ · │ · │ · │ · │
  ├───┼───┼───┼───┼───┤
1 │ · │ · │ 1 │ 1 │ 1 │
  ├───┼───┼───┼───┼───┤
2 │ · │ · │ 1 │ X │ 1 │
  ├───┼───┼───┼───┼───┤
3 │ · │ · │ 1 │ 1 │ 1 │
  ├───┼───┼───┼───┼───┤
4 │ · │ · │ · │ · │ · │
  └───┴───┴───┴───┴───┘

Click vào (0,0):
→ DFS sẽ mở tất cả ô "·" (trống)
→ Dừng lại ở các ô số "1" (biên, vẫn được mở nhưng không expand)
→ Tổng cộng mở 24/25 ô (trừ ô mìn X)
```

### 3.4. Chi tiết từng bước của thuật toán

```python
def reveal(self, row, col):
    # ========================================
    # BƯỚC 1: Khởi tạo cấu trúc dữ liệu
    # ========================================
    stack = [(row, col)]   # Stack (LIFO) — dùng list của Python
                           # append() = push, pop() = pop
    visited = set()        # Set lưu các ô đã xử lý
                           # → tránh xử lý lại cùng 1 ô (tránh vòng lặp vô hạn)
    revealed_cells = []    # Danh sách kết quả — các ô đã mở

    # ========================================
    # BƯỚC 2: Vòng lặp DFS chính
    # ========================================
    while stack:                    # Lặp cho đến khi stack rỗng
        r, c = stack.pop()         # Pop phần tử cuối → LIFO → DFS
                                   # (khác với BFS dùng queue pop(0) = FIFO)

        # ========================================
        # BƯỚC 3: Kiểm tra đã visit chưa
        # ========================================
        if (r, c) in visited:      # Đã xử lý ô này rồi → bỏ qua
            continue               # → O(1) nhờ dùng set
        visited.add((r, c))        # Đánh dấu đã visit

        # ========================================
        # BƯỚC 4: Kiểm tra trạng thái ô
        # ========================================
        if state[r][c] != "hidden":  # Ô đã revealed hoặc flagged
            continue                  # → bỏ qua, không xử lý lại

        # ========================================
        # BƯỚC 5: Mở ô (reveal)
        # ========================================
        state[r][c] = "revealed"           # Đổi trạng thái
        revealed_count += 1                 # Tăng bộ đếm
        revealed_cells.append((r, c, board[r][c]))  # Thêm vào kết quả

        # ========================================
        # BƯỚC 6: Quyết định có expand không
        # ========================================
        if board[r][c] == 0:         # Ô TRỐNG (không có mìn xung quanh)
            # → Push 8 neighbor vào stack để tiếp tục DFS
            for dr, dc in DIRECTIONS:     # 8 hướng: ↑↓←→↗↘↙↖
                nr, nc = r + dr, c + dc
                if (in_bounds(nr, nc)           # Trong phạm vi grid
                    and (nr, nc) not in visited  # Chưa visit
                    and board[nr][nc] != -1):    # Không phải mìn
                    stack.append((nr, nc))       # Push vào stack

        # Nếu board[r][c] > 0 (ô có SỐ):
        # → Ô này là BIÊN, vẫn được mở nhưng KHÔNG push neighbor
        # → DFS dừng mở rộng tại đây

    return revealed_cells
```

### 3.5. Phân tích độ phức tạp

| Thành phần | Độ phức tạp | Giải thích |
|------------|-------------|------------|
| **Thời gian** | O(V + E) | V = số ô (rows × cols), E = số cạnh (~4V cho grid) |
| **Không gian** | O(V) | Stack + visited set, tối đa lưu tất cả ô |
| **Worst case** | O(rows × cols) | Khi toàn bộ board là ô trống (không có mìn nào cạnh) |

### 3.6. Minh họa từng bước DFS

```
Ví dụ: Grid 3×3, mìn ở (2,2), click vào (0,0)

Board:          State ban đầu:
 0  0  0         H  H  H        (H = hidden)
 0  1  1         H  H  H
 0  1  X         H  H  H

═══════════════════════════════════════════════
Bước 1: stack = [(0,0)]
        Pop (0,0) → value=0 → reveal → push neighbors
        stack = [(1,0), (1,1), (0,1)]

Bước 2: Pop (0,1) → value=0 → reveal → push neighbors
        stack = [(1,0), (1,1), (1,1), (1,2), (0,2)]
        (1,1 xuất hiện 2 lần nhưng visited set sẽ xử lý)

Bước 3: Pop (0,2) → value=0 → reveal → push neighbors
        stack = [..., (1,2), (1,1)]

Bước 4: Pop (1,1) → value=1 → reveal → KHÔNG push (ô số = biên)

Bước 5: Pop (1,2) → value=1 → reveal → KHÔNG push

Bước 6: Pop (1,1) → đã visited → skip

Bước 7: Pop (1,0) → value=0 → reveal → push neighbors
        stack = [(2,0), (2,1)]

Bước 8: Pop (2,1) → value=1 → reveal → KHÔNG push

Bước 9: Pop (2,0) → value=0 → reveal → push neighbors
        → Không còn neighbor hợp lệ

Stack rỗng → KẾT THÚC

Kết quả: Mở 8/9 ô (trừ ô mìn X)
═══════════════════════════════════════════════
```

---

## 4. AI Solver — Bộ giải tự động

Ngoài thuật toán DFS Flood Fill, chương trình còn cài đặt một **AI Solver** có khả năng tự động chơi Minesweeper với tỉ lệ thắng cao. AI sử dụng **3 tầng suy luận** theo thứ tự ưu tiên:

### 4.1. Tầng 1: Rule-Based Inference (Suy luận mệnh đề)

Dùng **propositional logic** để suy luận chính xác 100% từ các ô số đã mở:

**Rule 1 — All Safe (tất cả an toàn):**

> Nếu một ô số đã có đủ cờ xung quanh (số cờ = giá trị ô), thì tất cả các ô hidden còn lại xung quanh chắc chắn **an toàn** → reveal.

```
Ví dụ: Ô số 2, đã cắm 2 cờ 🚩
→ 0 mìn còn lại → các ô hidden xung quanh đều safe → reveal hết
```

**Rule 2 — All Mines (tất cả là mìn):**

> Nếu số mìn còn thiếu (giá trị ô - số cờ) bằng số ô hidden xung quanh, thì tất cả các ô hidden đó chắc chắn là **mìn** → flag.

```
Ví dụ: Ô số 3, cắm 1 cờ, còn 2 ô hidden
→ 3 - 1 = 2 mìn trong 2 ô → cả 2 là mìn → flag hết
```

### 4.2. Tầng 2: DFS Backtracking (Constraint Satisfaction)

Khi Rule-Based không suy luận được gì (bế tắc), AI chuyển sang **DFS Backtracking**:

1. **Xây dựng constraints:** Mỗi ô số đã revealed tạo ra 1 constraint: *"Trong tập các ô hidden xung quanh, có đúng X ô là mìn"* (X = giá trị ô - số cờ).

2. **Quick check từng ô:** Thử gán từng frontier cell (các ô hidden giáp với ô revealed) là mine hoặc safe, kiểm tra tính nhất quán (consistency). Nếu chỉ có 1 phương án hợp lệ → kết luận được.

3. **Cluster enumeration:** Nhóm các frontier cells thành clusters (connected components) bằng **Union-Find**. Với mỗi cluster nhỏ (≤ 15 ô), **enumerate tất cả tổ hợp** mine/safe bằng DFS iterative:
   - Nếu 1 ô là mine trong **100%** tổ hợp hợp lệ → chắc chắn mine
   - Nếu 1 ô là mine trong **0%** tổ hợp hợp lệ → chắc chắn safe

```python
# pseudocode cho backtracking enumeration
stack = [(0, {})]  # (cell_index, assignment)
while stack:
    idx, assignment = stack.pop()  # DFS, LIFO
    if idx == len(cluster):
        if all constraints satisfied:
            count valid combination
        continue
    # thử gán cell[idx] = safe, nếu consistent → push
    # thử gán cell[idx] = mine, nếu consistent → push
```

### 4.3. Tầng 3: Educated Guess (Đoán có cơ sở)

Khi cả 2 tầng trên đều không tìm được nước đi chắc chắn:

- **Ưu tiên ô non-frontier** (ô hidden không giáp ô nào đã mở) — thường có xác suất mìn thấp hơn.
- Nếu tất cả ô đều là frontier → chọn ô có **xác suất mìn thấp nhất** dựa trên tỉ lệ `(value - flags) / hidden_count` của các ô số xung quanh.

### 4.4. Kết quả kiểm thử AI

| Kích thước | Số mìn | Số game | Kết quả |
|-----------|--------|---------|----------|
| 5×5 | 3 | 50 | **96% win rate** (48/50) |
| 9×9 | 10 | 20 | **95% win rate** (19/20) |

### 4.5. Luồng xử lý AI

```
User click 💡 Hint → Controller.on_ai_hint()
    → ai.get_next_move()
        → Tầng 1: Rule-based
        → Tầng 2: Backtracking + cluster enumeration
        → Tầng 3: Educated guess
    → Controller thực hiện 1 action (reveal/flag), cập nhật View

User click 🤖 Solve → Controller.on_ai_solve()
    → chạy nhiều bước bằng after() để UI không bị đơ
    → lặp: ai.get_next_move() → thực hiện action → cập nhật View
    → dừng khi win/lose/không còn nước đi
```

---

## 5. Các tính năng chính & Edge Cases

### 5.1. First-Click Safety (An toàn lượt đầu)

**Vấn đề:** Nếu sinh mìn trước khi người chơi click, lượt click đầu tiên có thể trúng mìn ngay → trải nghiệm xấu.

**Giải pháp:** Trì hoãn việc sinh mìn cho đến khi người chơi click lần đầu:

1. Khi khởi tạo game, board hoàn toàn trống (chưa có mìn).
2. Khi người chơi click lần đầu tại ô `(r, c)`:
   - Tạo **vùng an toàn 3×3** gồm ô `(r, c)` và 8 ô xung quanh.
   - Sinh mìn ngẫu nhiên trên các ô **nằm ngoài** vùng an toàn.
   - Tính số mìn xung quanh (`adjacent count`) cho tất cả ô.
3. Sau đó mới thực hiện `reveal()` bình thường.

**Lợi ích:** Lượt click đầu luôn an toàn VÀ luôn mở được một vùng rộng (vì vùng 3×3 không có mìn → flood fill được kích hoạt).

### 5.2. Boundary Checking (Kiểm tra biên)

Mỗi ô có tối đa 8 neighbor, nhưng ô ở **góc** chỉ có 3 và ô ở **cạnh** chỉ có 5. Hàm `_in_bounds(r, c)` kiểm tra tọa độ có nằm trong phạm vi grid hay không trước mỗi lần truy cập:

```python
def _in_bounds(self, r, c):
    return 0 <= r < self.rows and 0 <= c < self.cols
```

Hàm này được gọi ở mọi nơi cần truy cập neighbor: tính adjacent count, DFS flood fill, chord reveal — đảm bảo không bao giờ bị lỗi `IndexError`.

### 5.3. Bảng tổng hợp Edge Cases

| Tình huống | Cách xử lý |
|------------|------------|
| Click đầu tiên trúng vị trí mìn | Mìn chưa được sinh → sinh mìn với vùng an toàn 3×3 → luôn OK |
| Click phải vào ô đã mở | Bỏ qua — chỉ toggle flag trên ô `"hidden"` hoặc `"flagged"` |
| Click trái vào ô đã cắm cờ | Bỏ qua — phải bỏ cờ trước mới mở được |
| Double-click vào ô số (Chord) | Nếu số flag xung quanh = giá trị ô → tự động mở các ô hidden xung quanh |
| Số mìn ≥ tổng ô − 9 | Tự động clamp (giới hạn) lại → đảm bảo luôn có đủ ô cho vùng an toàn |
| Kiểm tra thắng | Dùng bộ đếm O(1): `revealed_count == rows × cols − num_mines` |
| Game over | Hiển thị tất cả mìn 💣, đánh dấu flag sai ❌, disable toàn bộ grid |

### 5.4. Các mức độ khó

| Mức | Kích thước | Số mìn | Tỷ lệ mìn |
|-----|------------|--------|-----------|
| Dễ | 9 × 9 | 10 | 12.3% |
| Trung bình | 16 × 16 | 40 | 15.6% |
| Khó | 16 × 30 | 99 | 20.6% |
| Tùy chỉnh | 5–30 × 5–30 | Tùy ý | Tùy ý |

---

## 6. Hướng dẫn chạy chương trình

### Yêu cầu

- **Python 3.6+** (khuyến nghị Python 3.8 trở lên)
- **Tkinter** — thư viện GUI có sẵn trong Python (không cần cài thêm)
- Hệ điều hành: Windows / macOS / Linux

### Chạy game

```bash
python minesweeper.py
```

### Cấu trúc file (sau refactor)

- `minesweeper.py`: entrypoint để chạy game (giữ lệnh chạy như cũ)
- `minesweeper_model.py`: Model (logic game)
- `minesweeper_ai.py`: AI solver
- `minesweeper_view.py`: View (Tkinter UI)
- `minesweeper_controller.py`: Controller (kết nối Model–View–AI)

### Điều khiển

| Thao tác | Chức năng |
|----------|-----------|
| 🖱️ Click trái | Mở ô |
| 🖱️ Click phải | Cắm/bỏ cờ 🚩 |
| 🖱️ Double-click trái | Chord reveal (mở nhanh ô xung quanh ô số) |
| 🙂 Nút mặt cười | Reset game (giữ nguyên difficulty) |
| 💡 Hint | AI thực hiện 1 bước (step-by-step) |
| 🤖 Solve | AI tự chơi liên tục cho tới khi kết thúc |
| Menu **Game** | Chọn mức độ khó hoặc tùy chỉnh kích thước |
| Menu **Hướng dẫn** | Cách chơi, thuật toán DFS, AI Solver |
