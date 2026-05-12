"""
Validate notebook: chạy thử từng code cell tuần tự,
bắt lỗi NameError/ImportError và báo cáo cell nào lỗi.
Bỏ qua các cell cần GPU/Atari ROM (pip install, train, evaluate).
"""
import json
import sys

# Các PATTERN nếu xuất hiện thì SKIP TOÀN BỘ cell
FULL_SKIP_PATTERNS = [
    "!pip", "!AutoROM",
    "subprocess.run(['AutoROM",
    "train_model(",
    "evaluate(",
    "display(Video(",
    "gym.make(",
    "make_atari_env(",
    "google.colab",
]

# Các dòng cần patch (thay thế để chạy được locally)
LINE_PATCHES = [
    # gym / ale
    ("import ale_py",                                       ""),
    ("gym.register_envs(ale_py)",                           ""),
    ("import gymnasium as gym",                             "import gymnasium as gym"),  # giữ nguyên, gym cài được local
    # tqdm notebook
    ("from tqdm.notebook import tqdm",                      "from tqdm import tqdm"),
    # IPython display
    ("from IPython.display import Video, display",          "Video = display = lambda *a, **kw: None"),
    # cv2
    ("import cv2",                                          "import cv2"),  # ok nếu cài
]

nb = json.load(open("HW4_D3QN.ipynb", encoding="utf-8"))
namespace = {}

passed = 0
failed = 0
skipped = 0

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue

    src = "".join(cell["source"])

    # Kiểm tra skip toàn bộ cell
    if any(p in src for p in FULL_SKIP_PATTERNS):
        first_line = src.strip().split("\n")[0][:60]
        print(f"  [SKIP] Cell {i:2d} — {first_line}")
        skipped += 1
        continue

    # Patch từng dòng
    patched_lines = []
    for line in src.split("\n"):
        new_line = line
        for old, new in LINE_PATCHES:
            if old in line:
                new_line = line.replace(old, new)
                break
        patched_lines.append(new_line)
    src_patched = "\n".join(patched_lines)

    try:
        exec(compile(src_patched, f"<cell_{i}>", "exec"), namespace)
        first_line = src.strip().split("\n")[0][:60]
        print(f"  [OK]   Cell {i:2d} — {first_line}")
        passed += 1
    except Exception as e:
        first_line = src.strip().split("\n")[0][:60]
        print(f"  [FAIL] Cell {i:2d} — {first_line}")
        print(f"         ERROR: {type(e).__name__}: {e}")
        failed += 1

print()
print(f"Result: {passed} passed, {failed} failed, {skipped} skipped")
if failed == 0:
    print("[VALIDATION PASSED] Safe to upload to Colab!")
else:
    print("[VALIDATION FAILED] Fix the FAIL cells before uploading.")
    sys.exit(1)
