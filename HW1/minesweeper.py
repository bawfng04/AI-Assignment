"""minesweeper.py — entrypoint chạy Minesweeper (refactor bản tách module).

Chạy:
  python minesweeper.py

Modules:
  - minesweeper_model.py
  - minesweeper_ai.py
  - minesweeper_view.py
  - minesweeper_controller.py
"""

from __future__ import annotations

import tkinter as tk

from minesweeper_controller import MinesweeperController


def main() -> None:
    root = tk.Tk()
    root.title("Minesweeper — AI Assignment")

    MinesweeperController(root)

    root.mainloop()


if __name__ == "__main__":
    main()
