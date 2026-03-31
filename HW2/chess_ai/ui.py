from __future__ import annotations

try:
    import tkinter as tk
    from tkinter import scrolledtext, messagebox
except ImportError:
    tk = None
    scrolledtext = None
    messagebox = None

import threading
import os
from typing import Callable, Optional

import chess

from .agents.alphabeta import AlphaBetaAgent
from .agents.mcts import MCTSAgent
from .comparison import ComparisonSummary, compare_agents_parallel
from .config import AlphaBetaConfig, MCTSConfig, MatchConfig

# Color scheme
LIGHT_SQUARE = "#F0D9B5"
DARK_SQUARE = "#B58863"
HIGHLIGHT_SQUARE = "#BACA44"
HIGHLIGHT_MOVE = "#BACA44"

# Piece symbols using Unicode
PIECE_SYMBOLS = {
    chess.PAWN: "♟",
    chess.KNIGHT: "♞",
    chess.BISHOP: "♝",
    chess.ROOK: "♜",
    chess.QUEEN: "♛",
    chess.KING: "♚",
}

PIECE_UNICODE = {
    (chess.PAWN, chess.WHITE): "♙",
    (chess.PAWN, chess.BLACK): "♟",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.ROOK, chess.WHITE): "♖",
    (chess.ROOK, chess.BLACK): "♜",
    (chess.QUEEN, chess.WHITE): "♕",
    (chess.QUEEN, chess.BLACK): "♛",
    (chess.KING, chess.WHITE): "♔",
    (chess.KING, chess.BLACK): "♚",
}


class ChessBoardWidget(tk.Canvas):
    """Interactive chess board widget."""

    def __init__(
        self,
        parent: tk.Widget,
        board: chess.Board,
        on_move: Callable[[chess.Move], None] | None = None,
        square_size: int = 60,
        **kwargs,
    ) -> None:
        self.board = board
        self.square_size = square_size
        self.on_move = on_move
        self.selected_square: Optional[int] = None
        self.legal_moves_from_selected: list[chess.Move] = []

        super().__init__(
            parent,
            width=square_size * 8,
            height=square_size * 8,
            bg="white",
            highlightthickness=0,
            **kwargs,
        )

        self.bind("<Button-1>", self._on_click)
        self.draw_board()

    def draw_board(self) -> None:
        self.delete("all")

        # Draw squares
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)
                x1 = file * self.square_size
                y1 = rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size

                is_light = (file + rank) % 2 == 0
                color = LIGHT_SQUARE if is_light else DARK_SQUARE

                # Highlight selected square and legal moves
                if square == self.selected_square:
                    color = HIGHLIGHT_SQUARE
                else:
                    for move in self.legal_moves_from_selected:
                        if move.to_square == square:
                            color = HIGHLIGHT_MOVE
                            break

                self.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline="black", width=1
                )

                # Draw coordinates
                if file == 0:
                    self.create_text(
                        x1 + 2,
                        y2 - 2,
                        text=str(8 - rank),
                        font=("Arial", 8),
                        anchor="sw",
                    )
                if rank == 7:
                    self.create_text(
                        x2 - 2,
                        y2 - 2,
                        text=chr(97 + file),
                        font=("Arial", 8),
                        anchor="se",
                    )

        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                x = file * self.square_size + self.square_size // 2
                y = (7 - rank) * self.square_size + self.square_size // 2

                symbol = PIECE_UNICODE[(piece.piece_type, piece.color)]
                font = ("Arial", int(self.square_size * 0.8), "bold")

                # Draw a small opposite-color shadow so pieces stay readable
                # on both light and dark squares.
                if piece.color == chess.WHITE:
                    self.create_text(x + 1, y + 1, text=symbol, font=font, fill="#1A1A1A")
                    self.create_text(x, y, text=symbol, font=font, fill="#F8F8F8")
                else:
                    self.create_text(x + 1, y + 1, text=symbol, font=font, fill="#F2F2F2")
                    self.create_text(x, y, text=symbol, font=font, fill="#111111")

    def _on_click(self, event: tk.Event) -> None:
        file = event.x // self.square_size
        rank = 7 - (event.y // self.square_size)

        if file < 0 or file > 7 or rank < 0 or rank > 7:
            return

        square = chess.square(file, rank)

        if self.selected_square is None:
            # Select a piece
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.legal_moves_from_selected = [
                    m for m in self.board.legal_moves if m.from_square == square
                ]
        else:
            # Try to move
            if square == self.selected_square:
                self.selected_square = None
                self.legal_moves_from_selected = []
            else:
                move = chess.Move(self.selected_square, square)
                # Check for promotion
                piece = self.board.piece_at(self.selected_square)
                if piece and piece.piece_type == chess.PAWN:
                    target_rank = chess.square_rank(square)
                    if (piece.color == chess.WHITE and target_rank == 7) or (
                        piece.color == chess.BLACK and target_rank == 0
                    ):
                        # Default to queen promotion
                        move.promotion = chess.QUEEN

                if move in self.board.legal_moves:
                    self.selected_square = None
                    self.legal_moves_from_selected = []
                    if self.on_move:
                        self.on_move(move)
                else:
                    self.selected_square = None
                    self.legal_moves_from_selected = []

        self.draw_board()

    def update_board(self, board: chess.Board) -> None:
        self.board = board.copy(stack=False)
        self.selected_square = None
        self.legal_moves_from_selected = []
        self.draw_board()


class GameWindow(tk.Tk):
    """Main game window with board and game info."""

    def __init__(self, game_title: str = "Chess AI") -> None:
        super().__init__()
        self.title(game_title)
        self.geometry("1100x800")
        self.resizable(True, True)

        # Main layout
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Chess board
        board_frame = tk.Frame(main_frame)
        board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.board_widget: Optional[ChessBoardWidget] = None

        # Right: Game info
        info_frame = tk.Frame(main_frame, width=250)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(0, 0))

        # Status label
        self.status_label = tk.Label(
            info_frame,
            text="Initializing...",
            font=("Arial", 11, "bold"),
            wraplength=230,
        )
        self.status_label.pack(anchor="w", pady=3)

        # Move history
        tk.Label(info_frame, text="Move History:", font=("Arial", 9, "bold")).pack(
            anchor="w", pady=(5, 2)
        )
        self.move_history = scrolledtext.ScrolledText(
            info_frame, height=6, width=28, font=("Courier", 8)
        )
        self.move_history.pack(padx=3, pady=2, fill=tk.BOTH, expand=False)
        self.move_history.config(state=tk.DISABLED)

        # FEN display
        tk.Label(info_frame, text="FEN:", font=("Arial", 8, "bold")).pack(
            anchor="w", pady=(5, 2)
        )
        self.fen_text = tk.Text(info_frame, height=2, width=28, font=("Courier", 7))
        self.fen_text.pack(padx=3, pady=2, fill=tk.X)
        self.fen_text.config(state=tk.DISABLED)

        # Buttons
        button_frame = tk.Frame(info_frame)
        button_frame.pack(padx=3, pady=5, fill=tk.X)

        self.reset_button = tk.Button(
            button_frame, text="Reset", command=self.on_reset_requested, width=10
        )
        self.reset_button.pack(side=tk.LEFT, padx=2)

        self.resign_button = tk.Button(
            button_frame, text="Resign", command=self.on_resign_requested, width=10
        )
        self.resign_button.pack(side=tk.LEFT, padx=2)

        self.board = chess.Board()
        self.move_san_list: list[str] = []
        self.on_reset_callback: Optional[Callable[[], None]] = None
        self.on_resign_callback: Optional[Callable[[], None]] = None
        self.thinking_label: Optional[tk.Label] = None

    def set_board(
        self, board: chess.Board, on_move: Callable[[chess.Move], None] | None = None
    ) -> None:
        if self.board_widget:
            self.board_widget.destroy()

        self.board = board.copy(stack=False)
        self.board_widget = ChessBoardWidget(
            self, self.board, on_move=on_move, square_size=70
        )
        self.board_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

    def update_from_board(self, board: chess.Board) -> None:
        self.board = board.copy(stack=False)
        if self.board_widget:
            self.board_widget.update_board(board)
        self._update_move_history()
        self._update_fen()

    def _update_move_history(self) -> None:
        self.move_san_list = []
        temp_board = chess.Board()
        for move in self.board.move_stack:
            self.move_san_list.append(temp_board.san(move))
            temp_board.push(move)

        self.move_history.config(state=tk.NORMAL)
        self.move_history.delete("1.0", tk.END)
        move_str = ""
        for i, move in enumerate(self.move_san_list):
            if i % 2 == 0:
                move_str += f"{i // 2 + 1}. {move} "
            else:
                move_str += f"{move}\n"
        if len(self.move_san_list) % 2 == 1:
            move_str += "\n"
        self.move_history.insert("1.0", move_str)
        self.move_history.see(tk.END)
        self.move_history.config(state=tk.DISABLED)

    def _update_fen(self) -> None:
        self.fen_text.config(state=tk.NORMAL)
        self.fen_text.delete("1.0", tk.END)
        self.fen_text.insert("1.0", self.board.fen())
        self.fen_text.config(state=tk.DISABLED)

    def set_status(self, text: str) -> None:
        self.status_label.config(text=text)
        self.update_idletasks()

    def show_thinking(self, text: str = "AI is thinking...") -> None:
        self.set_status(text)

    def hide_thinking(self) -> None:
        self._update_status_from_board()

    def _update_status_from_board(self) -> None:
        if self.board.is_game_over(claim_draw=True):
            outcome = self.board.outcome(claim_draw=True)
            if outcome:
                if outcome.winner is None:
                    text = "Game Over: Draw"
                elif outcome.winner == chess.WHITE:
                    text = "Game Over: White Wins"
                else:
                    text = "Game Over: Black Wins"
            else:
                text = "Game Over"
        else:
            turn = "White" if self.board.turn == chess.WHITE else "Black"
            text = f"{turn} to move ({len(self.board.move_stack)} plies)"

        self.set_status(text)

    def on_reset_requested(self) -> None:
        if self.on_reset_callback:
            self.on_reset_callback()

    def on_resign_requested(self) -> None:
        if self.on_resign_callback:
            self.on_resign_callback()

    def show_error(self, title: str, message: str) -> None:
        messagebox.showerror(title, message)

    def show_info(self, title: str, message: str) -> None:
        messagebox.showinfo(title, message)


class GameController:
    """Manages a chess game with optional AI players."""

    def __init__(self, window: GameWindow) -> None:
        self.window = window
        self.board = chess.Board()
        self.white_agent: Optional[object] = None
        self.black_agent: Optional[object] = None
        self.game_thread: Optional[threading.Thread] = None
        self.is_running = False

        window.set_board(self.board, on_move=self._on_player_move)
        window.on_reset_callback = self.reset_game
        window.on_resign_callback = self.resign_game

    def set_agents(
        self, white_agent: object | None, black_agent: object | None
    ) -> None:
        self.white_agent = white_agent
        self.black_agent = black_agent

    def _on_player_move(self, move: chess.Move) -> None:
        if self.board.is_game_over(claim_draw=True):
            return

        if (self.board.turn == chess.WHITE and self.white_agent is None) or (
            self.board.turn == chess.BLACK and self.black_agent is None
        ):
            if move in self.board.legal_moves:
                self.board.push(move)
                self.window.update_from_board(self.board)
                self._continue_game()

    def _continue_game(self) -> None:
        if self.game_thread and self.game_thread.is_alive():
            return

        self.game_thread = threading.Thread(target=self._game_loop, daemon=True)
        self.game_thread.start()

    def _game_loop(self) -> None:
        while not self.board.is_game_over(claim_draw=True):
            current_agent = (
                self.white_agent if self.board.turn == chess.WHITE else self.black_agent
            )

            if current_agent is None:
                break

            agent_name = getattr(current_agent, "name", "Agent")
            self.window.show_thinking(f"{agent_name} is thinking...")

            try:
                move = current_agent.choose_move(self.board)
                if move is None or move not in self.board.legal_moves:
                    winner = "Black" if self.board.turn == chess.WHITE else "White"
                    self.window.set_status(f"Illegal/no move! {winner} wins.")
                    break

                self.board.push(move)
                self.window.update_from_board(self.board)
            except Exception as e:
                self.window.show_error("Error", f"Agent error: {e}")
                break

        self.window.hide_thinking()

    def reset_game(self) -> None:
        self.board = chess.Board()
        self.window.update_from_board(self.board)

    def resign_game(self) -> None:
        winner = "Black" if self.board.turn == chess.WHITE else "White"
        self.window.set_status(f"{winner} wins (opponent resigned).")


def play_game_ui(
    white_agent: object | None = None,
    black_agent: object | None = None,
    title: str = "Chess AI",
) -> None:
    """Launch interactive chess game UI."""
    if tk is None:
        raise ImportError(
            "tkinter is not available. Please install Python with tkinter support."
        )

    window = GameWindow(title)
    controller = GameController(window)
    controller.set_agents(white_agent, black_agent)
    controller.window.update_from_board(controller.board)
    window.mainloop()


class ComparisonBarChart(tk.Canvas):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, width=360, height=240, bg="white", highlightthickness=1)

    def render(self, summary: ComparisonSummary) -> None:
        self.delete("all")
        metrics = [
            ("Win rate", summary.alpha_beta_win_rate * 100.0, summary.mcts_win_rate * 100.0, "%"),
            (
                "Avg move time",
                summary.alpha_beta_avg_move_time_ms,
                summary.mcts_avg_move_time_ms,
                "ms",
            ),
            (
                "Avg nodes",
                summary.alpha_beta_avg_nodes_visited,
                summary.mcts_avg_nodes_visited,
                "",
            ),
        ]

        self.create_text(180, 12, text="Comparison Overview", font=("Arial", 10, "bold"))

        top = 38
        bar_left = 130
        bar_max_width = 190
        row_height = 62

        for idx, (label, ab_val, mcts_val, suffix) in enumerate(metrics):
            y = top + idx * row_height
            max_value = max(ab_val, mcts_val, 1e-9)
            ab_width = int((ab_val / max_value) * bar_max_width)
            mcts_width = int((mcts_val / max_value) * bar_max_width)

            self.create_text(12, y, text=label, anchor="w", font=("Arial", 9, "bold"))

            self.create_rectangle(bar_left, y - 12, bar_left + ab_width, y, fill="#1f77b4", outline="")
            self.create_text(
                bar_left + bar_max_width + 8,
                y - 6,
                text=f"AB {ab_val:.2f}{suffix}",
                anchor="w",
                font=("Arial", 8),
            )

            self.create_rectangle(bar_left, y + 8, bar_left + mcts_width, y + 20, fill="#ff7f0e", outline="")
            self.create_text(
                bar_left + bar_max_width + 8,
                y + 14,
                text=f"MCTS {mcts_val:.2f}{suffix}",
                anchor="w",
                font=("Arial", 8),
            )


class UnifiedChessApp(tk.Tk):
    MODE_HUMAN_AB = "Human vs AlphaBeta"
    MODE_HUMAN_MCTS = "Human vs MCTS"
    MODE_AB_MCTS = "AlphaBeta vs MCTS"
    MODE_COMPARE = "Compare methods"
    PRESET_FAST = "Fast"
    PRESET_BALANCED = "Balanced"
    PRESET_STRONG = "Strong"

    PRESET_VALUES = {
        PRESET_FAST: {
            "ab_depth": 3,
            "ab_qdepth": 2,
            "mcts_iter": 600,
            "rollout": 24,
            "games": 6,
            "max_plies": 180,
        },
        PRESET_BALANCED: {
            "ab_depth": 4,
            "ab_qdepth": 3,
            "mcts_iter": 1200,
            "rollout": 40,
            "games": 10,
            "max_plies": 300,
        },
        PRESET_STRONG: {
            "ab_depth": 5,
            "ab_qdepth": 3,
            "mcts_iter": 2000,
            "rollout": 50,
            "games": 20,
            "max_plies": 400,
        },
    }

    def __init__(self) -> None:
        super().__init__()
        self.title("HW2 Chess AI")
        self.geometry("1250x820")
        self.resizable(True, True)
        self.minsize(980, 700)

        self._board_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self.board = chess.Board()
        self.white_agent: Optional[object] = None
        self.black_agent: Optional[object] = None

        self.mode_var = tk.StringVar(value=self.MODE_HUMAN_AB)
        self.preset_var = tk.StringVar(value=self.PRESET_BALANCED)
        self.ab_depth_var = tk.IntVar(value=4)
        self.ab_quiescence_var = tk.IntVar(value=3)
        self.mcts_iterations_var = tk.IntVar(value=1200)
        self.mcts_rollout_var = tk.IntVar(value=40)
        self.compare_games_var = tk.IntVar(value=10)
        self.compare_workers_var = tk.IntVar(value=max(1, min(os.cpu_count() or 1, 8)))
        self.max_plies_var = tk.IntVar(value=300)
        self.seed_var = tk.IntVar(value=42)

        self._build_layout()
        self._refresh_board_and_info()

    def _build_layout(self) -> None:
        controls = tk.LabelFrame(self, text="Controls")
        controls.pack(fill=tk.X, padx=8, pady=8)

        top_row = tk.Frame(controls)
        top_row.pack(fill=tk.X, padx=6, pady=(4, 2))
        bottom_row = tk.Frame(controls)
        bottom_row.pack(fill=tk.X, padx=6, pady=(2, 4))

        tk.Label(top_row, text="Preset", font=("Arial", 9, "bold")).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        tk.OptionMenu(
            top_row,
            self.preset_var,
            self.PRESET_FAST,
            self.PRESET_BALANCED,
            self.PRESET_STRONG,
            command=lambda _value: self._apply_selected_preset(),
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(top_row, text="Mode", font=("Arial", 9, "bold")).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        tk.OptionMenu(
            top_row,
            self.mode_var,
            self.MODE_HUMAN_AB,
            self.MODE_HUMAN_MCTS,
            self.MODE_AB_MCTS,
            self.MODE_COMPARE,
            command=lambda _value: self._on_mode_changed(),
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(top_row, text="AB depth").pack(side=tk.LEFT)
        tk.Spinbox(
            top_row, from_=1, to=6, width=4, textvariable=self.ab_depth_var
        ).pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(top_row, text="AB q-depth").pack(side=tk.LEFT)
        tk.Spinbox(
            top_row, from_=0, to=5, width=4, textvariable=self.ab_quiescence_var
        ).pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(top_row, text="MCTS iter").pack(side=tk.LEFT)
        tk.Spinbox(
            top_row,
            from_=100,
            to=5000,
            increment=100,
            width=6,
            textvariable=self.mcts_iterations_var,
        ).pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(top_row, text="Rollout").pack(side=tk.LEFT)
        tk.Spinbox(
            top_row, from_=10, to=120, width=4, textvariable=self.mcts_rollout_var
        ).pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(bottom_row, text="Games").pack(side=tk.LEFT)
        self.games_spinbox = tk.Spinbox(
            bottom_row, from_=2, to=100, width=5, textvariable=self.compare_games_var
        )
        self.games_spinbox.pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(bottom_row, text="Workers").pack(side=tk.LEFT)
        self.workers_spinbox = tk.Spinbox(
            bottom_row,
            from_=1,
            to=max(1, os.cpu_count() or 1),
            width=5,
            textvariable=self.compare_workers_var,
        )
        self.workers_spinbox.pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(bottom_row, text="Max plies").pack(side=tk.LEFT)
        self.max_plies_spinbox = tk.Spinbox(
            bottom_row,
            from_=60,
            to=600,
            increment=20,
            width=6,
            textvariable=self.max_plies_var,
        )
        self.max_plies_spinbox.pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(bottom_row, text="Seed").pack(side=tk.LEFT)
        tk.Spinbox(
            bottom_row, from_=0, to=99999, width=7, textvariable=self.seed_var
        ).pack(side=tk.LEFT, padx=(2, 8))

        tk.Button(bottom_row, text="Start", command=self.start).pack(
            side=tk.LEFT, padx=(6, 4)
        )
        tk.Button(bottom_row, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=4)

        main = tk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.board_widget = ChessBoardWidget(left, self.board, on_move=self._on_player_move, square_size=78)
        self.board_widget.pack(side=tk.LEFT, anchor="n")

        right = tk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self.status_label = tk.Label(right, text="Ready", font=("Arial", 11, "bold"), anchor="w")
        self.status_label.pack(fill=tk.X, pady=(0, 6))

        tk.Label(right, text="Move History", font=("Arial", 9, "bold"), anchor="w").pack(fill=tk.X)
        self.move_history = scrolledtext.ScrolledText(right, height=12, width=44, font=("Courier", 9))
        self.move_history.pack(fill=tk.X, pady=(2, 8))
        self.move_history.config(state=tk.DISABLED)

        tk.Label(right, text="Comparison Results", font=("Arial", 9, "bold"), anchor="w").pack(fill=tk.X)
        self.results_text = tk.Text(right, height=8, width=44, font=("Courier", 9))
        self.results_text.pack(fill=tk.X, pady=(2, 8))
        self.results_text.config(state=tk.DISABLED)

        self.chart = ComparisonBarChart(right)
        self.chart.pack(fill=tk.X)

        self.legend_label = tk.Label(
            right,
            text="Blue: Alpha-Beta   Orange: MCTS",
            anchor="w",
            font=("Arial", 8),
        )
        self.legend_label.pack(fill=tk.X, pady=(4, 0))

        self._apply_selected_preset()
        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        compare_mode = self.mode_var.get() == self.MODE_COMPARE
        self.games_spinbox.config(state=tk.NORMAL if compare_mode else tk.DISABLED)
        self.workers_spinbox.config(state=tk.NORMAL if compare_mode else tk.DISABLED)

    def _apply_selected_preset(self) -> None:
        values = self.PRESET_VALUES[self.preset_var.get()]
        self.ab_depth_var.set(values["ab_depth"])
        self.ab_quiescence_var.set(values["ab_qdepth"])
        self.mcts_iterations_var.set(values["mcts_iter"])
        self.mcts_rollout_var.set(values["rollout"])
        self.compare_games_var.set(values["games"])
        self.max_plies_var.set(values["max_plies"])

    def _build_alphabeta(self) -> AlphaBetaAgent:
        return AlphaBetaAgent(config=self._get_alphabeta_config())

    def _get_alphabeta_config(self) -> AlphaBetaConfig:
        return AlphaBetaConfig(
            max_depth=self.ab_depth_var.get(),
            quiescence_depth=self.ab_quiescence_var.get(),
        )

    def _build_mcts(self) -> MCTSAgent:
        return MCTSAgent(config=self._get_mcts_config(), seed=self.seed_var.get())

    def _get_mcts_config(self) -> MCTSConfig:
        return MCTSConfig(
            iterations=self.mcts_iterations_var.get(),
            rollout_depth=self.mcts_rollout_var.get(),
        )

    def start(self) -> None:
        self._stop_current_worker()
        mode = self.mode_var.get()
        if mode == self.MODE_COMPARE:
            self._start_compare_mode()
            return

        with self._board_lock:
            self.board = chess.Board()
            if mode == self.MODE_HUMAN_AB:
                self.white_agent = None
                self.black_agent = self._build_alphabeta()
            elif mode == self.MODE_HUMAN_MCTS:
                self.white_agent = None
                self.black_agent = self._build_mcts()
            else:
                self.white_agent = self._build_alphabeta()
                self.black_agent = self._build_mcts()

        self._refresh_board_and_info()
        self._set_status("Game started")
        self._start_ai_loop_if_needed()

    def reset(self) -> None:
        self._stop_current_worker()
        with self._board_lock:
            self.board = chess.Board()
        self.white_agent = None
        self.black_agent = None
        self._refresh_board_and_info()
        self._set_status("Board reset")

    def _on_player_move(self, move: chess.Move) -> None:
        if self.mode_var.get() == self.MODE_COMPARE:
            return

        with self._board_lock:
            if self.board.is_game_over(claim_draw=True):
                return

            is_human_turn = (self.board.turn == chess.WHITE and self.white_agent is None) or (
                self.board.turn == chess.BLACK and self.black_agent is None
            )
            if not is_human_turn or move not in self.board.legal_moves:
                return

            self.board.push(move)

        self._refresh_board_and_info()
        self._start_ai_loop_if_needed()

    def _start_ai_loop_if_needed(self) -> None:
        with self._board_lock:
            if self.board.is_game_over(claim_draw=True):
                self._refresh_board_and_info()
                return
            current_agent = self.white_agent if self.board.turn == chess.WHITE else self.black_agent
            if current_agent is None:
                return

        self._worker_thread = threading.Thread(target=self._ai_loop_worker, daemon=True)
        self._worker_thread.start()

    def _ai_loop_worker(self) -> None:
        while not self._stop_event.is_set():
            with self._board_lock:
                if self.board.is_game_over(claim_draw=True):
                    break
                current_agent = self.white_agent if self.board.turn == chess.WHITE else self.black_agent
                if current_agent is None:
                    break

            self.after(0, lambda: self._set_status(f"{getattr(current_agent, 'name', 'Agent')} is thinking..."))
            try:
                move = current_agent.choose_move(self.board)
            except Exception as exc:
                self.after(0, lambda e=exc: self._show_error(f"Agent error: {e}"))
                return

            with self._board_lock:
                if self._stop_event.is_set():
                    return
                if move is None or move not in self.board.legal_moves:
                    break
                self.board.push(move)

            self.after(0, self._refresh_board_and_info)

        self.after(0, self._refresh_board_and_info)

    def _start_compare_mode(self) -> None:
        with self._board_lock:
            self.board = chess.Board()
        self.white_agent = None
        self.black_agent = None
        self._refresh_board_and_info()
        self._set_results_text("Running comparison...\n")

        games = max(2, self.compare_games_var.get())
        workers = max(1, self.compare_workers_var.get())
        config = MatchConfig(
            max_plies=max(60, self.max_plies_var.get()),
            random_opening_plies=2,
            seed=self.seed_var.get(),
        )
        ab_config = self._get_alphabeta_config()
        mcts_config = self._get_mcts_config()

        def worker() -> None:
            summary = compare_agents_parallel(
                games=games,
                alpha_beta_config=ab_config,
                mcts_config=mcts_config,
                config=config,
                workers=workers,
                progress_callback=lambda done, total: self.after(
                    0,
                    lambda: self._set_status(
                        f"Comparing methods ({workers} workers): {done}/{total} games"
                    ),
                ),
            )
            self.after(0, lambda: self._render_comparison_summary(summary))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _render_comparison_summary(self, summary: ComparisonSummary) -> None:
        self._set_status("Comparison complete")
        text = (
            f"Games: {summary.games}\n"
            f"Alpha-Beta wins: {summary.alpha_beta_wins} ({summary.alpha_beta_win_rate:.1%})\n"
            f"MCTS wins:       {summary.mcts_wins} ({summary.mcts_win_rate:.1%})\n"
            f"Draws:           {summary.draws}\n"
            f"AB avg time:     {summary.alpha_beta_avg_move_time_ms:.2f} ms\n"
            f"MCTS avg time:   {summary.mcts_avg_move_time_ms:.2f} ms\n"
            f"AB avg nodes:    {summary.alpha_beta_avg_nodes_visited:.2f}\n"
            f"MCTS avg nodes:  {summary.mcts_avg_nodes_visited:.2f}\n"
            f"Score:           {summary.alpha_beta_points:.1f} - {summary.mcts_points:.1f}"
        )
        self._set_results_text(text)
        self.chart.render(summary)

    def _set_status(self, text: str) -> None:
        self.status_label.config(text=text)

    def _set_results_text(self, text: str) -> None:
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", text)
        self.results_text.config(state=tk.DISABLED)

    def _refresh_board_and_info(self) -> None:
        with self._board_lock:
            board_snapshot = self.board.copy(stack=True)
        self.board_widget.update_board(board_snapshot)
        self._update_move_history(board_snapshot)

        if board_snapshot.is_game_over(claim_draw=True):
            outcome = board_snapshot.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                self._set_status("Game over: draw")
            else:
                winner = "White" if outcome.winner == chess.WHITE else "Black"
                self._set_status(f"Game over: {winner} wins")
        else:
            turn = "White" if board_snapshot.turn == chess.WHITE else "Black"
            self._set_status(f"{turn} to move")

    def _update_move_history(self, board_snapshot: chess.Board) -> None:
        move_san_list: list[str] = []
        temp_board = chess.Board()
        for move in board_snapshot.move_stack:
            move_san_list.append(temp_board.san(move))
            temp_board.push(move)

        text = ""
        for i, move in enumerate(move_san_list):
            if i % 2 == 0:
                text += f"{i // 2 + 1}. {move} "
            else:
                text += f"{move}\n"

        self.move_history.config(state=tk.NORMAL)
        self.move_history.delete("1.0", tk.END)
        self.move_history.insert("1.0", text)
        self.move_history.see(tk.END)
        self.move_history.config(state=tk.DISABLED)

    def _show_error(self, message: str) -> None:
        messagebox.showerror("Error", message)

    def _stop_current_worker(self) -> None:
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.2)
        self._stop_event.clear()


def launch_unified_ui() -> None:
    if tk is None:
        raise ImportError(
            "tkinter is not available. Please install Python with tkinter support."
        )

    app = UnifiedChessApp()
    app.mainloop()
