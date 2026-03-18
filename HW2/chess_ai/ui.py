from __future__ import annotations

try:
    import tkinter as tk
    from tkinter import scrolledtext, messagebox
except ImportError:
    tk = None
    scrolledtext = None
    messagebox = None

import threading
from typing import Callable, Optional

import chess

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
                color = "white" if piece.color == chess.WHITE else "black"

                self.create_text(
                    x,
                    y,
                    text=symbol,
                    font=("Arial", int(self.square_size * 0.8)),
                    fill=color if piece.color == chess.WHITE else "white",
                )

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
