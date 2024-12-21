import numpy as np

BOARD_WIDTH = 7
BOARD_HEIGHT = 6
WINNING_LENGTH = 4

MASTER_DEPTH = 2

MAX_DEPTH = 7

CPU = 1
HUMAN = 2

WORK_TAG = 0
COMPLETED_TAG = 1

class Board:
    def __init__(self, width=7, height=6):
        self.width = width
        self.height = height
        self.board = np.full((height, width), 0, dtype=int)

    def __str__(self):
        return str(self.board).replace('1', 'X').replace('2', 'O').replace('0', '.')

    def get_cell(self, row: int, col: int) -> int:
        return int(self.board[row, col])

    def get_column(self, col: int):
        return self.board[:, col]

    def get_row(self, row: int):
        return self.board[row, :]

    def column_full(self, col: int):
        return int(self.board[0, col]) != 0

    def column_height(self, col: int):
        return np.count_nonzero(self.get_column(col))

    def load_board(self, board):
        self.board = board

    def copy(self):
        new_board = Board(self.width, self.height)
        new_board.load_board(self.board.copy())
        return new_board

    def move_legal(self, col: int):
        if col < 0 or col >= self.board.shape[1]:
            return False
        elif self.column_full(col):
            # print(self.board)
            return False
        else:
            return True

    def make_move(self, col: int, player: int) -> (int, int):
        if not self.move_legal(col):
            raise ValueError("Illegal move")

        # Find the first empty row in the column
        for row in range(self.board.shape[0] - 1, -1, -1):
            if self.get_cell(row, col) == 0:
                self.board[row, col] = player
                return row, col

    def undo_move(self, col):
        if col < 0 or col >= self.board.shape[1]:
            return False
        if self.board[self.board.shape[0] - 1, col] == 0:
            return False

        for row in range(self.board.shape[0]):
            if self.get_cell(row, col) != 0:
                self.board[row, col] = 0
                return True

        return False

    def game_end(self, col: int) -> (bool, int):
        row = self.board.shape[0] - self.column_height(col)

        if row == BOARD_HEIGHT:
            return False, 0

        player = self.get_cell(row, col)

        # print(f"{player=}, {row=}, {col=}")

        # Check horizontal
        count = 1
        # Check right
        for i in range(1, WINNING_LENGTH):
            if col + i < BOARD_WIDTH and self.get_cell(row, col + i) == player:
                count += 1
            else:
                break
        # Check left
        for i in range(1, WINNING_LENGTH):
            if col - i >= 0 and self.get_cell(row, col - i) == player:
                count += 1
            else:
                break
        if count >= WINNING_LENGTH:
            # print("horizontal")
            return True, player

        # Check vertical
        count = 1
        for i in range(1, WINNING_LENGTH):
            if row + i < BOARD_HEIGHT and self.get_cell(row + i, col) == player:
                count += 1
            else:
                break
        if count >= WINNING_LENGTH:
            # print("vertical")
            return True, player

        # Check diagonal right
        count = 1
        # Check downwards
        for i in range(1, WINNING_LENGTH):
            if row + i < BOARD_HEIGHT and col + i < BOARD_WIDTH and self.get_cell(row + i, col + i) == player:
                count += 1
            else:
                break
        # Check upwards
        for i in range(1, WINNING_LENGTH):
            if row - i >= 0 and col - i >= 0 and self.get_cell(row - i, col - i) == player:
                count += 1
            else:
                break
        if count >= WINNING_LENGTH:
            # print("diagonal right")
            return True, player

        # Check diagonal left
        count = 1
        # Check downwards
        for i in range(1, WINNING_LENGTH):
            if row + i < BOARD_HEIGHT and col - i >= 0 and self.get_cell(row + i, col - i) == player:
                count += 1
            else:
                break
        # Check upwards
        for i in range(1, WINNING_LENGTH):
            if row - i >= 0 and col + i < BOARD_WIDTH and self.get_cell(row - i, col + i) == player:
                count += 1
            else:
                break
        if count >= WINNING_LENGTH:
            # print("diagonal left")
            return True, player

        return False, 0
