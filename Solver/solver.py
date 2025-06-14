"""
Backtracking Algorithm for Solving Sudoku:

The backtracking algorithm is a depth-first search technique used to explore all possible configurations of the Sudoku board
until a valid solution is found. Here's how it works:

1. Traverse the board cell by cell, looking for an empty cell (value 0).
2. For each empty cell, try placing a number from 1 to 9.
3. For each candidate number, check if placing the number is valid:
   - It must not already exist in the same row.
   - It must not already exist in the same column.
   - It must not already exist in the same 3x3 sub-grid.
4. If the number is valid, temporarily place it in the cell and recursively attempt to solve the rest of the board.
5. If the recursive call finds a solution, the function returns True.
6. If no valid number leads to a solution, reset the cell to 0 (backtrack) and return False to explore other options.

This process continues until the entire board is filled with valid numbers, or it’s determined that no solution exists.
"""

# Apparantly sudoklue does it better :(

from datetime import datetime
from enum import Enum
from itertools import combinations
from itertools import product
from collections import Counter
import numpy as np


class SudokuTechnique(Enum):
    NAKED_SINGLE_ROW_OR_COLUMN = 1,
    NAKED_SINGLE_BLOCK = 2,
    BOX_LINE_INTERACTION = 3,
    METADATA_BOX_LINE_INTERACTION = 4,
    COLUMN_LINE_INTERACTION = 5,
    NAKED_SINGLE = 6,
    NAKED_DOUBLE = 7
    NAKED_TRIPLE = 8,
    METADATA_FROM_CELL_NOTATION = 9,
    CELL_NOTATION_FROM_METADATA = 10,
    SKYSCRAPER = 11,
    TWO_STRING_KITE = 12,
    EMPTY_RECTANGLE = 13,
    XY_WING = 14

def update_cell_notation(cell_notation, row, col, digit):
    """
    Update the cells notation as if someone added a digit in (row, col)
    :param cell_notation:
    :param row:
    :param col:
    :param digit:
    :return:
    """
    for r in range(9):
        if digit in cell_notation[r][col]:
            cell_notation[r][col].remove(digit)

    for c in range(9):
        if digit in cell_notation[row][c]:
            cell_notation[row][c].remove(digit)

    block_row = row // 3
    block_col = col // 3

    for relative_row in range(3):
        for relative_col in range(3):
            if digit in cell_notation[3 * block_row + relative_row][3 * block_col + relative_col]:
                cell_notation[3 * block_row + relative_row][3 * block_col + relative_col].remove(digit)

    cell_notation[row][col] = []

class Board:
    def __init__(self, board):
        self.board = board.copy()
        self.original_board = self.board.copy()
        self.metadata_on_board = []
        self.last_used_technique = None
        self.last_step_description_str = ""

        self.cell_notation = [[0 for _ in range(9)] for _ in range(9)] # a 9 x 9 grid
        for row in range(9):
            for col in range(9):
                if board[row][col] != 0:
                    self.cell_notation[row][col] = []
                else:
                    self.cell_notation[row][col] = list(range(1, 10))
        for row in range(9):
            for col in range(9):
                if board[row][col] != 0:
                    update_cell_notation(self.cell_notation, row, col, board[row][col])

def is_valid(board, row, column, digit):
    """
    Check if a number is not in the block / row / column
    :param board:
    :param row:
    :param column:
    :param digit:
    :return:
    """

    for i in range(9):
        if board[row][i] == digit or board[i][column] == digit:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (column // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == digit:
                return False
    return True

def validate_neccessary_coherenet_soduku(board):
    """
    The function verifies no number appears already twice in the same row / column / block.
    It is not a guarantee that the soduko is solvable.
    :param board:
    :return:
    """
    for r in range(9):
        for c in range(9):
            n = board[r][c]
            if n == 0:
                continue
            board[r][c] = 0  # Temporarily clear the cell
            if not is_valid(board, r, c, n):
                board[r][c] = n  # Restore original value
                return False
            board[r][c] = n  # Restore original value
    return True

def solve_sudoku_automatic(board: np.ndarray) -> bool:
    # print("start solving sudoku at: " + str(datetime.now()))

    # Preliminary check to ensure all filled cells are valid
    if not validate_neccessary_coherenet_soduku(board):
        return False

    count = [0]
    def solve():
        count[0] += 1
        #print("count is: " + str(count[0]))
        if count[0] % 10000 == 0:
            pass
            # print(board)

        # MRV heuristic: find the cell with the fewest options
        min_options = 10
        target = None
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    options = [n for n in range(1, 10) if is_valid(r, c, n)]
                    if len(options) < min_options:
                        min_options = len(options)
                        target = (r, c, options)
                    if min_options == 1:
                        break
            if min_options == 1:
                break

        if target is None:
            return True  # solved

        r, c, options = target
        for num in options:
            board[r][c] = num
            if solve():
                return True
            board[r][c] = 0
        return False

    return solve()

class MetaDataType(Enum):
    LINE_OF_DIGIT = 1    # Example for a metadata: [LINE_OF_DIGIT, 4, 1, 2, 0] = digit 4 in block (1,2) is in the first relative line
    COL_OF_DIGIT = 2     # Example for a metadata: [COL_OF_DIGIT, 4, 1, 2, 0] = digit 4 in block (1,2) is in the first relative column

def remove_first_match(arr, target):
    for i in range(len(arr)):
        if np.array_equal(arr[i], target):
            return np.delete(arr, i, axis=0)
    return arr  # No match found

def is_digit_in_block(board, digit, block_row, block_col) -> bool:
    for r in range(3):
        for c in range(3):
            if board[3 * block_row + r][3 * block_col + c] == digit:
                return True
    return False

def is_digit_in_line(board, digit, line_idx) -> bool:
    for c in range(9):
        if board[line_idx][c] == digit:
            return True
    return False

def is_digit_in_col(board, digit, col_idx) -> bool:
    for r in range(9):
        if board[r][col_idx] == digit:
            return True
    return False

def safe_replace(board_inst: Board, digit, row, col):
    if board_inst.board[row][col] != 0:
        raise RuntimeError(f"Attempting to replace a digit")
    board_inst.board[row][col] = digit
    update_cell_notation(board_inst.cell_notation, row, col, digit)

def technique_naked_single_row_or_column(board_inst: Board) -> bool:
    """
    If only one number is left in a column / line.
    :param board:
    :param metadata_on_board_inst:
    :return:
    """
    for row in range(9):
        missing_numbers_in_row = get_missing_numbers_in_row(board_inst.board, row)
        if len(missing_numbers_in_row) == 1:
            safe_replace(board_inst,
                         missing_numbers_in_row[0],
                         row = row,
                         col = np.where(board_inst.board[row] == 0)[0][0])
            board_inst.last_step_description_str = f"In row {row + 1} there is only one missing digit: {missing_numbers_in_row[0]}"
            board_inst.last_used_technique = SudokuTechnique.NAKED_SINGLE_ROW_OR_COLUMN
            return True

    for col in range(9):
        missing_numbers_in_col = get_missing_numbers_in_col(board_inst.board, col)
        if len(missing_numbers_in_col) == 1:
            safe_replace(board_inst,
                         missing_numbers_in_col[0],
                         row=np.where(board_inst.board[:, col] == 0)[0][0],
                         col=col)
            board_inst.last_step_description_str = f"In column {col + 1} there is only one missing digit: {missing_numbers_in_col[0]}"
            board_inst.last_used_technique = SudokuTechnique.NAKED_SINGLE_ROW_OR_COLUMN
            return True

    return False

def technique_naked_single_block(board_inst: Board) -> bool:
    """
    If only one number is left in a block.
    :return:
    """
    for block_row in range(3):
        for block_col in range(3):
            num_of_missing_numbers_in_block = np.count_nonzero(board_inst.board[3 * block_row : 3 * block_row + 3, 3 * block_col : 3 * block_col + 3] == 0)
            if num_of_missing_numbers_in_block == 1:
                missing_digit = 0
                for digit in range(1, 10):
                    if not is_digit_in_block(board_inst.board, digit, block_row, block_col):
                        missing_digit = digit

                for relative_row in range(3):
                    for relative_col in range(3):
                        if board_inst.board[3 * block_row + relative_row][3 * block_col + relative_col] == 0:
                            safe_replace(board_inst,
                                         missing_digit,
                                         row=3 * block_row + relative_row,
                                         col=3 * block_col + relative_col)
                            board_inst.last_step_description_str = f"In block ({block_row + 1},{block_col + 1}) there is only one missing digit: {missing_digit}"
                            board_inst.last_used_technique = SudokuTechnique.NAKED_SINGLE_BLOCK
                            return True
    return False

def technique_box_line_interaction(board_inst: Board) -> bool:
    """
    The technique eliminates places for a digit based on the digit already placed in the line / column.
    If a digit is found that for it only one place in a given block is available it updates the board accordingly,
    and returns True.
    :return:
    """

    # First try to do it without using metadata.
    for digit in range(1, 10):
        for block_row in range(3):
            for block_col in range(3):
                if is_digit_in_block(board_inst.board, digit, block_row, block_col):
                    continue
                optional_relative_places_for_digit = np.array([[0, 0], [0, 1], [0, 2],
                                                               [1, 0], [1, 1], [1, 2],
                                                               [2, 0], [2, 1], [2, 2]])
                for optional_relative_place in optional_relative_places_for_digit:
                    if board_inst.board[3 * block_row + optional_relative_place[0]][3 * block_col + optional_relative_place[1]] != 0:
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)
                    if is_digit_in_line(board_inst.board, digit, 3 * block_row + optional_relative_place[0]):
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)
                    if is_digit_in_col(board_inst.board, digit, 3 * block_col + optional_relative_place[1]):
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)

                if optional_relative_places_for_digit.shape == (1,2):
                    safe_replace(board_inst,
                                 digit,
                                 row = 3 * block_row + optional_relative_places_for_digit[0][0],
                                 col = 3 * block_col + optional_relative_places_for_digit[0][1])
                    board_inst.last_step_description_str = (f"The digit {digit} in block ({block_row + 1},{block_col + 1}) can only "
                                                 f"be placed in line"
                                                 f" {3 * block_row + optional_relative_places_for_digit[0][0] + 1} "
                                                 f"and in column"
                                                 f" {3 * block_col + optional_relative_places_for_digit[0][1] + 1} ")
                    board_inst.last_used_technique = SudokuTechnique.BOX_LINE_INTERACTION
                    return True

    # Ok. Sometimes we need to use the metadata
    for digit in range(1, 10):
        for block_row in range(3):
            for block_col in range(3):


                if is_digit_in_block(board_inst.board, digit, block_row, block_col):
                    continue
                optional_relative_places_for_digit = np.array([[0, 0], [0, 1], [0, 2],
                                                               [1, 0], [1, 1], [1, 2],
                                                               [2, 0], [2, 1], [2, 2]])
                for optional_relative_place in optional_relative_places_for_digit:
                    if board_inst.board[3 * block_row + optional_relative_place[0]][3 * block_col + optional_relative_place[1]] != 0:
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)
                    if is_digit_in_line(board_inst.board, digit, 3 * block_row + optional_relative_place[0]):
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)
                    if is_digit_in_col(board_inst.board, digit, 3 * block_col + optional_relative_place[1]):
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)

                    for metadata in board_inst.metadata_on_board:
                        if metadata[1] == digit:
                            if metadata[0] == MetaDataType.LINE_OF_DIGIT and \
                                metadata[2] == block_row and \
                                metadata[4] == optional_relative_place[0] and \
                                metadata[3] != block_col:
                                optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)

                            if metadata[0] == MetaDataType.COL_OF_DIGIT and \
                                metadata[3] == block_col and \
                                metadata[4] == optional_relative_place[1] and \
                                metadata[2] != block_row:
                                optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)

                if optional_relative_places_for_digit.shape == (1,2):
                    safe_replace(board_inst,
                                 digit,
                                 row = 3 * block_row + optional_relative_places_for_digit[0][0],
                                 col = 3 * block_col + optional_relative_places_for_digit[0][1])
                    board_inst.last_step_description_str = (f"The digit {digit} in block ({block_row + 1},{block_col + 1}) can only "
                                                 f"be placed in line"
                                                 f" {3 * block_row + optional_relative_places_for_digit[0][0] + 1} "
                                                 f"and in column"
                                                 f" {3 * block_col + optional_relative_places_for_digit[0][1] + 1} ")
                    board_inst.last_used_technique = SudokuTechnique.BOX_LINE_INTERACTION

                    if block_col == 1 and block_row == 0 and digit == 8:
                        print("ob ho")

                    return True

    return False

def technique_metadata_box_line_interaction(board_inst: Board) -> bool:
    """
    The technique is used to find new metadata.
    The specific metadata it finds is to find a specific row / col in a block where a digit must be.
    :param board:
    :param metadata_on_board_inst:
    :return:
    """
    for digit in range(1, 10):
        for block_row in range(3):
            for block_col in range(3):
                if is_digit_in_block(board_inst.board, digit, block_row, block_col):
                    continue
                optional_relative_places_for_digit = np.array([[0, 0], [0, 1], [0, 2],
                                                               [1, 0], [1, 1], [1, 2],
                                                               [2, 0], [2, 1], [2, 2]])

                for optional_relative_place in optional_relative_places_for_digit:
                    if board_inst.board[3 * block_row + optional_relative_place[0]][3 * block_col + optional_relative_place[1]] != 0:
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit,
                                                                                optional_relative_place)
                        continue
                    if is_digit_in_line(board_inst.board, digit, 3 * block_row + optional_relative_place[0]):
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit,
                                                                                optional_relative_place)
                        continue
                    if is_digit_in_col(board_inst.board, digit, 3 * block_col + optional_relative_place[1]):
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit,
                                                                                optional_relative_place)
                        continue

                if optional_relative_places_for_digit.shape == (1,2):
                    continue # Another technique should find it

                is_same_optional_line_for_digit = True

                optional_relative_line_for_digit = optional_relative_places_for_digit[0][0]
                for optional_relative_place in optional_relative_places_for_digit:
                    if optional_relative_place[0] != optional_relative_line_for_digit:
                        is_same_optional_line_for_digit = False
                if is_same_optional_line_for_digit:
                    new_metadata = [MetaDataType.LINE_OF_DIGIT, digit, block_row, block_col, optional_relative_line_for_digit]

                    if new_metadata not in board_inst.metadata_on_board:
                        board_inst.metadata_on_board.append(new_metadata)
                        board_inst.last_step_description_str = (
                            f"New metadata. The digit {digit} in block ({block_row + 1},{block_col + 1}) can only "
                            f"be placed in line"
                            f" {3 * block_row + optional_relative_line_for_digit + 1} ")
                        board_inst.last_used_technique = SudokuTechnique.METADATA_BOX_LINE_INTERACTION
                        return True

                is_same_optional_col_for_digit = True
                optional_relative_col_for_digit = optional_relative_places_for_digit[0][1]
                for optional_relative_place in optional_relative_places_for_digit:
                    if optional_relative_place[1] != optional_relative_col_for_digit:
                        is_same_optional_col_for_digit = False
                if is_same_optional_col_for_digit:
                    new_metadata = [MetaDataType.COL_OF_DIGIT, digit, block_row, block_col, optional_relative_col_for_digit]
                    if new_metadata not in board_inst.metadata_on_board:
                        board_inst.metadata_on_board.append(new_metadata)
                        board_inst.last_step_description_str = (
                            f"New metadata. The digit {digit} in block ({block_row + 1},{block_col + 1}) can only "
                            f"be placed in column"
                            f" {3 * block_col + optional_relative_col_for_digit + 1} ")
                        board_inst.last_used_technique = SudokuTechnique.METADATA_BOX_LINE_INTERACTION
                        return True

    return False

def get_missing_numbers_in_row(board, row):
    missing_numbers = list(range(1,10))
    for column in range(9):
        if board[row][column] != 0:
            missing_numbers.remove(board[row][column])
    return missing_numbers

def get_missing_numbers_in_col(board, col):
    missing_numbers = list(range(1,10))
    for row in range(9):
        if board[row][col] != 0:
            missing_numbers.remove(board[row][col])
    return missing_numbers

def technique_column_line_interaction(board_inst: Board) -> bool:
    """
    The technique procedure:
    Look at a specific line / column. look at the missing numbers.
    If only one number is possible, because all other numbers are already in the column, we found it!
    :return:
    """
    for row in range(9):
        for col in range(9):
            if board_inst.board[row][col] == 0:
                optional_numbers_for_slot = get_missing_numbers_in_row(board_inst.board, row)
                optional_numbers_for_slot_copy = optional_numbers_for_slot.copy()
                for missing_number in optional_numbers_for_slot_copy:
                    if np.any(board_inst.board[:, col] == missing_number):
                        optional_numbers_for_slot.remove(missing_number)
                if len(optional_numbers_for_slot) == 1:
                    safe_replace(board_inst,
                                 optional_numbers_for_slot[0],
                                 row=row,
                                 col=col)
                    board_inst.last_step_description_str = (f"In row {row + 1}, missing numbers are: {optional_numbers_for_slot_copy}."
                                                 f" However only {optional_numbers_for_slot[0]} can be placed in column {col + 1}.")
                    board_inst.last_used_technique = SudokuTechnique.COLUMN_LINE_INTERACTION
                    return True

    for col in range(9):
        for row in range(9):
            if board_inst.board[row][col] == 0:
                optional_numbers_for_slot = get_missing_numbers_in_col(board_inst.board, col)
                optional_numbers_for_slot_copy = optional_numbers_for_slot.copy()
                for missing_number in optional_numbers_for_slot_copy:
                    if np.any(board_inst.board[row, :] == missing_number):
                        optional_numbers_for_slot.remove(missing_number)
                if len(optional_numbers_for_slot) == 1:
                    safe_replace(board_inst.board,
                                 optional_numbers_for_slot[0],
                                 row=row,
                                 col=col)
                    board_inst.last_step_description_str = (
                        f"In column {col + 1}, missing numbers are: {optional_numbers_for_slot_copy}."
                        f"However all but one candidate cannot be intersecting with row {row + 1}."
                        f"Therefore missing digit is: {optional_numbers_for_slot[0]}")
                    board_inst.last_used_technique = SudokuTechnique.COLUMN_LINE_INTERACTION
                    return True

    return False

def technique_naked_single(board_inst: Board) -> bool:
    """
    The technique looks at the cell notation matrix. if a cell is found with only one option - yay.
    :param board_inst:
    :return:
    """
    for row in range(9):
        for col in range(9):
            if len(board_inst.cell_notation[row][col]) == 1:
                board_inst.last_used_technique = SudokuTechnique.NAKED_SINGLE
                board_inst.last_step_description_str = "Looking at cell notation options, only one optional digit is possible in" \
                                                        f" ({row+1}, {col+1}) - The digit {board_inst.cell_notation[row][col][0]}"
                safe_replace(board_inst,
                             board_inst.cell_notation[row][col][0],
                             row,
                             col)
                return True

    return False

def technique_naked_double(board_inst: Board) -> bool:
    for block_row in range(3):
        for block_col in range(3):
            for g1, g2 in combinations(list(range(1, 10)), 2):
                combined = [g1, g2] # a digit double
                num_of_cells_with_only_these_double_options = []
                for relative_row in range(3):
                    for relative_col in range(3):
                        if board_inst.board[3 * block_row + relative_row][3 * block_col + relative_col] == 0 and \
                           all(item in combined for item in board_inst.cell_notation[3 * block_row + relative_row][3 * block_col + relative_col]):
                            num_of_cells_with_only_these_double_options.append([relative_row, relative_col])
                if len(num_of_cells_with_only_these_double_options) == 2:
                    found_new_information = False
                    for digit in combined:
                        for relative_row in range(3):
                            for relative_col in range(3):
                                if [relative_row, relative_col] not in num_of_cells_with_only_these_double_options and \
                                   digit in board_inst.cell_notation[3 * block_row + relative_row][3 * block_col + relative_col]:
                                    board_inst.cell_notation[3 * block_row + relative_row][3 * block_col + relative_col].remove(digit)
                                    found_new_information = True
                    if found_new_information:
                        board_inst.last_used_technique = SudokuTechnique.NAKED_TRIPLE
                        board_inst.last_step_description_str = f"In block ({block_row + 1},{block_col + 1}), the digits " \
                                                                f"{combined} along with relative positions {num_of_cells_with_only_these_double_options}" \
                                                                f" are a naked double. Therefore we can eliminate those digits from the cell notation" \
                                                                f" of all other cells in block"
                        return True

    for row in range(9):
        for c1, c2 in combinations(list(range(1, 10)), 2):
            combined = [c1, c2]  # a digit triplet
            num_of_cells_with_only_these_double_options = []
            for col in range(9):
                if board_inst.board[row][col] == 0 and\
                    all(item in combined for item in board_inst.cell_notation[row][col]):
                    num_of_cells_with_only_these_double_options.append([row, col])
            if len(num_of_cells_with_only_these_double_options) == 2:
                found_new_information = False
                for digit in combined:
                    for col in range(9):
                        if [row, col] not in num_of_cells_with_only_these_double_options and \
                                digit in board_inst.cell_notation[row][col]:
                            board_inst.cell_notation[row][col].remove(digit)
                            found_new_information = True
                if found_new_information:
                    board_inst.last_used_technique = SudokuTechnique.NAKED_TRIPLE
                    board_inst.last_step_description_str = f"In row {row + 1}, the digits " \
                                                           f"{combined} along with relative positions {num_of_cells_with_only_these_double_options}" \
                                                           f" are a naked double. Therefore we can eliminate those digits from the cell notation" \
                                                           f" of all other cells in the row"
                    return True

    for col in range(9):
        for r1, r2 in combinations(list(range(1, 10)), 2):
            combined = [r1, r2]  # a digit triplet
            num_of_cells_with_only_these_double_options = []
            for row in range(9):
                if board_inst.board[row][col] == 0 and\
                    all(item in combined for item in board_inst.cell_notation[row][col]):
                    num_of_cells_with_only_these_double_options.append([row, col])
            if len(num_of_cells_with_only_these_double_options) == 2:
                found_new_information = False
                for digit in combined:
                    for row in range(9):
                        if [row, col] not in num_of_cells_with_only_these_double_options and \
                                digit in board_inst.cell_notation[row][col]:
                            board_inst.cell_notation[row][col].remove(digit)
                            found_new_information = True
                if found_new_information:
                    board_inst.last_used_technique = SudokuTechnique.NAKED_TRIPLE
                    board_inst.last_step_description_str = f"In col {col + 1}, the digits " \
                                                           f"{combined} along with relative positions {num_of_cells_with_only_these_double_options}" \
                                                           f" are a naked double. Therefore we can eliminate those digits from the cell notation" \
                                                           f" of all other cells in the column"
                    return True

    return False

def technique_naked_triple(board_inst: Board) -> bool:
    """
    In a block, where 3 cells share only 3 total different options of numbers, all other cells in the block
    cannot be filled with these numbers.
    This is also true for lines / columns.
    see https://www.youtube.com/watch?v=Mh8-MICdO6s&ab_channel=LearnSomething 9:31 - for an example.
    :param board_inst:
    :return:
    """
    for block_row in range(3):
        for block_col in range(3):
            for g1, g2, g3 in combinations(list(range(1, 10)), 3):
                combined = [g1, g2, g3] # a digit triplet
                num_of_cells_with_only_these_triplet_options = []
                for relative_row in range(3):
                    for relative_col in range(3):
                        if board_inst.board[3 * block_row + relative_row][3 * block_col + relative_col] == 0 and \
                           all(item in combined for item in board_inst.cell_notation[3 * block_row + relative_row][3 * block_col + relative_col]):
                            num_of_cells_with_only_these_triplet_options.append([relative_row, relative_col])
                if len(num_of_cells_with_only_these_triplet_options) == 3:
                    found_new_information = False
                    for digit in combined:
                        for relative_row in range(3):
                            for relative_col in range(3):
                                if [relative_row, relative_col] not in num_of_cells_with_only_these_triplet_options and \
                                   digit in board_inst.cell_notation[3 * block_row + relative_row][3 * block_col + relative_col]:
                                    board_inst.cell_notation[3 * block_row + relative_row][3 * block_col + relative_col].remove(digit)
                                    found_new_information = True
                    if found_new_information:
                        board_inst.last_used_technique = SudokuTechnique.NAKED_TRIPLE
                        board_inst.last_step_description_str = f"In block ({block_row + 1},{block_col + 1}), the digits " \
                                                                f"{combined} along with relative positions {num_of_cells_with_only_these_triplet_options}" \
                                                                f" are a naked triplet. Therefore we can eliminate those digits from the cell notation" \
                                                                f" of all other cells in block"
                        return True

    for row in range(9):
        for c1, c2, c3 in combinations(list(range(1, 10)), 3):
            combined = [c1, c2, c3]  # a digit triplet
            num_of_cells_with_only_these_triplet_options = []
            for col in range(9):
                if board_inst.board[row][col] == 0 and\
                    all(item in combined for item in board_inst.cell_notation[row][col]):
                    num_of_cells_with_only_these_triplet_options.append([row, col])
            if len(num_of_cells_with_only_these_triplet_options) == 3:
                found_new_information = False
                for digit in combined:
                    for col in range(9):
                        if [row, col] not in num_of_cells_with_only_these_triplet_options and \
                                digit in board_inst.cell_notation[row][col]:
                            board_inst.cell_notation[row][col].remove(digit)
                            found_new_information = True
                if found_new_information:
                    board_inst.last_used_technique = SudokuTechnique.NAKED_TRIPLE
                    board_inst.last_step_description_str = f"In row {row + 1}, the digits " \
                                                           f"{combined} along with relative positions {num_of_cells_with_only_these_triplet_options}" \
                                                           f" are a naked triplet. Therefore we can eliminate those digits from the cell notation" \
                                                           f" of all other cells in the row"
                    return True

    for col in range(9):
        for r1, r2, r3 in combinations(list(range(1, 10)), 3):
            combined = [r1, r2, r3]  # a digit triplet
            num_of_cells_with_only_these_triplet_options = []
            for row in range(9):
                if board_inst.board[row][col] == 0 and\
                    all(item in combined for item in board_inst.cell_notation[row][col]):
                    num_of_cells_with_only_these_triplet_options.append([row, col])
            if len(num_of_cells_with_only_these_triplet_options) == 3:
                found_new_information = False
                for digit in combined:
                    for row in range(9):
                        if [row, col] not in num_of_cells_with_only_these_triplet_options and \
                                digit in board_inst.cell_notation[row][col]:
                            board_inst.cell_notation[row][col].remove(digit)
                            found_new_information = True
                if found_new_information:
                    board_inst.last_used_technique = SudokuTechnique.NAKED_TRIPLE
                    board_inst.last_step_description_str = f"In col {col + 1}, the digits " \
                                                           f"{combined} along with relative positions {num_of_cells_with_only_these_triplet_options}" \
                                                           f" are a naked triplet. Therefore we can eliminate those digits from the cell notation" \
                                                           f" of all other cells in the column"
                    return True

    return False

def technique_metadata_from_cell_notation(board_inst: Board) -> bool:
    """
    Example: Cell notation in block (1,2) should be that 8 is on the first line
        |    6     **59***  *4789** | **489**  **489**     1    |    3        2     **578** |
        |    3        2     *4789** |    6        5     **49*** | **478**     1     **78*** |
        | **158**  **15***  **148** | **23***     7     **23*** | **458**     6        9    |
        -------------------------------
        | **189**     6        3    | *2589**  **289**  **259** | **17***     4     **17*** |
        |    2        4        5    |    1        6        7    |    9        8        3    |
        |    7     **19***  **189** | *3489**  **489**  **349** |    2        5        6    |
        -------------------------------
        |    4        3     **29*** | **259**     1        6    | **58***     7     **258** |
        | **15***     7     **12*** | **25***     3        8    |    6        9        4    |
        | **159**     8        6    |    7     **249**  *2459** | **15***     3     **125** |
        -------------------------------
    """
    for digit in range(1,10):
        for block_row in range(3):
            for block_col in range(3):
                if is_digit_in_block(board_inst.board, digit, block_row, block_col):
                    continue
                optional_relative_places_for_digit = np.array([[0, 0], [0, 1], [0, 2],
                                                               [1, 0], [1, 1], [1, 2],
                                                               [2, 0], [2, 1], [2, 2]])

                for optional_relative_place in optional_relative_places_for_digit:
                    if board_inst.board[3 * block_row + optional_relative_place[0]][
                        3 * block_col + optional_relative_place[1]] != 0:
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit,
                                                                                optional_relative_place)
                        continue
                    if digit not in board_inst.cell_notation[3 * block_row + optional_relative_place[0]][3 * block_col + optional_relative_place[1]]:
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit,
                                                                                optional_relative_place)
                        continue

                if optional_relative_places_for_digit.shape == (1, 2):
                    continue  # Another technique should find it

                is_same_optional_line_for_digit = True
                optional_relative_line_for_digit = optional_relative_places_for_digit[0][0]
                for optional_relative_place in optional_relative_places_for_digit:
                    if optional_relative_place[0] != optional_relative_line_for_digit:
                        is_same_optional_line_for_digit = False
                if is_same_optional_line_for_digit:
                    new_metadata = [MetaDataType.LINE_OF_DIGIT, digit, block_row, block_col,
                                    optional_relative_line_for_digit]

                    if new_metadata not in board_inst.metadata_on_board:
                        board_inst.metadata_on_board.append(new_metadata)
                        board_inst.last_step_description_str = (
                            f"New metadata. The digit {digit} in block ({block_row + 1},{block_col + 1}) can only "
                            f"be placed in line"
                            f" {3 * block_row + optional_relative_line_for_digit + 1} based on the current"
                            f" cell notation view.")
                        board_inst.last_used_technique = SudokuTechnique.METADATA_FROM_CELL_NOTATION
                        return True

                is_same_optional_col_for_digit = True
                optional_relative_col_for_digit = optional_relative_places_for_digit[0][1]
                for optional_relative_place in optional_relative_places_for_digit:
                    if optional_relative_place[1] != optional_relative_col_for_digit:
                        is_same_optional_col_for_digit = False
                if is_same_optional_col_for_digit:
                    new_metadata = [MetaDataType.COL_OF_DIGIT, digit, block_row, block_col,
                                    optional_relative_col_for_digit]
                    if new_metadata not in board_inst.metadata_on_board:
                        board_inst.metadata_on_board.append(new_metadata)
                        board_inst.last_step_description_str = (
                            f"New metadata. The digit {digit} in block ({block_row + 1},{block_col + 1}) can only "
                            f"be placed in column"
                            f" {3 * block_col + optional_relative_col_for_digit + 1} based on the current"
                            f" cell notation view.")
                        board_inst.last_used_technique = SudokuTechnique.METADATA_FROM_CELL_NOTATION
                        return True

    return False

def technique_update_cell_notation_from_metadata(board_inst: Board) -> bool:
    for metadata in board_inst.metadata_on_board:
        if len(metadata) == 5: # i.e it did not update the cell notation yet.
            updated_something = False
            for cell_idx in range(9):
                if metadata[0] == MetaDataType.LINE_OF_DIGIT and \
                    metadata[1] in board_inst.cell_notation[3 * metadata[2] + metadata[4]][cell_idx] and \
                    cell_idx // 3 != metadata[3]:
                    board_inst.cell_notation[3 * metadata[2] + metadata[4]][cell_idx].remove(metadata[1])
                    board_inst.last_step_description_str = (f"Updated cell notation based on metadata that digit {metadata[1]} must be"
                                                            f" placed in line {3 * metadata[2] + metadata[4]}")
                    board_inst.last_used_technique = SudokuTechnique.CELL_NOTATION_FROM_METADATA
                    updated_something = True
                if metadata[0] == MetaDataType.COL_OF_DIGIT and \
                    metadata[1] in board_inst.cell_notation[cell_idx][3 * metadata[3] + metadata[4]] and \
                    cell_idx // 3 != metadata[2]:
                    board_inst.cell_notation[cell_idx][3 * metadata[3] + metadata[4]].remove(metadata[1])
                    board_inst.last_step_description_str = (
                        f"Updated cell notation based on metadata that digit {metadata[1]} must be"
                        f" placed in column {3 * metadata[3] + metadata[4]}")
                    board_inst.last_used_technique = SudokuTechnique.CELL_NOTATION_FROM_METADATA
                    updated_something = True
            if updated_something:
                return True
            metadata.append(1) # I.E it already updated the cell notation
    return False

def remove_uninflicted_positions(row, col, l = None):
    """
    Given a list of positions in the grid l, positions which are not directly influenced by
    :param row1:
    :param col1:
    :param l:
    :return:
    """
    if l is None:
        l = [(i, j) for i in range(9) for j in range(9)]
    if (row, col) in l:
        l.remove((row, col))
    l_copy = l.copy()
    for r,c in l_copy:
        if r != row and \
            c != col and \
            not(r // 3 == row // 3 and c // 3 == col // 3):
            l.remove((r,c))
    return l

def technique_skyscraper(board_inst: Board) -> bool:
    for digit in range(1, 10):
        for row1 in range(9):
            for row2 in range(row1 + 1, 9):
                if sum(row_cell_notation.count(digit) for row_cell_notation in board_inst.cell_notation[row1]) == 2 and \
                   sum(row_cell_notation.count(digit) for row_cell_notation in board_inst.cell_notation[row2]) == 2:
                    row1_col_positions = [col for col, optional_digits in enumerate(board_inst.cell_notation[row1]) if digit in optional_digits]
                    row2_col_positions = [col for col, optional_digits in enumerate(board_inst.cell_notation[row2]) if digit in optional_digits]
                    same_col = 0
                    different_col1 = 0
                    different_col2 = 0
                    if min(row1_col_positions) == min(row2_col_positions):
                        same_col = min(row1_col_positions)
                        different_col1 = max(row1_col_positions)
                        different_col2 = max(row2_col_positions)
                        if max(row1_col_positions) == max(row2_col_positions): # This is an x - wing. not the technique we need to implement in this function.
                            continue
                    elif max(row1_col_positions) == max(row2_col_positions):
                        same_col = max(row1_col_positions)
                        different_col1 = min(row1_col_positions)
                        different_col2 = min(row2_col_positions)
                        if min(row1_col_positions) == min(row2_col_positions): # This is an x-wing. not the technique we need to implement in this function.
                            continue
                    else:
                        continue

                    changed_anything = False
                    removed_digit_from_positions = []

                    # Remove the possible digit from the inflicted upon by the other two unperpendicular points
                    mutual_inflicted_positions = remove_uninflicted_positions(row2, different_col2, remove_uninflicted_positions(row1, different_col1))
                    # Verify it is not a small skyscraper
                    if (row1, same_col) in mutual_inflicted_positions:
                        mutual_inflicted_positions.remove((row1, same_col))
                    if (row2, same_col) in mutual_inflicted_positions:
                        mutual_inflicted_positions.remove((row2, same_col))
                    for mutual_inflicted_position in mutual_inflicted_positions:
                        if digit in board_inst.cell_notation[mutual_inflicted_position[0]][mutual_inflicted_position[1]]:
                            board_inst.cell_notation[mutual_inflicted_position[0]][mutual_inflicted_position[1]].remove(digit)
                            removed_digit_from_positions.append(mutual_inflicted_position)
                            changed_anything = True

                    if changed_anything:
                        board_inst.last_used_technique = SudokuTechnique.SKYSCRAPER
                        board_inst.last_step_description_str = (f"A skyscraper of the digit {digit} was found build from the positions:"
                                                                f" ({row1},{same_col}) - ({row1},{different_col1}) - ({row2},{same_col}) - ({row2},{different_col2}).\n"
                                                                f"Removing {digit} from being possible in positions: {removed_digit_from_positions}")
                        return True

    # Vertical skyscraper
        for col1 in range(9):
            for col2 in range(col1 + 1, 9):
                if sum(col_cell_notation.count(digit) for col_cell_notation in
                       [board_inst.cell_notation[roww][col1] for roww in range(9)]) == 2 and \
                        sum(col_cell_notation.count(digit) for col_cell_notation in
                            [board_inst.cell_notation[roww][col2] for roww in range(9)]) == 2:
                    col1_row_positions = [row for row, optional_digits in enumerate([board_inst.cell_notation[roww][col1] for roww in range(9)])
                                          if digit in optional_digits]
                    col2_row_positions = [row for row, optional_digits in enumerate([board_inst.cell_notation[roww][col2] for roww in range(9)])
                                          if digit in optional_digits]
                    same_row = 0
                    different_row1 = 0
                    different_row2 = 0
                    if min(col1_row_positions) == min(col2_row_positions):
                        same_row = min(col1_row_positions)
                        different_row1 = max(col1_row_positions)
                        different_row2 = max(col2_row_positions)
                        if max(col1_row_positions) == max(
                                col2_row_positions):  # This is an x - wing. not the technique we need to implement in this function.
                            continue
                    elif max(col1_row_positions) == max(col2_row_positions):
                        same_row = max(col1_row_positions)
                        different_row1 = min(col1_row_positions)
                        different_row2 = min(col2_row_positions)
                        if min(col1_row_positions) == min(
                                col2_row_positions):  # This is an x-wing. not the technique we need to implement in this function.
                            continue
                    else:
                        continue

                    changed_anything = False
                    removed_digit_from_positions = []

                    # Remove the possible digit from the inflicted upon by the other two unperpendicular points
                    mutual_inflicted_positions = remove_uninflicted_positions(different_row2, col2,
                                                                              remove_uninflicted_positions(different_row1, col1))
                    # Verify it is not a small skyscraper
                    if (same_row, col1) in mutual_inflicted_positions:
                        mutual_inflicted_positions.remove((same_col, col1))
                    if (same_row, col2) in mutual_inflicted_positions:
                        mutual_inflicted_positions.remove((same_col, col2))
                    for mutual_inflicted_position in mutual_inflicted_positions:
                        if digit in board_inst.cell_notation[mutual_inflicted_position[0]][
                            mutual_inflicted_position[1]]:
                            board_inst.cell_notation[mutual_inflicted_position[0]][
                                mutual_inflicted_position[1]].remove(digit)
                            removed_digit_from_positions.append(mutual_inflicted_position)
                            changed_anything = True

                    if changed_anything:
                        board_inst.last_used_technique = SudokuTechnique.SKYSCRAPER
                        board_inst.last_step_description_str = (
                            f"A skyscraper of the digit {digit} was found build from the positions:"
                            f" ({same_row},{col1}) - ({different_row1},{col1}) - ({same_row},{col2}) - ({different_row2},{col2}).\n"
                            f"Removing {digit} from being possible in positions: {removed_digit_from_positions}")
                        return True

    return False

def technique_two_string_kite(board_inst: Board) -> bool:
    for digit in range(1, 10):
        for kite_row in range(9):
            for kite_col in range(9):
                if sum(row_cell_notation.count(digit) for row_cell_notation in board_inst.cell_notation[kite_row]) == 2 and \
                    sum(col_cell_notation.count(digit) for col_cell_notation in
                            [board_inst.cell_notation[roww][kite_col] for roww in range(9)]) == 2:
                    if digit in board_inst.cell_notation[kite_row][kite_col]:
                        continue # The string kite must not have an intersection of strings.
                    kite_horizontal_digit_positions = [col for col, optional_digits in enumerate(board_inst.cell_notation[kite_row]) if digit in optional_digits]
                    kite_vertical_digit_positions = [row for row, optional_digits in
                                          enumerate([board_inst.cell_notation[roww][kite_col] for roww in range(9)])
                                          if digit in optional_digits]

                    end_of_kite_col = 0
                    if not ((kite_col // 3 == kite_horizontal_digit_positions[0] // 3) or (kite_col // 3 == kite_horizontal_digit_positions[1] // 3)):
                        continue # Not a kite
                    elif kite_col // 3 == kite_horizontal_digit_positions[0] // 3:
                        end_of_kite_col = kite_horizontal_digit_positions[1]
                    else:
                        end_of_kite_col = kite_horizontal_digit_positions[0]

                    end_of_kite_row = 0
                    if not (kite_row // 3 == kite_vertical_digit_positions[0] // 3 or kite_row // 3 == kite_vertical_digit_positions[1] // 3):
                        continue  # Not a kite
                    elif kite_row // 3 == kite_vertical_digit_positions[0] // 3:
                        end_of_kite_row = kite_vertical_digit_positions[1]
                    else:
                        end_of_kite_row = kite_vertical_digit_positions[0]

                    if digit in board_inst.cell_notation[end_of_kite_row][end_of_kite_col]:
                        board_inst.cell_notation[end_of_kite_row][end_of_kite_col].remove(digit)
                        board_inst.last_used_technique = SudokuTechnique.TWO_STRING_KITE
                        board_inst.last_step_description_str = (f"A kite can be created for the digit {digit} from the positions:"
                                                                f" ({kite_row},{kite_horizontal_digit_positions[0]}) -"
                                                                f" ({kite_row},{kite_horizontal_digit_positions[1]}) -"
                                                                f" ({kite_vertical_digit_positions[0]},{kite_col}) -"
                                                                f" ({kite_vertical_digit_positions[1]},{kite_col})")
                        return True
    return False

def technique_empty_rectangle_continuation(board_inst: Board, digit, stair_row, stair_col) -> bool:
    for row in range(9):
        if row // 3 == stair_row // 3:
            continue
        if digit not in board_inst.cell_notation[row][stair_col]:
            continue
        if sum(cell_notation.count(digit) for cell_notation in
                [board_inst.cell_notation[row][c]
                 for c in range(9)]) != 2:
            continue
        digit_occurrences_in_row = [r for r, digits in enumerate(board_inst.cell_notation[row]) if digit in digits]
        digit_occurrences_in_row.remove(stair_col)

        if digit_occurrences_in_row[0] // 3 == stair_col // 3: # It shall not create a proper empty rectangle technique
            continue

        if digit in board_inst.cell_notation[stair_row][digit_occurrences_in_row[0]]:
            board_inst.cell_notation[stair_row][digit_occurrences_in_row[0]].remove(digit)
            board_inst.last_used_technique = SudokuTechnique.EMPTY_RECTANGLE
            board_inst.last_step_description_str = (f"An empty rectangle was found for the digit {digit}"
                                                    f" in the block: ({stair_row // 3 + 1},{stair_col // 3 + 1}))."
                                                    f" Along with only 2 options for {digit} in row {row + 1}, therefore"
                                                    f"we can eliminate the {digit} in "
                                                    f"position ({stair_row},{digit_occurrences_in_row[0]})")
            return True

    for col in range(9):
        if col // 3 == stair_col // 3:
            continue
        if digit not in board_inst.cell_notation[stair_row][col]:
            continue
        if sum(cell_notation.count(digit) for cell_notation in
               [board_inst.cell_notation[r][col]
                for r in range(9)]) != 2:
            continue
        digit_occurrences_in_col = [c for c, digits in enumerate([board_inst.cell_notation[r][col] for r in range(9)]) if digit in digits]
        digit_occurrences_in_col.remove(stair_row)

        if digit_occurrences_in_col[0] // 3 == stair_row // 3: # It shall not create a proper empty rectangle technique
            continue

        if digit in board_inst.cell_notation[digit_occurrences_in_col[0]][stair_col]:
            board_inst.cell_notation[digit_occurrences_in_col[0]][stair_col].remove(digit)
            board_inst.last_used_technique = SudokuTechnique.EMPTY_RECTANGLE
            board_inst.last_step_description_str = (f"An empty rectangle was found for the digit {digit}"
                                                    f" in the block: ({stair_row // 3 + 1},{stair_col // 3 + 1}))."
                                                    f" Along with only 2 options for {digit} in col {col + 1}, therefore"
                                                    f"we can eliminate the {digit} in "
                                                    f"position ({digit_occurrences_in_col[0]},{stair_col})")
            return True

    return False

def technique_empty_rectangle(board_inst: Board) -> bool:
    for digit in range(1, 10):
        for block_row in range(3):
            for block_col in range(3):
                # First attempt: Look for a stair where the horizontal line holds more than 2 occurrences.
                stair_row = -1
                stair_col = -1
                is_empty_rectangle = True
                for row in range(3 * block_row, 3 * block_row + 3):
                    digit_occurrences_in_block_in_row = sum(cell_notation.count(digit) for cell_notation in
                                                            [board_inst.cell_notation[row][c]
                                                             for c in range(3 * block_col, 3 * block_col + 3)])
                    if digit_occurrences_in_block_in_row > 1:
                        if stair_row == -1:
                            stair_row = row
                        else:
                            is_empty_rectangle = False
                    elif digit_occurrences_in_block_in_row == 1:
                        occurrence_index_in_row = (next((i for i, digits in enumerate([board_inst.cell_notation[row][c]
                                                for c in range(3 * block_col, 3 * block_col + 3)])
                                                       if digit in digits), -1)) + 3 * block_col
                        if stair_col == -1:
                            stair_col = occurrence_index_in_row
                        elif stair_col != occurrence_index_in_row:
                            is_empty_rectangle = False

                if is_empty_rectangle and stair_row != -1 and stair_col != -1:
                    if technique_empty_rectangle_continuation(board_inst, digit, stair_row, stair_col):
                        return True

                # Second attempt: Look for a stair where the vertical line holds more than 2 occurences
                stair_row = -1
                stair_col = -1
                is_empty_rectangle = True
                for col in range(3 * block_col, 3 * block_col + 3):
                    digit_occurrences_in_block_in_col = sum(cell_notation.count(digit) for cell_notation in
                                                            [board_inst.cell_notation[r][col]
                                                             for r in range(3 * block_row, 3 * block_row + 3)])
                    if digit_occurrences_in_block_in_col > 1:
                        if stair_col == -1:
                            stair_col = col
                        else:
                            is_empty_rectangle = False
                    elif digit_occurrences_in_block_in_col == 1:
                        occurrence_index_in_col = (next((i for i, digits in enumerate([board_inst.cell_notation[r][col]
                                                   for r in range(3 * block_row, 3 * block_row + 3)])
                                                     if digit in digits), -1))+ 3 * block_row
                        if stair_row == -1:
                            stair_row = occurrence_index_in_col
                        elif stair_row != occurrence_index_in_col:
                            is_empty_rectangle = False

                if is_empty_rectangle and stair_row != -1 and stair_col != -1:
                    if technique_empty_rectangle_continuation(board_inst, digit, stair_row, stair_col):
                        return True

                # Third attempt - handle special case of digit which can only be in two places in a block and not in the same row or column.
                digit_occurrences_in_block_count = sum(cell_notation.count(digit) for cell_notation in
                                                [board_inst.cell_notation[r][c]
                                                for c in range(3 * block_col, 3 * block_col + 3)
                                                 for r in range(3 * block_row, 3 * block_row + 3)])

                if digit_occurrences_in_block_count == 2:
                    digit_occurrences_in_block = [(i, j) for i in range(3 * block_row, 3 * block_row + 3)
                                                           for j in range(3 * block_col, 3 * block_col + 3) if
                                                  digit in board_inst.cell_notation[i][j]]
                    # If they are on the same line or column we cannot create a stair
                    if digit_occurrences_in_block[0][0] == digit_occurrences_in_block[1][0] or \
                        digit_occurrences_in_block[0][1] == digit_occurrences_in_block[1][1]:
                        continue

                    if technique_empty_rectangle_continuation(board_inst, digit, digit_occurrences_in_block[0][0], digit_occurrences_in_block[1][1]):
                        return True

                    if technique_empty_rectangle_continuation(board_inst, digit, digit_occurrences_in_block[1][0], digit_occurrences_in_block[0][1]):
                        return True

    return False

def technique_xy_wing(board_inst: Board):
    for row in range(9):
        for col in range(9):
            if len(board_inst.cell_notation[row][col]) == 2:
                x_digit = board_inst.cell_notation[row][col][0]
                y_digit = board_inst.cell_notation[row][col][1]
                for z_digit in range(1, 10):
                    if z_digit == x_digit or z_digit == y_digit:
                        continue

                    positions_with_xz_remaining = []
                    positions_with_yz_remaining = []
                    for position in remove_uninflicted_positions(row, col):
                        if len(board_inst.cell_notation[position[0]][position[1]]) == 2:
                            if x_digit in board_inst.cell_notation[position[0]][position[1]] and \
                                z_digit in board_inst.cell_notation[position[0]][position[1]]:
                                positions_with_xz_remaining.append(position)
                            if y_digit in board_inst.cell_notation[position[0]][position[1]] and \
                                z_digit in board_inst.cell_notation[position[0]][position[1]]:
                                positions_with_yz_remaining.append(position)
                    for position_xz, position_yz in list(product(positions_with_xz_remaining, positions_with_yz_remaining)):
                        inflicted_positions = remove_uninflicted_positions(position_yz[0], position_yz[1],
                                                                           remove_uninflicted_positions(position_xz[0], position_xz[1]))
                        affected_positions = []
                        for position in inflicted_positions:
                            if z_digit in board_inst.cell_notation[position[0]][position[1]]:
                                board_inst.cell_notation[position[0]][position[1]].remove(z_digit)
                                affected_positions.append(position)

                        if len(affected_positions) > 0:
                            board_inst.last_used_technique = SudokuTechnique.XY_WING
                            board_inst.last_step_description_str = (f"XY technique was used. where X = {x_digit}, Y = {y_digit}, Z = {z_digit}. "
                                                                    f"XY position: ({row},{col}), YZ position: ({position_yz[0]},{position_yz[1]}), "
                                                                    f"XZ position: ({position_xz[0]},{position_xz[1]}). "
                                                                    f"Inflicted positions: {affected_positions}")
                            return True
    return False

def next_step_sudoku_human_solver(board_inst: Board) -> bool:
    technique_naked_single_row_or_column_success = technique_naked_single_row_or_column(board_inst)
    if technique_naked_single_row_or_column_success:
        return True

    technique_naked_single_block_success = technique_naked_single_block(board_inst)
    if technique_naked_single_block_success:
        return True

    technique_box_line_interaction_success = technique_box_line_interaction(board_inst)
    if technique_box_line_interaction_success:
        return True

    technique_metadata_box_line_interaction_success = technique_metadata_box_line_interaction(board_inst)
    if technique_metadata_box_line_interaction_success:
        return True

    technique_column_line_interaction_success = technique_column_line_interaction(board_inst)
    if technique_column_line_interaction_success:
        return True

    technique_naked_single_success = technique_naked_single(board_inst)
    if technique_naked_single_success:
        return True

    technique_naked_double_success = technique_naked_double(board_inst)
    if technique_naked_double_success:
        return True

    technique_naked_triple_success = technique_naked_triple(board_inst)
    if technique_naked_triple_success:
        return True

    technique_metadata_from_cell_notation_success = technique_metadata_from_cell_notation(board_inst)
    if technique_metadata_from_cell_notation_success:
        return True

    technique_update_cell_notation_from_metadata_success = technique_update_cell_notation_from_metadata(board_inst)
    if technique_update_cell_notation_from_metadata_success:
        return True

    technique_skyscraper_success = technique_skyscraper(board_inst)
    if technique_skyscraper_success:
        return True

    technique_two_string_kite_success = technique_two_string_kite(board_inst)
    if technique_two_string_kite_success:
        return True

    technique_empty_rectangle_success = technique_empty_rectangle(board_inst)
    if technique_empty_rectangle_success:
        return True

    technique_xy_wing_success = technique_xy_wing(board_inst)
    if technique_xy_wing_success:
        return True

    return False

def pad_with_char(s, required_length, char = '*'):
    needed = required_length - len(s)
    left = needed // 2
    right = needed - left
    return char * left + s + char * right

def cli_print_board(board_inst: Board, print_cell_notation = False) -> None:
    if board_inst.last_used_technique is not None:
        print("Last used technique:", board_inst.last_used_technique.name)
        print("Description:", board_inst.last_step_description_str)
    for r in range(9):
        if r % 3 == 0:
            print("-------------------------------")
        for c in range(9):
            if c % 3 == 0:
                print("|", end="")
            print(" ", end="")
            if board_inst.board[r][c] != 0:
                print(str(board_inst.board[r][c]), end=" ")
            else:
                print(" ", end=" ")
        print("|")
    print("-------------------------------")

    if print_cell_notation:
        for r in range(9):
            if r % 3 == 0:
                print("-------------------------------")
            for c in range(9):
                if c % 3 == 0:
                    print("|", end="")
                print(" ", end="")
                if board_inst.board[r][c] != 0:
                    print("   ", str(board_inst.board[r][c]), "   ", sep="", end=" ")
                else:
                    if len(board_inst.cell_notation[r][c]) > 5:
                        print("**...**", end=" ")
                    else:
                        print(pad_with_char(''.join(str(x) for x in board_inst.cell_notation[r][c]), 7),
                              sep="", end=" ")
            print("|")
        print("-------------------------------")

if __name__ == "__main__":
    example_board = np.array([[0, 0, 0, 7, 0, 0, 0, 8, 0],
                              [0, 9, 0, 0, 0, 3, 1, 0, 0],
                              [0, 0, 6, 8, 0, 5, 0, 7, 0],
                              [0, 2, 0, 6, 0, 0, 0, 4, 9],
                              [0, 0, 0, 2, 0, 0, 0, 5, 0],
                              [0, 0, 8, 0, 4, 0, 0, 0, 7],
                              [0, 0, 0, 9, 0, 0, 0, 3, 0],
                              [3, 7, 0, 0, 0, 0, 0, 0, 6],
                              [1, 0, 5, 0, 0, 4, 0, 0, 0]])

    example_board = np.array([[9, 0, 3, 0, 0, 0, 0, 0, 2],
                               [0, 6, 0, 4, 9, 0, 1, 0, 3],
                               [0, 0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 9, 0, 0],
                               [3, 0, 1, 0, 0, 4, 0, 0, 0],
                               [0, 8, 0, 7, 0, 2, 4, 3, 0],
                               [1, 7, 8, 5, 0, 9, 3, 0, 0],
                               [0, 0, 0, 0, 0, 0, 7, 5, 9],
                               [0, 0, 0, 3, 6, 7, 0, 0, 0]])

    example_board = np.array([[0, 0, 0, 0, 0, 0, 3, 2, 0],
                              [3, 2, 0, 6, 5, 0, 0, 1, 0],
                              [0, 0, 0, 0, 7, 0, 0, 0, 9],
                              [0, 6, 3, 0, 0, 0, 0, 0, 0],
                              [0, 4, 5, 1, 6, 7, 9, 8, 0],
                              [0, 0, 0, 0, 0, 0, 2, 5, 0],
                              [4, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 7, 0, 0, 3, 8, 0, 9, 4],
                              [0, 8, 6, 0, 0, 0, 0, 0, 0]]) # https://www.youtube.com/watch?v=Mh8-MICdO6s&ab_channel=LearnSomething

    example_board5 = np.array([[6, 0, 0, 0, 0, 0, 3, 2, 0],
                              [3, 2, 0, 6, 5, 0, 0, 1, 0],
                              [0, 0, 0, 0, 7, 0, 0, 6, 9],
                              [0, 6, 3, 0, 0, 0, 0, 4, 0],
                              [2, 4, 5, 1, 6, 7, 9, 8, 3],
                              [7, 0, 0, 0, 0, 0, 2, 5, 6],
                              [4, 3, 0, 0, 1, 6, 0, 7, 0],
                              [0, 7, 0, 0, 3, 8, 6, 9, 4],
                              [0, 8, 6, 7, 0, 0, 0, 3, 0]])  # https://www.youtube.com/watch?v=Mh8-MICdO6s&ab_channel=LearnSomething - minute 7:21

    example_board = np.array([[0, 0, 3, 8, 0, 0, 5, 1, 0],
                               [0, 0, 8, 7, 0, 0, 9, 3, 0],
                               [1, 0, 0, 3, 0, 5, 7, 2, 8],
                               [0, 0, 0, 2, 0, 0, 8, 4, 9],
                               [8, 0, 1, 9, 0, 6, 2, 5, 7],
                               [0, 0, 0, 5, 0, 0, 1, 6, 3],
                               [9, 6, 4, 1, 2, 7, 3, 8, 5],
                               [3, 8, 2, 6, 5, 9, 4, 7, 1],
                               [0, 1, 0, 4, 0, 0, 6, 9, 2]])    # a needed x-wing technique example

    example_board_inst = Board(example_board)

    cli_print_board(example_board_inst)

    # Continue as long as there are zeros in the board. i.e the board is not solved
    while np.any(example_board_inst.board == 0):
        success = next_step_sudoku_human_solver(example_board_inst)
        if not success:
            print("Failed to solve the Sudoku board:")
            cli_print_board(example_board_inst, print_cell_notation=True)
            break
        else:
            print("Next step found. current board:")
            cli_print_board(example_board_inst, print_cell_notation=True)

    if not np.any(example_board_inst.board == 0):
        print("Board solved successfully")

