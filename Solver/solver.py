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
from datetime import datetime
from enum import Enum

import numpy as np
from numpy.ma.extras import row_stack

from Loader.SudokuImageExtractor.digit_recognition.digit_classification_model_training_and_using import \
    add_no_digit_images

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

def safe_replace(board, digit, row, col):
    if board[row][col] != 0:
        raise RuntimeError(f"Attempting to replace a digit")
    board[row][col] = digit

def technique_1(board: np.ndarray, metadata_on_board_inst) -> bool:
    """
    The technique eliminates places for a digit based on the digit already placed in the line / column.
    If a digit is found that for it only one place in a given block is avaialbe it updates the board accordingly,
    and returns True.
    :param metadata_on_board_inst: 
    :param board:
    :return:
    """
    for digit in range(1, 10):
        for block_row in range(3):
            for block_col in range(3):
                if is_digit_in_block(board, digit, block_row, block_col):
                    continue
                optional_relative_places_for_digit = np.array([[0, 0], [0, 1], [0, 2],
                                                               [1, 0], [1, 1], [1, 2],
                                                               [2, 0], [2, 1], [2, 2]])
                for optional_relative_place in optional_relative_places_for_digit:
                    if board[3 * block_row + optional_relative_place[0]][3 * block_col + optional_relative_place[1]] != 0:
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)
                    if is_digit_in_line(board, digit, 3 * block_row + optional_relative_place[0]):
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)
                    if is_digit_in_col(board, digit, 3 * block_col + optional_relative_place[1]):
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)

                    for metadata in metadata_on_board_inst:
                        if metadata[1] == digit:
                            if metadata[0] == MetaDataType.LINE_OF_DIGIT and \
                                metadata[2] == block_row and \
                                metadata[4] == optional_relative_place[0]:
                                optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)

                            if metadata[0] == MetaDataType.COL_OF_DIGIT and \
                                metadata[3] == block_col and \
                                metadata[4] == optional_relative_place[1]:
                                optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit, optional_relative_place)

                if optional_relative_places_for_digit.shape == (1,2):
                    safe_replace(board,
                                 digit,
                                 row = 3 * block_row + optional_relative_places_for_digit[0][0],
                                 col = 3 * block_col + optional_relative_places_for_digit[0][1])
                    return True
    return False

def technique_2(board: np.ndarray, metadata_on_board_inst) -> bool:
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
                if is_digit_in_block(board, digit, block_row, block_col):
                    continue
                optional_relative_places_for_digit = np.array([[0, 0], [0, 1], [0, 2],
                                                               [1, 0], [1, 1], [1, 2],
                                                               [2, 0], [2, 1], [2, 2]])

                for optional_relative_place in optional_relative_places_for_digit:
                    if board[3 * block_row + optional_relative_place[0]][3 * block_col + optional_relative_place[1]] != 0:
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit,
                                                                                optional_relative_place)
                        continue
                    if is_digit_in_line(board, digit, 3 * block_row + optional_relative_place[0]):
                        optional_relative_places_for_digit = remove_first_match(optional_relative_places_for_digit,
                                                                                optional_relative_place)
                        continue
                    if is_digit_in_col(board, digit, 3 * block_col + optional_relative_place[1]):
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
                    if new_metadata not in metadata_on_board_inst:
                        metadata_on_board_inst.append(new_metadata)
                        print("new metadata: ", new_metadata)
                        return True

                is_same_optional_col_for_digit = True
                optional_relative_col_for_digit = optional_relative_places_for_digit[0][1]
                for optional_relative_place in optional_relative_places_for_digit:
                    if optional_relative_place[1] != optional_relative_col_for_digit:
                        is_same_optional_col_for_digit = False
                if is_same_optional_col_for_digit:
                    new_metadata = [MetaDataType.COL_OF_DIGIT, digit, block_row, block_col, optional_relative_col_for_digit]
                    if new_metadata not in metadata_on_board_inst:
                        metadata_on_board_inst.append(new_metadata)
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

def technique_3(board: np.ndarray, metadata_on_board_inst) -> bool:
    """
    The technique procedure:
    Look at a specific line / column. look at the missiing numbers.
    If only one number is possible, because all other numbers are already in the column, we found it!
    :param board:
    :param metadata_on_board_inst:
    :return:
    """
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                optional_numbers_for_slot = get_missing_numbers_in_row(board, row)
                for missing_number in optional_numbers_for_slot:
                    if np.any(board[:, col] == missing_number):
                        optional_numbers_for_slot.remove(missing_number)
                if len(optional_numbers_for_slot) == 1:
                    safe_replace(board,
                                 optional_numbers_for_slot[0],
                                 row=row,
                                 col=col)
                    return True

    for col in range(9):
        for row in range(9):
            if board[row][col] == 0:
                optional_numbers_for_slot = get_missing_numbers_in_col(board, col)
                for missing_number in optional_numbers_for_slot:
                    if np.any(board[row, :] == missing_number):
                        optional_numbers_for_slot.remove(missing_number)
                if len(optional_numbers_for_slot) == 1:
                    safe_replace(board,
                                 optional_numbers_for_slot[0],
                                 row=row,
                                 col=col)
                    return True

    return False

def next_step_sudoku_human_solver(board: np.ndarray, metadata_on_board_inst) -> bool:
    technique1_success = technique_1(board, metadata_on_board_inst)
    if technique1_success:
        return True

    technique2_success = technique_2(board, metadata_on_board_inst)
    if technique2_success:
        return True

    technique3_success = technique_3(board, metadata_on_board_inst)
    if technique3_success:
        return True

    return False

def cli_print_board(board: np.ndarray) -> None:
    for r in range(9):
        if r % 3 == 0:
            print("-------------------------------")
        for c in range(9):
            if c % 3 == 0:
                print("|", end="")
            print(" ", end="")
            if board[r][c] != 0:
                print(str(board[r][c]), end=" ")
            else:
                print(" ", end=" ")
        print("|")
    print("-------------------------------")

if __name__ == "__main__":
    example_board2 = np.array([[0, 0, 0, 7, 0, 0, 0, 8, 0],
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

    metadata_on_board = []

    cli_print_board(example_board)

    # Continue as long as there are zeros in the board. i.e the board is not solved
    while np.any(example_board == 0):
        success = next_step_sudoku_human_solver(example_board, metadata_on_board)
        if not success:
            print("Failed to solve the Sudoku board")
            break
        else:
            print("Next step found. current board:")
            cli_print_board(example_board)

    if not np.any(example_board == 0):
        print("Board solved successfully")

