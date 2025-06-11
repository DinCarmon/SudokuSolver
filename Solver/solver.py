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

import numpy as np

def solve_sudoku(board: np.ndarray) -> bool:
    # print("start solving sudoku at: " + str(datetime.now()))
    def is_valid(r, c, n):
        for i in range(9):
            if board[r][i] == n or board[i][c] == n:
                return False
        start_row, start_col = 3 * (r // 3), 3 * (c // 3)
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == n:
                    return False
        return True

    # Preliminary check to ensure all filled cells are valid
    for r in range(9):
        for c in range(9):
            n = board[r][c]
            if n == 0:
                continue
            board[r][c] = 0  # Temporarily clear the cell
            if not is_valid(r, c, n):
                board[r][c] = n  # Restore original value
                return False
            board[r][c] = n  # Restore original value

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
