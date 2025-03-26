import random
import numpy as np
import os

def create_test_case(size):
    traps = np.random.choice([True, False], size=(size, size), p=[0.2, 0.8])
    numbers = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if traps[i, j]:
                continue
            count = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < size and 0 <= nj < size and traps[ni, nj]:
                        count += 1
            if count > 0:
                numbers[i, j] = count
    puzzle = np.full((size, size), '_', dtype=object)
    for i in range(size):
        for j in range(size):
            if numbers[i, j] > 0:
                if random.random() < 0.85:
                    puzzle[i, j] = str(numbers[i, j])
    solution = np.full((size, size), '_', dtype=object)
    for i in range(size):
        for j in range(size):
            if traps[i, j]:
                solution[i, j] = 'T'
            else:
                solution[i, j] = 'G'
            if numbers[i, j] > 0 and puzzle[i, j] != '_':
                solution[i, j] = puzzle[i, j]
    clue_count = np.sum(puzzle != '_')
    number_cells_count = np.sum(numbers > 0)
    visibility_percentage = (clue_count / number_cells_count * 100) if number_cells_count > 0 else 0
    is_solvable = clue_count >= size
    return puzzle, solution, is_solvable, clue_count, number_cells_count, visibility_percentage

def save_test_case(puzzle, solution, puzzle_file, solution_file):
    with open(puzzle_file, 'w') as f:
        for row in puzzle:
            f.write(', '.join(row) + '\n')
    with open(solution_file, 'w') as f:
        for row in solution:
            f.write(', '.join(row) + '\n')

os.makedirs('source/testcases', exist_ok=True)

sizes = [5, 8, 11, 20]
for i, size in enumerate(sizes, 1):
    best_puzzle = None
    best_solution = None
    best_clue_count = 0
    best_stats = None
    for attempt in range(30): 
        puzzle, solution, is_solvable, clue_count, number_cells_count, visibility_percentage = create_test_case(size)
        if is_solvable and clue_count > best_clue_count:
            best_puzzle = puzzle
            best_solution = solution
            best_clue_count = clue_count
            best_stats = (number_cells_count, visibility_percentage)
    if best_puzzle is not None:
        number_cells, visibility = best_stats
        print(f"Created {size}x{size} test case with {best_clue_count} clues")
        print(f"Number cells: {number_cells}, Visibility: {visibility:.1f}%")
        print("\nPuzzle (first 5 rows):")
        for row in best_puzzle[:min(5, size)]:
            print(', '.join(row))
        print("\nSolution (first 5 rows):")
        for row in best_solution[:min(5, size)]:
            print(', '.join(row))
        puzzle_file = f"source/testcases/input_{i}.txt"
        solution_file = f"source/testcases/output_{i}.txt"
        save_test_case(best_puzzle, best_solution, puzzle_file, solution_file)
        print(f"Saved to {puzzle_file} and {solution_file}")
    else:
        print(f"Failed to create a good {size}x{size} test case")
