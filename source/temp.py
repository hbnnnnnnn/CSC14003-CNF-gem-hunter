import random
import numpy as np
import os

def create_test_case(size):
    """
    Create a random solvable test case of the given size.
    Returns both the puzzle (with numbers and empty cells) and the solution.
    """
    # Create a random trap layout (True for trap, False for gem)
    traps = np.random.choice([True, False], size=(size, size), p=[0.3, 0.7])
    
    # Calculate the number of adjacent traps for each cell
    numbers = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if traps[i, j]:
                continue  # Skip trap cells
            
            # Count adjacent traps
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
    
    # Create the puzzle (hide some number cells)
    puzzle = np.full((size, size), '_', dtype=object)
    for i in range(size):
        for j in range(size):
            if numbers[i, j] > 0:
                # Keep about 60% of number cells visible
                if random.random() < 0.6:
                    puzzle[i, j] = str(numbers[i, j])
    
    # Create the solution
    solution = np.full((size, size), '_', dtype=object)
    for i in range(size):
        for j in range(size):
            if traps[i, j]:
                solution[i, j] = 'T'
            else:
                solution[i, j] = 'G'
            if numbers[i, j] > 0 and puzzle[i, j] != '_':
                solution[i, j] = puzzle[i, j]
    
    # Simple heuristic to check if the puzzle has enough clues
    clue_count = np.sum(puzzle != '_')
    is_solvable = clue_count >= size  # At least one clue per row on average
    
    return puzzle, solution, is_solvable, clue_count

def save_test_case(puzzle, solution, puzzle_file, solution_file):
    """
    Save the puzzle and solution to files.
    """
    with open(puzzle_file, 'w') as f:
        for row in puzzle:
            f.write(', '.join(row) + '\n')
    
    with open(solution_file, 'w') as f:
        for row in solution:
            f.write(', '.join(row) + '\n')

# Create directory for test cases if it doesn't exist
os.makedirs('/home/user/testcases', exist_ok=True)

# Create test cases
sizes = [5, 11, 20]
for i, size in enumerate(sizes, 1):
    best_puzzle = None
    best_solution = None
    best_clue_count = 0
    
    # Try multiple times to get a puzzle with a good number of clues
    for attempt in range(20):
        puzzle, solution, is_solvable, clue_count = create_test_case(size)
        
        # Keep the puzzle with the most clues
        if is_solvable and clue_count > best_clue_count:
            best_puzzle = puzzle
            best_solution = solution
            best_clue_count = clue_count
    
    if best_puzzle is not None:
        print(f"Created {size}x{size} test case with {best_clue_count} clues")
        
        # Display a sample of the test case (first few rows)
        print("\nPuzzle (first 5 rows):")
        for row in best_puzzle[:min(5, size)]:
            print(', '.join(row))
        
        print("\nSolution (first 5 rows):")
        for row in best_solution[:min(5, size)]:
            print(', '.join(row))
        
        # Save to files
        puzzle_file = f"testcases\input_{i}.txt"
        solution_file = f"testcases\output_{i}.txt"
        save_test_case(best_puzzle, best_solution, puzzle_file, solution_file)
        print(f"Saved to {puzzle_file} and {solution_file}")
    else:
        print(f"Failed to create a good {size}x{size} test case")