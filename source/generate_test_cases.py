import random
import numpy as np
import os

def create_test_case(size):
    """
    Create a random solvable test case of the given size.
    Returns both the puzzle (with numbers and empty cells) and the solution.
    """
    # Create a random trap layout (True for trap, False for gem)
    # Reduce trap probability from 0.3 to 0.2 to create more gem cells that can have numbers
    traps = np.random.choice([True, False], size=(size, size), p=[0.2, 0.8])
    
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
    
    # Create the puzzle (show more number cells)
    puzzle = np.full((size, size), '_', dtype=object)
    for i in range(size):
        for j in range(size):
            if numbers[i, j] > 0:
                # Increase visibility from 60% to 85% of number cells
                if random.random() < 0.85:
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
    
    # Count the number of clues and cells with numbers
    clue_count = np.sum(puzzle != '_')
    number_cells_count = np.sum(numbers > 0)
    
    # Calculate percentage of number cells that are visible
    visibility_percentage = (clue_count / number_cells_count * 100) if number_cells_count > 0 else 0
    
    # Simple heuristic to check if the puzzle has enough clues
    is_solvable = clue_count >= size  # At least one clue per row on average
    
    return puzzle, solution, is_solvable, clue_count, number_cells_count, visibility_percentage

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
os.makedirs('source/testcases', exist_ok=True)

# Create test cases
sizes = [5, 6, 7, 8, 9, 10, 11, 20]
for i, size in enumerate(sizes, 1):
    best_puzzle = None
    best_solution = None
    best_clue_count = 0
    best_stats = None
    
    # Try multiple times to get a puzzle with a good number of clues
    for attempt in range(30):  # Increased from 20 to 30 attempts
        puzzle, solution, is_solvable, clue_count, number_cells_count, visibility_percentage = create_test_case(size)
        
        # Keep the puzzle with the most clues
        if is_solvable and clue_count > best_clue_count:
            best_puzzle = puzzle
            best_solution = solution
            best_clue_count = clue_count
            best_stats = (number_cells_count, visibility_percentage)
    
    if best_puzzle is not None:
        number_cells, visibility = best_stats
        print(f"Created {size}x{size} test case with {best_clue_count} clues")
        print(f"Number cells: {number_cells}, Visibility: {visibility:.1f}%")
        
        # Display a sample of the test case (first few rows)
        print("\nPuzzle (first 5 rows):")
        for row in best_puzzle[:min(5, size)]:
            print(', '.join(row))
        
        print("\nSolution (first 5 rows):")
        for row in best_solution[:min(5, size)]:
            print(', '.join(row))
        
        # Save to files
        puzzle_file = f"source/testcases/input_{i}.txt"
        solution_file = f"source/testcases/output_{i}.txt"
        save_test_case(best_puzzle, best_solution, puzzle_file, solution_file)
        print(f"Saved to {puzzle_file} and {solution_file}")
    else:
        print(f"Failed to create a good {size}x{size} test case")
