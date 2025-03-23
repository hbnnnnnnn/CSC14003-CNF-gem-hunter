import time
import signal
import threading
import os
from itertools import combinations, product
from pysat.solvers import Solver
from pysat.formula import CNF

def read_matrix(file_name):
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        return
    with open(file_name, 'r') as f:
        matrix = []
        for line in f:
            row = []
            for num in line.replace(',', '').split():
                if num.isdigit(): 
                    row.append(int(num))
                elif num == '_': 
                    row.append('_')
                else:
                    raise ValueError(f"Invalid character in file: {num}")
            matrix.append(row)
        return matrix

def write_matrix(file_name, matrix):
    with open(file_name, 'w') as f:
        for row in matrix:
            f.write(','.join(map(str, row)) + '\n')

def coordinate_to_literal(i, j, m):
    return i * m + j + 1

def literal_to_coordinate(literal, m):
    i = (abs(literal) - 1) // m
    j = (abs(literal) - 1) % m
    return i, j

def cell_neighbors(i, j, matrix):
    n = len(matrix)
    m = len(matrix[0])
    neighbors = []
    for x in range(i-1, i+2):
        for y in range(j-1, j+2):
            if x >= 0 and x < n and y >= 0 and y < m and (x != i or y != j):
                if matrix[x][y] == '_':
                    neighbors.append(coordinate_to_literal(x, y, m))
    return neighbors



def at_most_k(neighbors, k):
    return [list(-x for x in combo) for combo in combinations(neighbors, k + 1)]

def at_least_k(neighbors, k):
    return [list(combo) for combo in combinations(neighbors, len(neighbors)-k+1)]

def exactly_k(neighbors, k):
    # print(at_least_k(neighbors, k))
    # print(at_most_k(neighbors, k) )
    return at_least_k(neighbors, k) + at_most_k(neighbors, k)

def remove_duplicates_keep_order(cnf):
    seen = set()
    unique_cnf = []
    for clause in cnf:
        sorted_clause = tuple(sorted(clause))
        if sorted_clause not in seen:
            seen.add(sorted_clause)
            unique_cnf.append(clause)
    return unique_cnf

def generate_cnf_from_matrix (matrix):
    n = len(matrix)
    m = len(matrix[0])
    cnf = CNF()
    clauses = []
    
    for i in range(0,n):
        for j in range(0,m):
            if (matrix[i][j] != '_'):
                neighbors = cell_neighbors(i,j,matrix)
                k = matrix[i][j]
                clauses += (exactly_k(neighbors, k))

    clauses = remove_duplicates_keep_order(clauses)
    for clause in clauses:
        cnf.append(clause)
    return cnf


def interpret_model(model, matrix):
    n = len(matrix)
    m = len(matrix[0])
    result = [[cell for cell in row] for row in matrix]

    for i in range(n):
        for j in range(m):
            if matrix[i][j] == '_':
                literal = coordinate_to_literal(i, j, m)
                if literal in model: 
                    result[i][j] = 'T'
                else:  
                    result[i][j] = 'G'

    return result
    
def solve_cnf_backtrack(matrix):
    n = len(matrix)
    m = len(matrix[0])
    variables = [coordinate_to_literal(i, j, m) for i in range(n) for j in range(m) if matrix[i][j] == '_']
    cnf = generate_cnf_from_matrix(matrix)

    def backtrack(assignment, index):
        if index == len(variables):
            # Check if the current assignment satisfies the CNF
            for clause in cnf.clauses:
                if not any((literal > 0 and assignment.get(abs(literal), False)) or
                           (literal < 0 and not assignment.get(abs(literal), False)) for literal in clause):
                    return False
            return True

        # Try assigning True to the current variable
        assignment[variables[index]] = True
        if backtrack(assignment, index + 1):
            return True

        # Try assigning False to the current variable
        assignment[variables[index]] = False
        if backtrack(assignment, index + 1):
            return True

        # Backtrack
        del assignment[variables[index]]
        return False

    assignment = {}
    if backtrack(assignment, 0):
        print("SAT (Backtracking)")
        model = [var if assignment[var] else -var for var in variables]
        print("Model:", model)
        result = interpret_model(model, matrix)
        print("Result:", result)
        write_matrix("output_backtrack.txt", result)
    else:
        print("UNSAT (Backtracking)")

def solve_cnf_brute_force(matrix):

    n = len(matrix)
    m = len(matrix[0])
    variables = [coordinate_to_literal(i, j, m) for i in range(n) for j in range(m) if matrix[i][j] == '_']
    cnf = generate_cnf_from_matrix(matrix)

    for assignment in product([True, False], repeat=len(variables)):
        model = {var: val for var, val in zip(variables, assignment)}
        satisfied = True

        for clause in cnf.clauses:
            clause_satisfied = any((literal > 0 and model.get(abs(literal), False)) or 
                                   (literal < 0 and not model.get(abs(literal), False)) for literal in clause)
            if not clause_satisfied:
                satisfied = False
                break

        if satisfied:
            print("SAT (Brute Force)")
            print("Model:", [var if model[var] else -var for var in variables])
            result = interpret_model([var if model[var] else -var for var in variables], matrix)
            print("Result:", result)
            write_matrix("output_brute_force.txt", result)
            return

    print("UNSAT (Brute Force)")

def solve_cnf_brute_force_optimized(matrix):
    n = len(matrix)
    m = len(matrix[0])
    variables = [coordinate_to_literal(i, j, m) for i in range(n) for j in range(m) if matrix[i][j] == '_']
    cnf = generate_cnf_from_matrix(matrix)
    
    # Pre-process clauses to group by variables for faster checking
    var_to_clauses = {}
    for i, clause in enumerate(cnf.clauses):
        for literal in clause:
            var = abs(literal)
            if var not in var_to_clauses:
                var_to_clauses[var] = []
            var_to_clauses[var].append((i, literal > 0))
    
    # Convert clauses to sets for faster lookup
    clause_status = [False] * len(cnf.clauses)
    
    # Use numpy for faster array operations if available
    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False
    
    # Use bit manipulation for assignments if the number of variables is small enough
    if len(variables) <= 63:  # 64-bit integer limit
        max_combinations = 2 ** len(variables)
        for i in range(max_combinations):
            # Reset clause status
            for j in range(len(clause_status)):
                clause_status[j] = False
                
            # Create assignment from bits
            model = {}
            all_satisfied = True
            
            for var_idx, var in enumerate(variables):
                val = (i >> var_idx) & 1 == 1
                model[var] = val
                
                # Check only affected clauses
                if var in var_to_clauses:
                    for clause_idx, is_positive in var_to_clauses[var]:
                        if (is_positive and val) or (not is_positive and not val):
                            clause_status[clause_idx] = True
            
            # Check if all clauses are satisfied
            if all(clause_status):
                print("SAT (Brute Force Optimized)")
                result = interpret_model([var if model[var] else -var for var in variables], matrix)
                write_matrix("output_brute_force.txt", result)
                return
    else:
        # Fall back to itertools for larger variable sets
        for assignment in product([True, False], repeat=len(variables)):
            # Reset clause status
            clause_status = [False] * len(cnf.clauses)
            
            # Create assignment
            model = {var: val for var, val in zip(variables, assignment)}
            
            # Check clauses
            for var, val in model.items():
                if var in var_to_clauses:
                    for clause_idx, is_positive in var_to_clauses[var]:
                        if (is_positive and val) or (not is_positive and not val):
                            clause_status[clause_idx] = True
            
            # Check if all clauses are satisfied
            if all(clause_status):
                print("SAT (Brute Force Optimized)")
                result = interpret_model([var if model[var] else -var for var in variables], matrix)
                write_matrix("output_brute_force.txt", result)
                return

    print("UNSAT (Brute Force Optimized)")


# Timeout handling for Windows (using threading instead of signal)
class TimeoutError(Exception):
    pass

def run_with_timeout(func, args=(), timeout=60):
    result = [None]
    exception = [None]
    is_timeout = [False]
    
    def worker():
        try:
            result[0] = func(*args)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    
    start_time = time.time()
    thread.start()
    thread.join(timeout)
    
    execution_time = time.time() - start_time
    
    if thread.is_alive():
        is_timeout[0] = True
        execution_time = timeout
        return None, execution_time, True
    
    if exception[0] is not None:
        raise exception[0]
    
    return result[0], execution_time, False

# PySAT solver with timeout
def solve_cnf_pysat(matrix, timeout=60):
    start_time = time.time()
    solver = Solver(name='glucose4')  # Use glucose4 for better performance
    cnf = generate_cnf_from_matrix(matrix)
    solver.append_formula(cnf)
    
    try:
        if solver.solve():
            model = solver.get_model()
            result = interpret_model(model, matrix)
            print(f"SAT (PySAT) - {time.time() - start_time:.4f}s")
            write_matrix("output_pysat.txt", result)
            solver.delete()
            return True
        else:
            print(f"UNSAT (PySAT) - {time.time() - start_time:.4f}s")
            solver.delete()
            return False
    except Exception as e:
        print(f"PySAT error: {e} - {time.time() - start_time:.4f}s")
        solver.delete()
        return False

# Further optimized backtracking for large puzzles
def solve_cnf_backtrack_advanced(matrix, timeout=60):
    # Your existing implementation...
    # (Remove any signal.alarm calls inside this function)
    
    # Add a time check inside the backtrack function
    start_time = time.time()
    
    def backtrack(assignment):
        # Check for timeout periodically
        if time.time() - start_time > timeout:
            return False
            
        # Rest of your backtracking implementation...
        pass
    
    # Rest of the function...
    pass

# Comprehensive benchmarking function
def benchmark_solvers(test_cases, timeout=60):
    results = {}
    
    for test_name, file_path in test_cases.items():
        print(f"\n=== Benchmarking {test_name} ===")
        matrix = read_matrix(file_path)
        n = len(matrix)
        m = len(matrix[0])
        empty_cells = sum(1 for i in range(n) for j in range(m) if matrix[i][j] == '_')
        
        print(f"Puzzle size: {n}x{m} with {empty_cells} empty cells")
        results[test_name] = {
            'size': f"{n}x{m}",
            'empty_cells': empty_cells,
            'results': {}
        }
        
        # Always run PySAT (should be fast for all puzzles)
        print("\nRunning PySAT solver...")
        try:
            _, pysat_time, pysat_timeout = run_with_timeout(solve_cnf_pysat, (matrix, timeout), timeout)
            results[test_name]['results']['pysat'] = {
                'time': pysat_time,
                'timeout': pysat_timeout
            }
        except Exception as e:
            print(f"Error running PySAT: {e}")
            results[test_name]['results']['pysat'] = {
                'time': None,
                'error': str(e)
            }
        
        # Run advanced backtracking for all but the largest puzzles
        if n * m <= 225:  # Up to 15x15
            print("\nRunning advanced backtracking...")
            try:
                _, backtrack_time, backtrack_timeout = run_with_timeout(
                    solve_cnf_backtrack_advanced, (matrix, timeout), timeout
                )
                results[test_name]['results']['backtrack_advanced'] = {
                    'time': backtrack_time,
                    'timeout': backtrack_timeout
                }
            except Exception as e:
                print(f"Error running advanced backtracking: {e}")
                results[test_name]['results']['backtrack_advanced'] = {
                    'time': None,
                    'error': str(e)
                }
        
        # Only run basic backtracking for small puzzles
        if n * m <= 100:  # Up to 10x10
            print("\nRunning basic backtracking...")
            try:
                _, basic_backtrack_time, basic_backtrack_timeout = run_with_timeout(
                    solve_cnf_backtrack, (matrix,), timeout
                )
                results[test_name]['results']['backtrack_basic'] = {
                    'time': basic_backtrack_time,
                    'timeout': basic_backtrack_timeout
                }
            except Exception as e:
                print(f"Error running basic backtracking: {e}")
                results[test_name]['results']['backtrack_basic'] = {
                    'time': None,
                    'error': str(e)
                }
        
        # Only run brute force for tiny puzzles
        if n * m <= 25:  # Up to 5x5
            print("\nRunning brute force...")
            try:
                _, brute_time, brute_timeout = run_with_timeout(
                    solve_cnf_brute_force, (matrix,), timeout
                )
                results[test_name]['results']['brute_force'] = {
                    'time': brute_time,
                    'timeout': brute_timeout
                }
            except Exception as e:
                print(f"Error running brute force: {e}")
                results[test_name]['results']['brute_force'] = {
                    'time': None,
                    'error': str(e)
                }
    
    # Print summary table
    print("\n=== Benchmark Summary ===")
    print(f"{'Test Case':<15} {'Size':<10} {'Empty':<6} {'PySAT':<15} {'Adv Backtrack':<15} {'Backtrack':<15} {'Brute Force':<15}")
    print("-" * 90)
    
    for test_name, data in results.items():
        size = data['size']
        empty = data['empty_cells']
        
        pysat = format_time(data['results'].get('pysat', {}).get('time'), 
                           data['results'].get('pysat', {}).get('timeout', False))
        
        adv_backtrack = format_time(data['results'].get('backtrack_advanced', {}).get('time'),
                                   data['results'].get('backtrack_advanced', {}).get('timeout', False))
        
        backtrack = format_time(data['results'].get('backtrack_basic', {}).get('time'),
                               data['results'].get('backtrack_basic', {}).get('timeout', False))
        
        brute = format_time(data['results'].get('brute_force', {}).get('time'),
                           data['results'].get('brute_force', {}).get('timeout', False))
        
        print(f"{test_name:<15} {size:<10} {empty:<6} {pysat:<15} {adv_backtrack:<15} {backtrack:<15} {brute:<15}")
    
    return results

def format_time(time_value, timed_out):
    if timed_out:
        return "TIMEOUT"
    elif time_value is None:
        return "ERROR"
    else:
        return f"{time_value:.4f}s"

# Simpler timing function for individual benchmarks
def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

# Main function with test cases
if __name__ == "__main__":
    test_cases = {
        "5x5 Puzzle": "testcases/input_1.txt",
        # "8x8 Puzzle": "testcases/8x8.txt",
        "11x11 Puzzle": "testcases/input_2.txt",
        # "20x20 Puzzle": "testcases/20x20.txt"  # Uncomment when ready to test
    }
    
    # Set timeout to 2 minutes for each algorithm
    results = benchmark_solvers(test_cases, timeout=120)
    
    # Alternative: Run and time individual functions without timeout mechanism
    print("\n=== Individual Function Timing ===")
    matrix_5x5 = read_matrix("testcases/input_2.txt")
    
    # Time PySAT
    _, pysat_time = time_function(solve_cnf_pysat, matrix_5x5)
    print(f"PySAT time for 5x5: {pysat_time:.4f}s")
    
    # Time backtracking
    _, backtrack_time = time_function(solve_cnf_backtrack, matrix_5x5)
    print(f"Backtracking time for 5x5: {backtrack_time:.4f}s")
    
    # Time brute force
    _, brute_time = time_function(solve_cnf_brute_force, matrix_5x5)
    print(f"Brute force time for 5x5: {brute_time:.4f}s")
    
    # Time optimized brute force
    _, opt_brute_time = time_function(solve_cnf_brute_force_optimized, matrix_5x5)
    print(f"Optimized brute force time for 5x5: {opt_brute_time:.4f}s")

