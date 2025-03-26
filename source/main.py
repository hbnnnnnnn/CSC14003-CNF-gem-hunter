import os
import time
import concurrent.futures
from itertools import combinations
from itertools import product
from pysat.solvers import Solver
import gc

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

def write_matrix(file_name, matrix, append=False, header=None):
    mode = 'a' if append else 'w'
    with open(file_name, mode) as f:
        if header and append:
            f.write('\n\n' + header + '\n')
        elif header:
            f.write(header + '\n')
            
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
    cnf = []
    
    for i in range(0,n):
        for j in range(0,m):
            if (matrix[i][j] != '_'):
                neighbors = cell_neighbors(i,j,matrix)
                k = matrix[i][j]
                cnf += (exactly_k(neighbors, k))

    cnf = remove_duplicates_keep_order(cnf)
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

def solve_cnf_pysat(cnf, variables):
    solver = Solver()
    solver.append_formula(cnf)
    try:
        if solver.solve():
            print("SAT")
            model = solver.get_model()
            return model
        else:
            print("UNSAT")
            return None
    finally:
        solver.delete()

def prepare_cnf_data(matrix):
    n = len(matrix)
    m = len(matrix[0])
    variables = [coordinate_to_literal(i, j, m) for i in range(n) for j in range(m) if matrix[i][j] == '_']
    cnf = generate_cnf_from_matrix(matrix)
    return cnf, variables

def solve_cnf_brute_force(cnf, variables):
    clauses_by_var = {}
    for i, clause in enumerate(cnf):
        for literal in clause:
            var = abs(literal)
            if var not in clauses_by_var:
                clauses_by_var[var] = []
            clauses_by_var[var].append((i, literal))
    
    variables_list = sorted(list(variables))
    
    if len(variables) <= 30:
        max_combinations = 2 ** len(variables_list)
        
        for i in range(max_combinations):
            assignment = {}
            for var_idx, var in enumerate(variables_list):
                assignment[var] = ((i >> var_idx) & 1) == 1
            
            all_satisfied = True
            
            # Iterate through clauses correctly
            for i, clause in enumerate(cnf):
                clause_satisfied = False
                for literal in clause:
                    var = abs(literal)
                    val = assignment[var]
                    if (literal > 0 and val) or (literal < 0 and not val):
                        clause_satisfied = True
                        break
                
                if not clause_satisfied:
                    all_satisfied = False
                    break
            
            if all_satisfied:
                print("SAT (Brute Force Optimized)")
                model = [var if assignment[var] else -var for var in variables_list]
                return model
    else:
        
        for values in product([True, False], repeat=len(variables_list)):
            assignment = {var: val for var, val in zip(variables_list, values)}
            
            all_satisfied = True
            
            for clause in cnf:
                clause_satisfied = False
                for literal in clause:
                    var = abs(literal)
                    val = assignment[var]
                    if (literal > 0 and val) or (literal < 0 and not val):
                        clause_satisfied = True
                        break
                
                if not clause_satisfied:
                    all_satisfied = False
                    break
            
            if all_satisfied:
                print("SAT (Brute Force)")
                model = [var if assignment[var] else -var for var in variables_list]
                return model
    
    print("UNSAT (Brute Force)")
    return None


def solve_cnf_backtracking(cnf, variables):
    cnf = [clause[:] for clause in cnf]
    variables_list = sorted(list(variables))
    
    assignment = {}
    
    def dpll(formula, assign):
        if len(formula) == 0:
            return assign
        
        if any(len(clause) == 0 for clause in formula):
            return None
        
        unit_clauses = [clause for clause in formula if len(clause) == 1]
        if unit_clauses:
            unit = unit_clauses[0][0]
            var = abs(unit)
            val = unit > 0
            
            new_assign = assign.copy()
            new_assign[var] = val
            
            new_formula = []
            for clause in formula:
                if unit in clause:
                    continue
                new_clause = [lit for lit in clause if lit != -unit]
                new_formula.append(new_clause)
            
            return dpll(new_formula, new_assign)
        
        all_literals = [lit for clause in formula for lit in clause]
        pure_literals = []
        
        for var in variables_list:
            if var in all_literals and -var not in all_literals:
                pure_literals.append(var)
            elif -var in all_literals and var not in all_literals:
                pure_literals.append(-var)
        
        if pure_literals:
            pure = pure_literals[0]
            var = abs(pure)
            val = pure > 0
            
            new_assign = assign.copy()
            new_assign[var] = val
            
            new_formula = [clause for clause in formula if pure not in clause]
            
            return dpll(new_formula, new_assign)
        
        for var in variables_list:
            if var not in assign:
                new_assign = assign.copy()
                new_assign[var] = True
                
                new_formula = []
                for clause in formula:
                    if var in clause:
                        continue
                    new_clause = [lit for lit in clause if lit != -var]
                    new_formula.append(new_clause)
                
                result = dpll(new_formula, new_assign)
                if result is not None:
                    return result
                
                new_assign = assign.copy()
                new_assign[var] = False
                
                new_formula = []
                for clause in formula:
                    if -var in clause:
                        continue
                    new_clause = [lit for lit in clause if lit != var]
                    new_formula.append(new_clause)
                
                return dpll(new_formula, new_assign)
        
        return None
    
    final_assignment = dpll(cnf, assignment)
    
    if final_assignment is not None:
        print("SAT (DPLL)")
        model = [var if final_assignment.get(var, False) else -var for var in variables_list]
        return model
    else:
        print("UNSAT (DPLL)")
        return None


def measure_time(time_out, func, *args, **kwargs):
    start_time = time.perf_counter() 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            model = future.result(timeout=time_out)
            elapsed_time = time.perf_counter() - start_time 
            return model, elapsed_time
        except concurrent.futures.TimeoutError:
            print(f"Function {func.__name__} timed out after {time_out} seconds")
            future.cancel()
            executor._threads.clear()
            concurrent.futures.thread._threads_queues.clear()
            return None, time_out
        
if __name__ == "__main__":
    try:
        for i in range(1, 5):
            print(f"=== Test case {i} ===")
            file_path = f'source/testcases/input_{i}.txt'
            matrix = read_matrix(file_path)
            cnf, variables = prepare_cnf_data(matrix)
            
            print("\n=== Timing PySAT Solver ===")
            model, pysat_time = measure_time(300, solve_cnf_pysat, cnf, variables)
            if model:
                result = interpret_model(model, matrix)
                write_matrix(f'source/testcases/output_{i}.txt', result, False, "PySAT")
            
            print("\n=== Timing Backtracking ===")
            model, backtrack_opt_time = measure_time(300, solve_cnf_backtracking, cnf, variables)
            if model:
                result = interpret_model(model, matrix)
                write_matrix(f'source/testcases/output_{i}.txt', result, True, "Backtracking")
            
            print("\n=== Timing Brute Force ===")
            model, brute_opt_time = measure_time(300, solve_cnf_brute_force, cnf, variables)
            if model:
                result = interpret_model(model, matrix)
                write_matrix(f'source/testcases/output_{i}.txt', result, True, "Brute Force")
            
            print("\n=== Performance Summary ===")
            print(f"PySAT Solver:          {pysat_time:.4f} seconds")
            print(f"Backtracking: {backtrack_opt_time:.4f} seconds")
            print(f"Brute Force: {brute_opt_time:.4f} seconds")

            gc.collect()
        os._exit(0)
    except Exception as e:
        print(f"Error occurred: {e}")
        os._exit(1)
