# CSC14003-CNF-gem-hunter
## About the Project

This project focuses on modeling logic-based puzzles using Conjunctive Normal Form (CNF) and solving them with a SAT solver. It includes multiple solving methods—such as brute-force, backtracking, and PySAT—for comparison. A simple game interface is provided to visualize and interact with the CNF-solving process.

## Features

- Converts logic constraints into CNF format automatically
- Supports three solving approaches: brute-force, backtracking, and SAT solver (PySAT)
- Measures and compares the performance of each solving method
- Includes a basic game interface to visualize the CNF problem and solutions
- Modular and easy to extend for different types of logic-based puzzles

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/hbnnnnnnn/CSC14003-CNF-gem-hunter
    ```
2. Navigate to the project directory:
    ```bash
    cd CSC14003-CNF-gem-hunter
    ```
3. Requirements:
    python 3.12.4
    pip (latest version recommended)
4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Run the CNF performance test:
    ```bash
    Run main.py by using command python .\source\main.py.
    ```
6. Launch the game:
    ```bash
    Run gem_hunter.py by using command python .\source\gem_hunter.py.
    ```
7. Project structure CSC14003-CNF-gem-hunter ...
CSC14003-CNF-gem-hunter/
├── assets/                 # Game assets 
|   ├──sounds/
|   ├──text/
├── source/                 # Source code
│   ├── gem_hunter.py       # Main game logic
│   ├── main.py             # CNF solver performance test
│   ├── generate_test_cases.py
│   └── testcases/          # CNF input/output test files
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignored files

