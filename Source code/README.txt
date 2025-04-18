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
    git clone https://github.com/hbnnnnnnn/CSC14003-CNF-gem-hunter
2. Navigate to the project directory:
    cd CSC14003-CNF-gem-hunter
3. Requirements:
    python 3.12.4
    pip (latest version recommended)
4. Install dependencies:
    pip install -r "Source code/requirements.txt"
5. Run the CNF performance test:
    Run main.py by using command python "./Source code/main.py"
6. Launch the game:
    Run gem_hunter.py by using command python "./Source code/gem_hunter.py"

## Project Structure

23127300/
│
├── Source code/                # Root source directory
│   ├── assets/                 # Game assets
│   │   ├── sounds/             # Sound effects used in the game
│   │   └── text/               # Text-based resources 
│   │
│   ├── testcases/              # Folder containing test cases
│   │   ├── input/              # Input files for CNF solving
│   │   └── output/             # Corresponding output files
│   │
│   ├── gem_hunter.py           # Main game logic 
│   ├── main.py                 # CNF algorithms and solver performance evaluation 
│   ├── generate_test_cases.py  # Script for generating CNF test cases
│   ├── README.txt              # Project description and instructions
│   └── requirements.txt        # Python dependencies for the project
│
├── Report.pdf                  # Project report
└── [Video Demo]                # Link included in the report
