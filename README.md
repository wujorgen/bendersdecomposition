# Benders Decomposition
Implementation of Benders Decomposition for mixed-integer programming problems.

# Algorithm
Benders Decomposition is a technique for solving large linear programming problems. It takes advantage of a block structure (partitioned matrix) that may be present in the problem to seperate, or decompose, the original problem into two stages. This can be used for two-stage stochastic programming, or as implemented here, for solving mixed-integer problems. 

Specifically for mixed-integer problems, the integer and continuous portions of the problem are seperated. The integer portion is solved first as the main problem and then the continous portion is solved second as the subproblem. The output of the main problem represents a lower bound on the optimal value of the objective function, while the output of the subproblem represents an upper bound on the optimal value of the objective function. 

Due to formatting restrictions in markdown files, please see the Python notebook ```algorithm_notes.ipynb``` for a more in depth description of the algorithm along with equations and examples.

# Set up
Install Python using the method of your choice. This could be Chocolately, Python.org, Brew, apt, or any other method. From there:
```shell
python -m venv venv
# on Linux/Mac:
source venv/bin/activate
# install requirements:
pip install -r Requirements.txt
```

# Run It
You can run the algorithm by calling the following command from the root directory of this repository:
```
python main.py
```

The system takes the form presented in the Mathematical Background section of ```algorithm_notes.ipynb```. You can update the system of equations by modifying the variables in ```equations.py```. The algorithm itself is contained in ```bendersalgo.py```.

# Open Items
This list was last updated 2/8/2025
- Quantum computing has potential when applied to combinatorial optimization. It would be interesting to see if D-Wave's quantum annealers could be used to solve the main problem, if it was reformatted as a QUBO.
- More benchmarking. The examples are small systems.