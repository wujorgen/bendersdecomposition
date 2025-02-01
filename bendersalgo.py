import numpy as np
import cvxpy as cp

assert "SCIPY" in cp.installed_solvers()
# assert "GUROBI" in cp.installed_solvers()

class BendersDecompositionProblem:
    """Benders Decomposition for a Mixed-Integer Problem.
    
    The problem is of form:
    - min z = c^T x + d^T y
    - s.t
    - A y >= b
    - E x + F y >= h
    - x >= 0, y >= 0
    - y is integer
    """
    def __init__(self, **kwargs):
        """Initializes the mixed-integer problem.
        
        Args:
        - c: np.ndarray
        - d: np.ndarray
        - A: np.ndarray
        - b: np.ndarray
        - E: np.ndarray
        - F: np.ndarray
        - h: np.ndarray
        """
        if "SCIPY" not in cp.installed_solvers():
            raise Exception("The cvxpy[SCIPY] solver is not installed.")
        if "GUROBI" not in cp.installed_solvers():
            print("***WARNING: cvxpy[GUROBI] is not installed. Defaulting to cvxpy[SCIPY]...")
        self._INTEGER_SOLVER = "GUROBI" if "GUROBI" in cp.installed_solvers() else "SCIPY"
        self._SOLVER = "SCIPY"
        self._CONVERGED = False
        self.x_sol = None
        self.y_sol = None
        self.u_sol = None
        self.itrs_sol = None
        self.c = None
        self.d = None
        self.A = None
        self.b = None
        self.E = None
        self.F = None
        self.h = None
        self.x_geq = 0
        self.x_leq = None
        self.y_geq = 0
        self.y_leq = None
        self.TOL = 1e-3
        self.MAXITR = 20
        self.LB = -np.inf
        self.UB = np.inf
        self.lower_bounds = []
        self.upper_bounds = []
        self.optimality_constraints = []
        self.feasibility_constraints = []
        for key, val in kwargs.items():
            setattr(self, key, val)
        # check dimensions
        assert self.c.shape[0] == self.E.shape[1]
        assert self.d.shape[0] == self.F.shape[1], f"{self.d.shape[0]} =/= {self.F.shape[1]}"
        assert self.h.shape[0] == self.E.shape[0]
        assert self.h.shape[0] == self.F.shape[0]
        if self.A is not None:
            assert self.d.shape[0] == self.A.shape[1]
        if self.b is not None or self.A is not None:
            assert self.b.shape[0] == self.A.shape[0]
        # assign number of variables
        self.Nx = self.c.shape[0]
        self.Ny = self.d.shape[0]
        self.Nc = self.E.shape[0]  # number of constraints
        # dual dimension?
        # Call Solution Algorithm
        # self.solve()
        # self.report()

    def solve(self):
        # (0) solve Main Problem
        y, z_lower, status = self._solve_main_problem(itrzero=True)
        if status == "unbounded":
            self.LB = -np.inf
        elif status == "infeasible":
            raise Exception("Main problem is infeasible.")
        else:
            self.LB = z_lower
        # y = np.array([4.]) # TODO: for some reason, end up using this for the Ab constraint test...

        for itr in range(self.MAXITR):
            # (itr) solve Sub Problem (dual form)
            u, x, objval, status = self._solve_dual_subproblem(y)
            if status == "optimal":
                self.UB = self.d@y + (self.h - self.F@y)@u
                # check for convergence
                if np.isclose(self.UB, self.LB, rtol=self.TOL):
                    self._CONVERGED = True
                    break
                # add optimality constraint
                self.optimality_constraints.append(u)
            elif status == "unbounded":
                self.UB = np.inf
                r_hat, _, status= self._perform_feasibility_check(y)
                if status == "optimal":
                    self.feasibility_constraints.append(r_hat)
                else:
                    raise Exception("Feasibility check was not able to find a solution.")
            elif status == "infeasible":
                raise Exception("The dual subproblem is infeasible. Stopping iterations.")

            # (itr + 1) solve Main Problem w/ additional constraints
            y, z_lower, status = self._solve_main_problem()
            if status == "unbounded":
                self.LB = -np.inf
            elif status == "infeasible":
                raise Exception("Main problem is infeasible.")
            # proceed if optimal solution found for main
            self.LB = np.max((self.LB, z_lower))
            # store convergence information
            self.lower_bounds.append(self.LB)
            self.upper_bounds.append(self.UB)
            
        # Store optimal or final solution
        self.x_sol = x
        self.y_sol = y
        self.u_sol = u
        self.z_sol = self.UB
        self.itrs_sol = itr

    def _solve_main_problem(self, itrzero:bool=False):
        """Solves the Main Problem.
        
        The form is:
        - min d^T y
        - A y >= b
        - y in S

        Args:
        - []

        Returns:
        - []
        """
        z = cp.Variable()
        y = cp.Variable(self.Ny, integer=True)
        if itrzero or self.optimality_constraints == []:
            objective = self.d @ y
        else:
            n = cp.Variable()
            objective = self.d @ y + n
        constraints = [y >= self.y_geq]
        if self.y_leq is not None:
            constraints.append(y <= self.y_leq)
        if self.A is not None and self.b is not None:
            constraints.append(self.A @ y >= self.b)
        for u_p in self.optimality_constraints:
            constraints.append(n >= (self.h - self.F@y)@u_p)
        for u_r in self.feasibility_constraints:
            constraints.append((self.h - self.F@y)@u_r <= 0)
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=self._INTEGER_SOLVER)
        return y.value, prob.value, prob.status

    def _solve_subproblem(self, y_hat):
        """Solves the subproblem.

        The form is:
        - min c^T x
        - Ex >= h - F y_hat
        - x >= 0
        """
        pass

    def _solve_dual_subproblem(self, y_hat):
        """Solves the dual subproblem.
        
        The form is:
        - max (h - F y_hat)^T u
        - E^T u <= c
        - u >= 0
        
        Args:
        - y_hat: np.ndarray: temporary integer solution

        Returns:
        - u.value: np.ndarray: dual variable
        - x: dual of the dual is...the original?!??
        - prob.value: value of objective function
        - prob.status: string: problem status
        """
        u = cp.Variable(self.Nc)
        objective = (self.h - self.F @ y_hat) @ u
        constraints = [
            u >= np.zeros_like(u), 
            self.E.T @ u <= self.c
        ]
        prob = cp.Problem(cp.Maximize(objective), constraints)
        prob.solve(solver=self._SOLVER)
        x = constraints[1].dual_value
        return u.value, x, prob.value, prob.status


    def _perform_feasibility_check(self, y_hat):
        """Performs the feasibility check.
        
        The form is:
        - min 1^T s
        - E x + I s >= h - F y_hat --> u_r
        - x >= 0, s >= 0
        
        Args:
        - y_hat: np.ndarray: temporary integer solution

        Returns:
        r_hat: np.ndarray: dual value(s)
        """
        s = cp.Variable(self.Nx)
        x = cp.Variable(self.Nx)
        objective = np.ones(self.Nx) @ s
        constraints = [
            x >= 0,
            s >= 0,
            self.E @ x + np.eye(self.Nx) @ s >= self.h - self.F @ y_hat,
        ]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=self._SOLVER)
        r_hat = constraints[2].dual_value
        return r_hat, prob.value, prob.status
    
    def report(self, verbose:bool=False):
        """Reports the results of the solution."""
        if self._CONVERGED:
            print("Solution converged.")
        else:
            print("Solution did not converge.")
        print(f"Objective function value: {self.z_sol:7.4f}")
        print("X values: ", end="")
        for val in self.x_sol:
            print(f"{val:7.4f}", end=", ")
        print()
        print("Y values: ", end="")
        for val in self.y_sol:
            print(f"{val:7.4f}", end=", ")
        print()
        print("U values: ", end="")
        for val in self.u_sol:
            print(f"{val:7.4f}", end=", ")
        print()
        print("The upper and lower bounds on the solution are:")
        print([float(x) for x in self.upper_bounds])
        print([float(x) for x in self.lower_bounds])
        if verbose:
            print("Confirm that the constraints are satisfied:")
            print("c^T @ x + d^T @ y =", self.c @ self.x_sol + self.d @ self.y_sol)
            print("E @ x + F @ y =", self.E @ self.x_sol + self.F @ self.y_sol, ">=", self.h)
            if self.A is not None and self.b is not None:
                print("A @ y =", self.A @ self.y_sol, ">=", self.b)



if __name__ == "__main__":
    def example_5_1():
        c = np.array([1., 3.])
        d = np.array([1., 4.])
        E = np.array([[-2., -1.], [2., 2.]])
        F = np.array([[1., -2.], [-1., 3.]])
        h = np.array([1., 1.])
        kwargs = {
            "c": c,
            "d": d,
            "E": E,
            "F": F,
            "h": h,
            "x_geq": 0,
            "y_geq": 0,
        }
        return kwargs
    
    def example_4_1():
        c = np.array([1.])
        d = np.array([1.])
        E = np.array([[2.]])
        F = np.array([[1.]])
        h = np.array([3.])
        kwargs = {
            "c": c,
            "d": d,
            "E": E,
            "F": F,
            "h": h,
            "x_geq": 0,
            "y_geq": -5,
            "y_leq": 4
        }
        return kwargs
    
    def A_b_constraint():
        # TODO: 
        # manually setting y_0 = 4 achieves a min value of -24
        # however, running the interation zero main problem chieves a in value of -20...
        c = np.array([-4., -3.])
        d = np.array([-5.])
        E = np.array([[-2., -3.],[-2., -1.]])
        F = np.array([[-1.], [-3.]])
        h = np.array([-12., -12.])
        A = np.array([[-1.]])
        b = np.array([-20.])
        kwargs = {
            "c": c,
            "d": d,
            "E": E,
            "F": F,
            "h": h,
            "A": A,
            "b": b,
            "x_geq": 0,
            "y_geq": 0,
        }
        return kwargs

    BP = BendersDecompositionProblem(**A_b_constraint())
    BP.solve()
    BP.report(verbose=True)
    breakpoint()
