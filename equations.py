import numpy as np

def system_of_equations() -> dict:
    """Returns the system of equations for a mixed-integer problem.

    Ensure that E and F are defined as 2d arrays.
    """
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
