import numpy as np

# Given that the functions are pre-defined:
# In the functions below we will store and if needed return a hardcoded hessian and gradient for computational purposes


def quadratic_function_1(x, compute_hessian=False):
    Q = np.array([[1, 0], [0, 1]])
    f = np.dot(x.T, np.dot(Q, x))
    g = np.dot(Q + Q.T, x)
    h = Q + Q.T if compute_hessian else None
    return f, g, h


def quadratic_function_2(x, compute_hessian=False):
    Q = np.array([[1, 0], [0, 100]])
    f = np.dot(x.T, np.dot(Q, x))
    g = np.dot(Q + Q.T, x)
    h = Q + Q.T if compute_hessian else None
    return f, g, h


def quadratic_function_3(x, compute_hessian=False):
    R = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    D = np.array([[100, 0], [0, 1]])
    Q = np.dot(np.dot(R.T, D), R)
    f = np.dot(x.T, np.dot(Q, x))
    g = np.dot(Q + Q.T, x)
    h = Q + Q.T if compute_hessian else None
    return f, g, h


def rosenbrock_function(x, compute_hessian=False):
    f = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    g = np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])
    if compute_hessian:
        h = np.array([
            [-400 * (x[1] - 3 * x[0]**2) + 2, -400 * x[0]],
            [-400 * x[0], 200]
        ])
    else:
        h = None
    return f, g, h


def linear_function(x, compute_hessian=False):
    a = np.array([1.5, 2])  # My choice of a
    f = np.dot(a.T, x)
    g = a
    h = np.zeros((len(x), len(x))) if compute_hessian else None
    return f, g, h


def boyds_book_function(x, compute_hessian=False):
    f = np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)
    g = np.array([
        np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) - np.exp(-x[0] - 0.1),
        3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1)
    ])
    if compute_hessian:
        h = np.array([
            [np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1),
             3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1)],
            [3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1),
             9*np.exp(x[0] + 3*x[1] - 0.1) + 9*np.exp(x[0] - 3*x[1] - 0.1)]
        ])
    else:
        h = None
    return f, g, h


######################################## HW 2
# Quadratic function
def qp_obj_func(x, compute_hessian=True):
    f = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2  # x^2 + y^2 + (z+1)^2
    g = np.array([2 * x[0], 2 * x[1], 2 * x[2] + 2])  # 2x + 2y + 2z + 2
    if compute_hessian:
        h = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
        ])
    else:
        h = None
    return f, g, h


# Quadratic function inequality constraints
def qp_ineq_const_1(x, compute_hessian=True):
    f = -x[0]  # x >= 0 -> -x <= 0
    g = np.array([-1, 0, 0])
    if compute_hessian:
        h = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
    else:
        h = None
    return f, g, h


def qp_ineq_const_2(x, compute_hessian=True):
    f = -x[1]  # y >= 0 -> -y <= 0
    g = np.array([0, -1, 0])
    if compute_hessian:
        h = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
    else:
        h = None
    return f, g, h


def qp_ineq_const_3(x, compute_hessian=True):
    f = -x[2]  # z >= 0 -> -z <= 0
    g = np.array([0, 0, -1])
    if compute_hessian:
        h = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
    else:
        h = None
    return f, g, h


# Linear function
def lp_obj_func(x, compute_hessian=True):
    f = -(x[0] + x[1])  # x + y maximization -> -(x+y) minimization
    g = np.array([-1, -1])
    if compute_hessian:
        h = np.array([
            [0, 0],
            [0, 0]
        ])
    else:
        h = None
    return f, g, h


# Linear function inequality constraints
def lp_ineq_const_1(x, compute_hessian=True):
    f = x[1] - 1  # y <= 1 -> y - 1 <= 0
    g = np.array([0, 1])
    if compute_hessian:
        h = np.array([
            [0, 0],
            [0, 0]
        ])
    else:
        h = None
    return f, g, h


def lp_ineq_const_2(x, compute_hessian=True):
    f = x[0] - 2  # x <= 2 -> x - 2 <= 0
    g = np.array([1, 0])
    if compute_hessian:
        h = np.array([
            [0, 0],
            [0, 0]
        ])
    else:
        h = None
    return f, g, h


def lp_ineq_const_3(x, compute_hessian=True):
    f = -x[1]  # y >= 0 -> -y <= 0
    g = np.array([0, -1])
    if compute_hessian:
        h = np.array([
            [0, 0],
            [0, 0]
        ])
    else:
        h = None
    return f, g, h


def lp_ineq_const_4(x, compute_hessian=True):
    f = -x[0] - x[1] + 1  # y >= -x + 1 -> -x -y + 1 <= 0
    g = np.array([-1, -1])
    if compute_hessian:
        h = np.array([
            [0, 0],
            [0, 0]
        ])
    else:
        h = None
    return f, g, h
