import numpy as np


def interior_pt(f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    # Set parameters to use, with values defined in previous HW1 as well as in HW2
    t = 1
    mu = 10
    obj_tol = 1e-12
    param_tol = 1e-8
    # set to 10 as it is sufficient, higher values exhibit similar behavior with equality constraints evaluating to
    # 1.00000002-1.00000009, while with 10 they evaluate to 1.0
    inner_max_iter = 10
    x = np.array(x0, dtype=float)

    # Calculate initial barrier function values
    f_val, grad_val, hess_val = log_barrier_wrapper(f, ineq_constraints, x, t)

    # Create iteration data to store history for plots
    iteration_data = [(x, f(x)[0])]
    print(f"Start Point: x = {x}, f(x) = {f_val}")
    outer_iteration_counter = 1

    # Termination condition m/t < eps = objective tolerance
    while len(ineq_constraints) / t >= obj_tol:
        for i in range(inner_max_iter):
            # Get direction p and step size a (alpha) - update x
            p = filter_p_by_eq_constraints(hess_val, eq_constraints_mat, eq_constraints_rhs, grad_val)
            a = wolfe_condition_with_ineq_constraints(f, x, p, grad_val, ineq_constraints)
            x = x + a * p

            # Get barrier function values at updated x
            f_val, grad_val, hess_val = log_barrier_wrapper(f, ineq_constraints, x, t)

            if param_tol_check(hess_val, p, param_tol):
                break

            print(f"Inner iteration {outer_iteration_counter - 1}.{i}: x = {x}, f(x) = {f(x)[0]}")

        print(f"Outer iteration {outer_iteration_counter}: x = {x}, f(x) = {f(x)[0]}")
        outer_iteration_counter += 1
        iteration_data.append((x, f(x)[0]))
        t *= mu  # increase by factor of mu = 10 each outer iteration as required

    return x, f(x)[0], iteration_data


def wolfe_condition_with_ineq_constraints(f, x, p, g, ineq_constraints, b=0.5, c=0.01, tol=1e-8):
    a = 1.0
    f_x, _, _ = f(x)  # get function value of x before trial updates
    while True:
        x_new = x + a * p  # calculate trial point
        f_x_new, _, _ = f(x_new)  # get func value of trial point
        all_ineq_met = True
        # Check if any inequality constraint is violated
        if any(q(x_new)[0] > 0 for q in ineq_constraints):  # If true, an inequality constraint was not satisfied
            a *= b
            all_ineq_met = False
        elif f_x_new > f_x + c * a * np.dot(g.T, p):  # if true, wolfe first condition not satisfied
            a *= b
        else:
            break
        if a < tol and all_ineq_met:  # Make sure not to terminate when an inequality constraint is not satisfied as per instructions
            break
    return a


def log_barrier(ineq_constraints, x):
    x_dim = x.shape[0]
    # set numpy arrays (or scalars) with appropriate dim for log barrier
    log_f = 0
    log_g = np.zeros((x_dim,))
    log_h = np.zeros((x_dim, x_dim))
    # Calculate values and update numpy arrays (or Scalars), this is as in slides but vectorized (hopefully optimally)
    for constraint in ineq_constraints:
        f, g, h = constraint(x)
        log_f += np.log(-f)
        log_g += -g / f
        log_h += (np.outer(-g, g) - f * h) / f**2
    return -log_f, log_g, -log_h


def log_barrier_wrapper(f, ineq_constraints, x, t):
    f_val, grad_val, hess_val = f(x)
    log_f, log_g, log_h = log_barrier(ineq_constraints, x)
    # Return the penalized log barrier values
    pen_f = t * f_val + log_f
    pen_g = t * grad_val + log_g
    pen_h = t * hess_val + log_h
    return pen_f, pen_g, pen_h


def get_p_with_eq(hessian, A, b, gradient):
    num_eq_constraints = A.shape[0]

    # Build KKT matrix to solve
    kkt_matrix = np.block([
        [hessian, A.T],
        [A, np.zeros((num_eq_constraints, num_eq_constraints))]
    ])
    rhs = np.hstack([-gradient, np.full(num_eq_constraints, b)])

    # Solve the KKT system
    kkt_sol = np.linalg.solve(kkt_matrix, rhs)
    p = kkt_sol[:gradient.shape[0]]

    return p


def get_p_no_eq(hessian, gradient):
    return np.linalg.solve(hessian, -gradient)


def filter_p_by_eq_constraints(hessian, A, b, gradient):
    # Filter to get direction p by LP or QP problem
    if A is None:
        return get_p_no_eq(hessian, gradient)
    else:
        return get_p_with_eq(hessian, A, b, gradient)


def param_tol_check(hess_val, p, param_tol):
    lamda = np.sqrt(np.dot(p, np.dot(hess_val, p.T)))
    return 0.5 * (lamda ** 2) < param_tol

