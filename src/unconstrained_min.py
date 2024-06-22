import numpy as np


def gradient_descent(f, x0, obj_tol, param_tol, max_iter):
    x, iteration_data = init_optimization(x0, f)
    for i in range(1, max_iter + 1):
        f_val, grad_val, _ = f(x, compute_hessian=False) # get function value and gradient
        if magnitude_check(grad_val, param_tol):  # the gradients magnitude small enough to stop - gradient norm check
            return x, f_val, True, iteration_data
        p = -grad_val  # Calculate the direction p based on the negative gradient
        x_new, f_x_new = update_by_direction(i, p, f, x, grad_val, iteration_data)
        # if either not enough change in objective function value, or small enough direction norm/magnitude
        if loss_or_param_tol_check(x_new, x, f_x_new, f_val, param_tol, obj_tol):
            return x_new, f_x_new, True, iteration_data

        x = x_new # update x to x_new for next iteration

    return x, f_val, False, iteration_data


def newton_method(f, x0, obj_tol, param_tol, max_iter):
    x, iteration_data = init_optimization(x0, f)
    for i in range(1,max_iter+1):
        f_val, grad_val, hess_val = f(x, compute_hessian=True) # Get Hessian as well for newton method
        if magnitude_check(grad_val, param_tol): # if the gradients magnitude is small enough to stop - gradient norm check
            return x, f_val, True, iteration_data
        p = np.dot(np.linalg.inv(hess_val), -grad_val) # get direction p by dot prod of inverse hessian with negative gradient
        x_new, f_x_new = update_by_direction(i, p, f, x, grad_val, iteration_data)
        # if either not enough change in objective/loss function value, or small enough direction norm/magnitude
        if loss_or_param_tol_check(x_new, x, f_x_new, f_val, param_tol, obj_tol):
            return x_new, f_x_new, True, iteration_data

        x = x_new

    return x, f_val, False, iteration_data


def wolfe_condition(f, x, p, g, b=0.5, c=0.01, tol=1e-8):
    a = 1.0 # Start from a = 1 initial value (alpha)
    f_x, _, _ = f(x, compute_hessian=False)  # get function value of x before trial updates
    while True:
        x_new = x + a * p  # calculate a trial point x_new based on step size * direction p = steepest descent (GD) or hessian * negative gradient (NM)
        f_x_new, _, _ = f(x_new, compute_hessian=False) # get function value no need hessian and gradient
        if f_x_new > f_x + c * a * np.dot(g.T, p): # check if wolfe condition is not fulfilled yet and update a
            a *= b
        else:
            break # wolfe condition is fulfilled return step size a
        if a < tol:  # This is 27 iterations as (1/2)^27 < 1e-8
            print("Wolfe condition line search terminated by tolerance")  # terminate with wolfe condition unfulfilled if alpha factor is so small its negligent to continue
            break
    return a


# Helper functions


def init_optimization(x0, f):
    x = np.array(x0, dtype=float)  # turn x into a float nparray for comfort
    iteration_data = [(0, x, f(x, compute_hessian=False)[0])]  # Create iteration data with initial x and func value
    print(f"Iteration {iteration_data[0][0]}: x = {iteration_data[0][1]}, f(x) = {iteration_data[0][2]}")
    return x, iteration_data


def update_by_direction(i, p, f, x, grad_val, iteration_data):
    a = wolfe_condition(f, x, p, grad_val)  # get step size from wolfe condition
    x_new = x + a * p  # update x to x_new with step size t to direction p
    f_x_new, _, _ = f(x_new, compute_hessian=False)
    iteration_data.append((i, x_new, f_x_new))
    print(f"Iteration {i}: x = {x_new}, f(x) = {f_x_new}")
    return x_new, f_x_new


def magnitude_check(grad_val, param_tol):
    return np.sqrt(np.dot(grad_val, grad_val)) < param_tol


def loss_or_param_tol_check(x_new, x, f_x_new, f_val, param_tol, obj_tol):
    return np.sqrt(np.sum((x_new - x)**2)) < param_tol or np.abs(f_x_new - f_val) < obj_tol

