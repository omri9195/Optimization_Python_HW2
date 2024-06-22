import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.unconstrained_min import gradient_descent, newton_method
from src.utils import plot_contours, plot_function_values
from tests.examples import quadratic_function_1, quadratic_function_2, quadratic_function_3, rosenbrock_function, linear_function, boyds_book_function


class OptimizationTestCase(unittest.TestCase):
    def setUp(self):
        # Setup initial conditions for each test
        self.x0 = np.array([1.0, 1.0])
        self.x0_rosen = np.array([-1.0, 2.0])
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 100
        self.max_iter_rosen = 10000
        self.plots_dir = 'plots_unconstrained'
        self.xlim = (-1.35, 1.35)
        self.ylim = (-1.35, 1.35)
        self.xlim_rosen = (-2.25, 2.25)
        self.ylim_rosen = (-0.25, 2.25)
        self.xlim_linear = (-220, 10)
        self.ylim_linear = (-220, 10)

    def test_quadratic_function_1(self):
        self.run_optimization_tests(quadratic_function_1, "Quadratic_1_circle_Function")

    def test_quadratic_function_2(self):
        self.run_optimization_tests(quadratic_function_2, "Quadratic_2_elipse_Function")

    def test_quadratic_function_3(self):
        self.run_optimization_tests(quadratic_function_3, "Quadratic_3_elipse2_Function")

    def test_rosenbrock_function(self):
        self.run_optimization_tests(rosenbrock_function, "Rosenbrock_Function")

    def test_linear_function(self):
        self.run_optimization_tests(linear_function, "Linear_Function")

    def test_boyds_book_function(self):
        self.run_optimization_tests(boyds_book_function, "Boyds_Book_Function")

    def run_optimization_tests(self, func, func_name):
        print(f"Running {func_name}")
        if func_name == "Rosenbrock_Function": # rosenbrock then change starting point and max iter as required
            x0 = self.x0_rosen
            max_iter = self.max_iter_rosen
        else: # if not rosenbrock, use the general provided starting point and max iter
            x0 = self.x0
            max_iter = self.max_iter

        # Create directory for plots (if it does not exist)
        func_dir = os.path.join(self.plots_dir, func_name)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)

        # Optimization tests
        try: # run optimization methods
            gd_result, gd_final_fval, gd_success, gd_iteration_data = gradient_descent(func, x0, self.obj_tol, self.param_tol, max_iter)
            print(f"{func_name} finished Gradient Descent. Success: {gd_success}")
            nm_result, nm_final_fval, nm_success, nm_iteration_data = None, None, None, []
            if not func_name == "Linear_Function": # if linear then do not run newton_method
                nm_result, nm_final_fval, nm_success, nm_iteration_data = newton_method(func, x0, self.obj_tol, self.param_tol, max_iter)
                print(f"{func_name} finished Newton Method. Success: {nm_success}")
        except Exception as e:
            print(f"Error during optimization: {e}")
            return

        print(f"Finished optimization: {func_name}")
        try:
            # Extract path data
            gd_path = np.array([step[1] for step in gd_iteration_data])
            nm_path = np.array([step[1] for step in nm_iteration_data]) if nm_iteration_data else []
            paths = [gd_path]
            path_labels = ["Gradient Descent"]
            if len(nm_path) > 0: # this is for cases where newton method was not run, so not to append its path in this case
                paths.append(nm_path)
                path_labels.append("Newton's Method")

            # Extract function values
            gd_values = [step[2] for step in gd_iteration_data]
            nm_values = [step[2] for step in nm_iteration_data] if nm_iteration_data else []
            function_values = [gd_values]
            if nm_values:
                function_values.append(nm_values)

            # Plotting, save is in plot functions
            contour_filename = os.path.join(func_dir, f"{func_name}_contours.png")
            values_filename = os.path.join(func_dir, f"{func_name}_function_values.png")
            if "Rosenbrock" in func_name: # Use different x and y limits for rosenbrock to account for different starting point
                xlim = self.xlim_rosen
                ylim = self.ylim_rosen
            elif "Linear" in func_name: # Use different limits for linear function as range is larger
                xlim = self.xlim_linear
                ylim = self.ylim_linear
            else:
                xlim = self.xlim
                ylim = self.ylim
            plot_contours(func, xlim, ylim, title=f"{func_name} Contours", path=paths, path_labels=path_labels, filename=contour_filename)
            plot_function_values(function_values, path_labels, filename=values_filename)
        except Exception as e:
            print(f"Error during plotting: {e}")
            return


if __name__ == "__main__":
    unittest.main()
