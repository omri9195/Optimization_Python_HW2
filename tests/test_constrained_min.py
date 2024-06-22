import unittest
import numpy as np
from src.utils import plot_results_qp, plot_results_lp, plot_values_graph
from src.constrained_min import interior_pt
from tests.examples import qp_obj_func, qp_ineq_const_1, qp_ineq_const_2, qp_ineq_const_3, lp_obj_func, lp_ineq_const_1, lp_ineq_const_2, lp_ineq_const_3, lp_ineq_const_4
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class TestMinimize(unittest.TestCase):
    """
    Note that self.eq_constraints_rhs_qp = 0 in this implementation. Intuitively it seems that it should be set to 1.
    BUT, to ensure satisfaction of x + y + z = 1, we set initial point self.x0_qp = np.array([0.1, 0.2, 0.7]).
    By ensuring the initial guess satisfies the constraint, the KKT system is set up to enforce this constraint.
    Conversely, setting b=1 by self.eq_constraints_rhs_qp = 1 (which seems intuitive), misses the way the solver
    handles the constraint, it assumes a homogeneous form ( = 0), and thus it would not work.
    TL:DR - solver assumes homogeneous form(=0). Ensure initial point x0 satisfies constraint, set b = 0.
    """
    def setUp(self):
        # Setup initial conditions for each test
        self.x0_qp = np.array([0.1, 0.2, 0.7])
        self.x0_lp = np.array([0.5, 0.75])
        self.ineq_constraints_qp = [qp_ineq_const_1, qp_ineq_const_2, qp_ineq_const_3]
        self.ineq_constraints_lp = [lp_ineq_const_1, lp_ineq_const_2, lp_ineq_const_3, lp_ineq_const_4]
        self.eq_constraints_mat_qp = np.array([1, 1, 1]).reshape(1, 3)
        self.eq_constraints_rhs_qp = 0
        self.eq_constraints_mat_lp = None
        self.eq_constraints_rhs_lp = None

        self.plots_dir = 'plots_constrained'

    def run_test(self, func, ineq_constraints, A, b, x0, plot_path_func, plot_values_title, plot_path_title):
        # Run the interior pt method and get values
        final_candidate, final_obj, iteration_data = interior_pt(func, ineq_constraints, A, b, x0)
        path, values = zip(*[(item[0], item[1]) for item in iteration_data])

        # Get ineq and eq constraint values at final candidate
        ineq_constraints_values = [ineq(final_candidate)[0] for ineq in ineq_constraints]
        if A is not None:
            eq_constraints_value = (np.dot(A, final_candidate)).sum()

        # Print metrics to use in report and cmd when running
        print('Final candidate:', final_candidate)
        print('Objective function value at final candidate:', final_obj)
        print('Inequality constraints values at final candidate:', ineq_constraints_values)
        if A is not None:
            print('Equality constraints values at final candidate:', eq_constraints_value)

        # Define subdirectory to save plots as LP or QP
        subdir_name = func.__name__.swapcase()[:2]
        # Create directory for plots (if it does not exist)
        func_dir = os.path.join(self.plots_dir, subdir_name)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)

        # Build filenames for plots
        feasible_path_filename = os.path.join(func_dir, f"{subdir_name}_feasible_and_path.png")
        obj_values_filename = os.path.join(func_dir, f"{subdir_name}_objective_function_values.png")

        # Plot results (this saves when filename is specified)
        plot_values_graph(values, plot_values_title, obj_values_filename)
        plot_path_func(path, final_obj, plot_path_title, feasible_path_filename)

    def test_qp(self):
        ineq_constraints_qp = self.ineq_constraints_qp
        A = self.eq_constraints_mat_qp
        b = self.eq_constraints_rhs_qp
        x0 = self.x0_qp

        self.run_test(qp_obj_func, ineq_constraints_qp, A, b, x0, plot_results_qp,
                      'QP - Objective function values as a function of outer iteration',
                      'QP - Path taken and feasible region, final candidate, objective'
                      ' and constraint values at final candidate')

    def test_lp(self):
        ineq_constraints_lp = self.ineq_constraints_lp
        A = self.eq_constraints_mat_lp
        b = self.eq_constraints_rhs_lp
        x0 = self.x0_lp

        self.run_test(lp_obj_func, ineq_constraints_lp, A, b, x0, plot_results_lp,
                      'LP - Objective function values as a function of outer iteration',
                      'LP - Path taken and feasible region, final candidate, objective'
                      ' and constraint values at final candidate')


if __name__ == "__main__":
    unittest.main()
