import numpy as np
import matplotlib.pyplot as plt


def plot_contours(f, x_limits, y_limits, title="Loss Function Contour", path=None, path_labels=None, filename=None):
    # Create grid / plot space based on limits input
    x = np.linspace(x_limits[0], x_limits[1], 500)
    y = np.linspace(y_limits[0], y_limits[1], 500)
    X, Y = np.meshgrid(x, y)

    # Get function values for points on plot space / grid
    Z = np.array([[f(np.array([X[i, j], Y[i, j]]))[0] for j in range(X.shape[1])] for i in range(X.shape[0])])

    plt.figure(figsize=(16, 12))

    # Handle linear differently without log scale/spacing for contours
    if "Linear" in title:
        levels = np.linspace(np.min(Z), np.max(Z), 30)
    else: # not linear, log scale/spacing for visualization
        levels = np.logspace(np.log10(np.min(Z)), np.log10(np.max(Z)), 30)

    # plot contour lines
    cp = plt.contour(X, Y, Z, levels=levels, cmap='cividis')

    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    # Plot optimization paths per function
    if path is not None and path_labels is not None:
        for p, label in zip(path, path_labels):
            if "Newton" in label:
                plt.plot(p[:, 0], p[:, 1], marker='o', label=label, alpha=0.8)
            else:
                plt.plot(p[:, 0], p[:, 1], marker='x', label=label, markersize = 10)
        plt.legend()

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_function_values(iteration_data, labels, filename=None):
    plt.figure(figsize=(16, 12))

    # Plot function values
    for data, label in zip(iteration_data, labels):
        if "Newton" in label:
            plt.plot(data, label=label, marker='o', alpha = 0.8)
        else:
            plt.plot(data, label=label, marker='x', markersize = 10)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Function Values by Iteration')
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


################################# HW2
def plot_results_qp(path, final_obj, title, filename = None):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    path = np.array(path)

    # Define feasible region
    vertices = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Plot feasible region
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='green', alpha=0.4)

    # Plot the path taken by the algorithm
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Alg Path', color='black')
    # Pinpoint final candidate
    ax.scatter(path[-1][0], path[-1][1], path[-1][2], s=200, c='magenta', marker='o', label='Final candidate')

    # Plotting the constraints as equalities (x=0,y=0,z=0) produces less clear visuals in 3d
    # Thus, these planes are not presented in the plot

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.grid(True)
    ax.view_init(25, 75)  # Can change this to play with viewpoint

    # As required, constraints and obj value at final candidate
    annotation = f"Objective function value: {final_obj:.4f}"
    plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_results_lp(path, final_obj, title, filename = None):
    fig, ax = plt.subplots(figsize=(16, 12))
    path = np.array(path)

    # Define the constraints for plot
    x = np.linspace(-0.5, 2, 1000)
    y = np.linspace(-1, 2, 1000)
    constraints_ineq = {
        'y=-x+1': (x, -x + 1),  # y = -x + 1
        'y=1': (x, np.ones_like(x)),  # y = 1
        'x=2': (np.ones_like(y) * 2, y),  # x = 2
        'y=0': (x, np.zeros_like(x))  # y = 0
    }

    # Plot the constraints
    for label, (x, y) in constraints_ineq.items():
        line, = ax.plot(x, y, label=label)
        line.set_dashes([1,2])

    # Plot the feasible region
    ax.fill([1, 2, 2, 0], [0, 0, 1, 1], color='green', alpha=0.25, label='Feasible region')

    # Plot the path taken by the algorithm
    ax.plot(path[:, 0], path[:, 1], label='Alg Path', color = "black")
    ax.scatter(path[-1][0], path[-1][1], s=200, c='magenta', marker='o', label='Final candidate')

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='lower left')
    plt.grid(True)

    # As required, constraints and obj value at final candidate
    annotation = f"Objective function value: {final_obj:.3f}"
    plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_values_graph(values, title, filename = None):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.plot(values, marker='.')
    ax.set_title(title)
    ax.set_xlabel('Outer Iterations')
    ax.set_ylabel('Objective function values')
    plt.grid(True)

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
