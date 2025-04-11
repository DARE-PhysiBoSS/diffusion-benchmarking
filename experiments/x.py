import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee, connected_components


# Function to create the NxN 2D diffusion matrix for implicit finite difference
def create_diffusion_matrix(N, D, delta_t, delta_x, delta_y):
    # Number of grid points in the 2D grid
    num_points = N * N
    # Create a sparse matrix with the structure for implicit FD method
    diagonal = -4 * np.ones(num_points)
    off_diagonal_x = np.ones(num_points - 1)
    off_diagonal_y = np.ones(num_points - N)

    # Main diagonal, sub-diagonal, and super-diagonal
    A = sp.diags([diagonal, off_diagonal_x, off_diagonal_x, off_diagonal_y, off_diagonal_y],
                 [0, 1, -1, N, -N], shape=(num_points, num_points))

    # The matrix needs to be scaled by D * delta_t / (delta_x^2)
    scale_factor = D * delta_t / (delta_x**2)
    A *= scale_factor

    return A

# Function to compute the minimum degree ordering based on the quotient graph
def minimum_degree_permutation(A):
    # Get the graph from the sparse matrix
    graph = A.tocsr()

    # Number of nodes in the graph (size of matrix)
    num_nodes = graph.shape[0]

    # Connected components of the graph
    n_components, labels = connected_components(graph)

    # Create the quotient graph by treating each connected component as a supernode
    quotient_graph = sp.lil_matrix((n_components, n_components))

    # Build quotient graph based on connected components
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if graph[i, j] > 0:
                # If nodes i and j are connected, update the quotient graph
                quotient_graph[labels[i], labels[j]] = 1
                quotient_graph[labels[j], labels[i]] = 1

    # Compute minimum degree ordering of the quotient graph
    degrees = quotient_graph.sum(axis=1).A1
    ordering = np.argsort(degrees)

    # Reverse the quotient graph ordering to permute original matrix
    return ordering

# Function to apply reverse Cuthill-McKee (RCM) permutation to reduce bandwidth
def apply_rcm_permutation(matrix):
    # Use Reverse Cuthill-McKee algorithm to reorder the matrix
    rcm_order = reverse_cuthill_mckee(csgraph.csgraph_from_dense(matrix))
    return matrix[rcm_order, :][:, rcm_order]

# Parameters
N = 10  # Grid size (NxN)
D = 1.0  # Diffusion coefficient
delta_t = 0.01  # Time step
delta_x = 1.0  # Grid spacing in x
delta_y = 1.0  # Grid spacing in y

# Step 1: Create the 2D diffusion matrix for the implicit finite difference method
A = create_diffusion_matrix(N, D, delta_t, delta_x, delta_y)

order = minimum_degree_permutation(A)
A_permuted_1 = A.toarray()[order, :][:, order]

# Convert the sparse matrix to a dense matrix
A = A.toarray()

# Step 2: Apply RCM permutation to reduce bandwidth
A_permuted = apply_rcm_permutation(A)

# Step 3: Plot the matrix before and after permutation
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

# Plot the original matrix
ax1.spy(A, markersize=1)
ax1.set_title("Original Matrix")

# Plot the permuted matrix
ax2.spy(A_permuted, markersize=1)
ax2.set_title("Permuted Matrix (RCM)")

# Plot the permuted matrix
ax3.spy(A_permuted_1, markersize=1)
ax3.set_title("Permuted Matrix (MD)")

plt.show()
