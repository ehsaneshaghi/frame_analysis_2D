import numpy as np
import copy


def build_matrix_reduced(matrix, restrained_DOF):
    # Reduce to structure stiffness matrix by deleting rows and columns for restrained DOF
    matrix = np.delete(matrix, restrained_DOF, 0)  # Delete rows
    if len(matrix.shape) > 1 and matrix.shape[1] > 1:
        matrix = np.delete(matrix, restrained_DOF, 1)  # Delete columns
    return matrix


def calc_displacement(K, force_vector, restrained_DOF):
    Kr = build_matrix_reduced(K, restrained_DOF)
    force_vector_restrained = copy.copy(
        force_vector
    )  # Make a copy of force_vector so the copy can be edited, leaving the original unchanged
    force_vector_restrained = np.delete(
        force_vector_restrained, restrained_DOF, 0
    )  # Delete rows corresponding to restrained DOF

    try:
        U = np.matmul(np.linalg.inv(Kr), force_vector_restrained).ravel()
    except:
        print("singular matrix")
        return None
    return U


def assemble_UG(U, n_DOF, restrained_DOF):
    # Construct the global displacement vector
    UG = np.zeros(n_DOF)  # Initialise an array to hold the global displacement vector
    c = 0  # Initialise a counter to track how many restraints have been imposed
    for i in np.arange(n_DOF):
        if i in restrained_DOF:
            # Impose zero displacement
            UG[i] = 0
        else:
            # Assign actual displacement
            UG[i] = U[c]
            c = c + 1

    return UG


def calc_reaction(UG, K):
    FG = np.matmul(K, UG)
    return FG
