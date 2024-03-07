import numpy as np
import math
import copy

# def calc_disp(forceVector,restrainedIndex,Ks):
#     forceVectorRed = copy.copy(forceVector)# Make a copy of forceVector so the copy can be edited, leaving the original unchanged
#     forceVectorRed = np.delete(forceVectorRed,restrainedIndex,0) #Delete rows corresponding to restrained DoF
#     U = Ks.I*forceVectorRed
#     return U


# def calc_member_forces_def(member,A,theta,L,U,E):

#     node_i = member[0] #Node number for node i of this member
#     node_j = member[1] #Node number for node j of this member
#     #Primary stiffness matrix indices associated with each node

#     #Transformation matrix
#     c = math.cos(theta)
#     s = math.sin(theta)
#     T = np.array([[c,s,0,0],[0,0,c,s]])

#     disp = np.array([[U[node_i][0],U[node_i][1],U[node_j][0],U[node_j][1]]]).T #Glocal displacements
#     disp_local = np.matmul(T,disp).ravel() #Local displacements
#     F_axial = (A*E/L)*(disp_local[1]-disp_local[0]) #Axial loads
#     mbrForce = F_axial #Store axial loads
#     mbrDef = disp_local[1]-disp_local[0]
#     return mbrForce,mbrDef

import numpy as np
import math


def transform_F_to_local(member, theta, K_member, U, PL):
    node_i = member[0]  # Node number for node i of this member
    node_j = member[1]  # Node number for node j of this member
    # Primary stiffness matrix indices associated with each node

    i = 3 * node_i
    j = 3 * node_j

    # Transformation matrix
    c = math.cos(theta)
    s = math.sin(theta)

    T = np.array(
        [
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    U_member = np.array([[U[i], U[i + 1], U[i + 2], U[j], U[j + 1], U[j + 2]]]).ravel()  # global displacement
    PL_member = np.array([[PL[i], PL[i + 1], PL[i + 2], PL[j], PL[j + 1], PL[j + 2]]]).ravel()  # global displacement

    # local_K = T @ K_member @ T
    # # local_F = local_K @ U_member
    # local_F = K_member @ T @ U_member
    a = K_member @ U_member
    b = -(a - PL_member)
    c = T @ b
    return c


def transform_U_to_local(vector, member, theta):
    node_i = member[0]  # Node number for node i of this member
    node_j = member[1]  # Node number for node j of this member
    # Primary stiffness matrix indices associated with each node

    i = 3 * node_i
    j = 3 * node_j

    # Transformation matrix
    c = math.cos(theta)
    s = math.sin(theta)

    T = np.array(
        [
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    vector = np.array(
        [[vector[i], vector[i + 1], vector[i + 2], vector[j], vector[j + 1], vector[j + 2]]]
    ).T  # global displacement

    local_vector = np.matmul(T, vector)

    return local_vector


def transform_V_to_local(v, theta):
    # Transformation matrix
    c = math.cos(theta)
    s = math.sin(theta)

    T = np.array(
        [
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    local_vector = np.matmul(T, v)

    return local_vector


def inverse_rotate_vector(vector, theta):
    # Transformation matrix
    c = math.cos(theta)
    s = math.sin(theta)

    T = np.array(
        [
            [c, -s],
            [s, c],
        ]
    ).T  # inverse rotation

    rotated_vector = np.matmul(vector, T)
    return rotated_vector


def calc_shear_diagram(F, L, step_size):
    bin_count = int(L / step_size)
    SD = np.ones(bin_count) * F[1]
    return SD


def calc_moment_diagram(F, L, step_size):
    bin_count = int(L / step_size)
    X = [k * step_size for k in range(bin_count)]
    a = (F[5] - F[2]) / (L - step_size)
    MD = np.array([F[2] + X[i] * a for i in range(len(X))]).flatten()
    return MD


def calc_deflection_y_diagram_numerical(MD, EI, L, step_size, U):
    bin_count = int(L / step_size)

    # Initialise containers to hold Rotation and Deflection
    rotation = np.zeros(bin_count)
    rotation[0] = U[2]
    deflection = np.zeros(bin_count)
    deflection[0] = U[1]

    M_im1 = MD[0]
    rotation_im1 = rotation[0]
    V_im1 = deflection[0]

    # Loop through data and integrate (Trapezoidal rule)
    for i in range(1, len(MD)):
        # M_avg = 0.5 * (MD[i] + M_im1)
        # rotation[i] = rotation_im1 + (M_avg / EI) * step_size  # Integrate moment values to get rotations
        rotation[i] = np.cumsum(MD[: i + 1])[-1] / EI * step_size + rotation[0]
        deflection[i] = (
            V_im1 + 0.5 * (rotation[i] + rotation_im1) * step_size
        )  # Integrate rotation values to get displacements

        # Update values for next loop iteration
        rotation_im1 = rotation[i]
        V_im1 = deflection[i]

    return deflection


def calc_deflection_x_diagram_numerical(L, step_size, U):
    bin_count = int(L / step_size)
    deflection = np.zeros(bin_count)
    deflection[0] = U[0]
    slope = (U[3] - U[0]) / L
    for i in range(bin_count):
        deflection[i] = slope * (i * step_size) + U[0]

    return deflection


def calc_deflection_diagram_analytical(F, EI, L, step_size, U):
    bin_count = int(L / step_size)
    deflection = [
        U[1]
        + U[2] * ((i * step_size))
        + 1 / EI * (-1 / 2 * F[2] * (i * step_size) ** 2 + 1 / 6 * F[1] * (i * step_size) ** 3)
        for i in range(bin_count)
    ]
    return np.array(deflection).ravel()


def calc_deflection_diagram_analytical2(F, EI, L, step_size, U):
    bin_count = int(L / step_size)
    deflection = [
        U[1]
        + U[2] * ((i * step_size))
        + 1 / EI * (-1 / 2 * F[2] * (i * step_size) ** 2 + 1 / 6 * F[1] * (i * step_size) ** 3)
        for i in range(bin_count)
    ]
    return np.array(deflection).ravel()


def calc_axial_force_from_displacement(L, U, E, member_area, node_i, node_j):
    L_new = np.sqrt((node_j[0] + U[3] - node_i[0] - U[0]) ** 2 + (node_j[1] + U[4] - node_i[1] - U[1]) ** 2)
    axial_disp = L - L_new
    F_axial = (member_area * E / L) * axial_disp  # Axial loads

    return F_axial


def extract_member_vector(member, v):
    node_i = member[0]  # Node number for node i of this member
    node_j = member[1]  # Node number for node j of this member
    # Primary stiffness matrix indices associated with each node

    i = 3 * node_i
    j = 3 * node_j

    vector_member = np.array([[v[i], v[i + 1], v[i + 2], v[j], v[j + 1], v[j + 2]]]).ravel()  # global displacement
    return vector_member
