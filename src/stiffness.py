import math  # Math functionality
import numpy as np  # Numpy for working with arrays
import copy  # Allows us to create copies of objects in memory


# Define a function to calculate the global stiffness matrix of an axially loaded bar
def calculate_Kg(theta, L, E, A, I):
    """
    Calculate the global stiffness matrix for an axially loaded bar
    """

    c = math.cos(theta)
    s = math.sin(theta)
    c2 = c**2
    s2 = s**2
    L2 = L**2

    # Top left quadrant of global stiffness matrix
    K11 = (E / L) * np.array(
        [
            [A * c2 + (12 * I * s2) / L2, s * c * (A - 12 * I / L2), -6 * I * s / L],
            [s * c * (A - 12 * I / L2), A * s2 + (12 * I * c2) / L2, 6 * I * c / L],
            [-6 * I * s / L, 6 * I * c / L, 4 * I],
        ]
    )

    # Top right quadrant of global stiffness matrix
    K12 = (E / L) * np.array(
        [
            [-(A * c2 + (12 * I * s2) / L2), -s * c * (A - 12 * I / L2), -6 * I * s / L],
            [-s * c * (A - 12 * I / L2), -(A * s2 + (12 * I * c2) / L2), 6 * I * c / L],
            [6 * I * s / L, -6 * I * c / L, 2 * I],
        ]
    )

    # Bottom left quadrant of global stiffness matrix
    K21 = K12.T

    # Bottom right quadrant of global stiffness matrix
    K22 = (E / L) * np.array(
        [
            [A * c2 + (12 * I * s2) / L2, s * c * (A - 12 * I / L2), 6 * I * s / L],
            [s * c * (A - 12 * I / L2), A * s2 + (12 * I * c2) / L2, -6 * I * c / L],
            [6 * I * s / L, -6 * I * c / L, 4 * I],
        ]
    )

    return [K11, K12, K21, K22]


def calc_DOF(members):
    DOF = len(set(members.flatten())) * 3  # Total number of degrees of freedom in the problem
    return DOF


def build_K(members, members_area, orientations, lengths, E, members_moment):
    n_DOF = calc_DOF(members)
    Kp = np.zeros([n_DOF, n_DOF])  # Initialise the primary stiffness matrix
    for n, mbr in enumerate(members):
        # note that enumerate adds a counter to an iterable (n)

        # Calculate the quadrants of the global stiffness matrix for the member
        theta = orientations[n]
        L = lengths[n]
        A = members_area[n]
        I = members_moment[n]
        [K11, K12, K21, K22] = calculate_Kg(theta, L, E, A, I)

        node_i = mbr[0]  # Node number for node i of this member
        node_j = mbr[1]  # Node number for node j of this member

        # Primary stiffness matrix indices associated with each node
        # i.e. node 1 occupies indices 0 and 1 (accessed in Python with [0:2])

        i = 3 * node_i
        j = 3 * node_j

        Kp[i : i + 3, i : i + 3] = Kp[i : i + 3, i : i + 3] + K11
        Kp[j : j + 3, j : j + 3] = Kp[j : j + 3, j : j + 3] + K22
        Kp[i : i + 3, j : j + 3] = Kp[i : i + 3, j : j + 3] + K12
        Kp[j : j + 3, i : i + 3] = Kp[j : j + 3, i : i + 3] + K21

    return Kp


def extract_member_K(member, K):
    node_i = member[0]  # Node number for node i of this member
    node_j = member[1]  # Node number for node j of this member

    # Primary stiffness matrix indices associated with each node
    # i.e. node 1 occupies indices 0 and 1 (accessed in Python with [0:2])

    i = 3 * node_i
    j = 3 * node_j

    k_local = np.zeros((6, 6))

    k_local[0:3, 0:3] = K[i : i + 3, i : i + 3]
    k_local[3:6, 3:6] = K[j : j + 3, j : j + 3]
    k_local[0:3, 3:6] = K[i : i + 3, j : j + 3]
    k_local[3:6, 0:3] = K[j : j + 3, i : i + 3]

    return k_local


def calc_KG_to_mat(KG):
    K = np.zeros((6, 6))
    K[:3, :3] = KG[0]
    K[3:, 3:] = KG[3]
    K[:3, 3:] = KG[1]
    K[3:, :3] = KG[2]
    return K


# def calculate_del_Kg__del_area(theta, mag, E, A):
#     """
#     Calculate the global stiffness matrix for an axially loaded bar
#     """

#     c = math.cos(theta)
#     s = math.sin(theta)

#     K11 = (E / mag) * np.array([[c**2, c * s], [c * s, s**2]])  # Top left quadrant of global stiffness matrix
#     K12 = (E / mag) * np.array(
#         [[-(c**2), -c * s], [-c * s, -(s**2)]]
#     )  # Top right quadrant of global stiffness matrix
#     K21 = (E / mag) * np.array(
#         [[-(c**2), -c * s], [-c * s, -(s**2)]]
#     )  # Bottom left quadrant of global stiffness matrix
#     K22 = (E / mag) * np.array([[c**2, c * s], [c * s, s**2]])  # Bottom right quadrant of global stiffness matrix

#     return [K11, K12, K21, K22]


# def build_del_K__del_area(members, members_area, orientations, lengths, E):
#     n_DOF = calc_DOF(members)
#     Kp = np.zeros([n_DOF, n_DOF])  # Initialise the primary stiffness matrix
#     d_Kp__d_area = np.zeros((n_DOF, n_DOF, len(members)))
#     for dimension in range(len(members)):
#         for n, mbr in enumerate(members):
#             # note that enumerate adds a counter to an iterable (n)

#             # Calculate the quadrants of the global stiffness matrix for the member
#             theta = orientations[n]
#             L = lengths[n]
#             A = members_area[n]
#             if n == dimension:
#                 [K11, K12, K21, K22] = calculate_del_Kg__del_area(theta, L, E, A)
#             else:
#                 continue

#             node_i = mbr[0]  # Node number for node i of this member
#             node_j = mbr[1]  # Node number for node j of this member

#             # Primary stiffness matrix indices associated with each node
#             # i.e. node 1 occupies indices 0 and 1 (accessed in Python with [0:2])

#             i = 2 * node_i
#             j = 2 * node_j

#             d_Kp__d_area[i : i + 2, i : i + 2, dimension] = Kp[i : i + 2, i : i + 2] + K11
#             d_Kp__d_area[j : j + 2, j : j + 2, dimension] = Kp[j : j + 2, j : j + 2] + K22
#             d_Kp__d_area[i : i + 2, j : j + 2, dimension] = Kp[i : i + 2, j : j + 2] + K12
#             d_Kp__d_area[j : j + 2, i : i + 2, dimension] = Kp[j : j + 2, i : i + 2] + K21

#     return d_Kp__d_area
