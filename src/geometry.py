import numpy as np
import math


# Define a function to calculate member rotation and length
def member_rotation(member, nodes):
    node_i = member[0]  # Node number for node i of this member
    node_j = member[1]  # Node number for node j of this member

    xi = nodes[node_i][0]  # x-coord for node i
    yi = nodes[node_i][1]  # y-coord for node i
    xj = nodes[node_j][0]  # x-coord for node j
    yj = nodes[node_j][1]  # y-coord for node j

    # Angle of member with respect to horizontal axis

    dx = xj - xi  # x-component of vector along member
    dy = yj - yi  # y-component of vector along member
    L = math.sqrt(dx**2 + dy**2)  # Magnitude of vector (length of member)
    member_vector = np.array([dx, dy])  # Member represented as a vector

    # Need to capture quadrant first then appropriate reference axis and offset angle
    if dx > 0 and dy == 0:
        theta = 0
    elif dx == 0 and dy > 0:
        theta = math.pi / 2
    elif dx < 0 and dy == 0:
        theta = math.pi
    elif dx == 0 and dy < 0:
        theta = 3 * math.pi / 2
    elif dx > 0 and dy > 0:
        # 0<theta<90
        ref_vector = np.array([1, 0])  # Vector describing the positive x-axis
        theta = math.acos(ref_vector.dot(member_vector) / (L))  # Standard formula for the angle between two vectors
    elif dx < 0 and dy > 0:
        # 90<theta<180
        ref_vector = np.array([0, 1])  # Vector describing the positive y-axis
        theta = (math.pi / 2) + math.acos(
            ref_vector.dot(member_vector) / (L)
        )  # Standard formula for the angle between two vectors
    elif dx < 0 and dy < 0:
        # 180<theta<270
        ref_vector = np.array([-1, 0])  # Vector describing the negative x-axis
        theta = math.pi + math.acos(
            ref_vector.dot(member_vector) / (L)
        )  # Standard formula for the angle between two vectors
    else:
        # 270<theta<360
        ref_vector = np.array([0, -1])  # Vector describing the negative y-axis
        theta = (3 * math.pi / 2) + math.acos(
            ref_vector.dot(member_vector) / (L)
        )  # Standard formula for the angle between two vectors

    return [theta, L]


def calc_rotation_length(members, nodes):
    # Calculate rotation and length for each member and store
    rotations = np.array([])  # Initialise an array to hold rotations
    lengths = np.array([])  # Initialise an array to hold lengths
    for member in members:
        [angle, length] = member_rotation(member, nodes)
        rotations = np.append(rotations, angle)
        lengths = np.append(lengths, length)
    return rotations, lengths
