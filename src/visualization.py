import matplotlib.pyplot as plt  # Plotting functionality
import member_reactions as mr
import numpy as np
import cv2
import os


def plot_structure(members, nodes, dir, name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = dir + name + ".png"

    fig = plt.figure()
    axes = fig.add_subplot(111)
    fig.gca().set_aspect("equal", adjustable="box")

    # Plot members
    for i, member in enumerate(members):
        member = member.astype("int")
        node_s = nodes[member[0]]  # Node number for node i of this member
        node_e = nodes[member[1]]  # Node number for node j of this member

        axes.plot([node_s[0], node_e[0]], [node_s[1], node_e[1]], "b")  # Member

    # Plot nodes
    for node in nodes:
        axes.plot([node[0]], [node[1]], "bo")

    axes.set_xlabel("Distance (m)")
    axes.set_ylabel("Distance (m)")
    axes.set_title("Structure to analyse")
    axes.grid()
    plt.savefig(path, bbox_inches="tight")


def plot_deflection(
    members,
    nodes,
    rotations,
    lengths,
    deflections,
    members_depth,
    members_width,
    x_fac,
    path,
    fname,
    valid_flags=None,
):
    if not os.path.exists(path):
        os.makedirs(path)
    members_area = members_depth * members_width
    members_area = members_area * 5
    fig = plt.figure()
    axes = fig.add_subplot(111)
    fig.gca().set_aspect("equal", adjustable="box")
    if valid_flags is None:
        valid_flags = np.ones(len(member))
    # Plot members
    for i, member in enumerate(members):
        bin_count = len(deflections[i])
        L = lengths[i]

        node_s = nodes[member[0]]
        node_e = nodes[member[1]]

        axes.plot(
            [node_s[0], node_e[0]], [node_s[1], node_e[1]], "green", lw=0.75
        )  # Member

        deflection_g = deflections[i]
        x_cor = [
            (node_e[0] - node_s[0]) * k / (bin_count - 1) + node_s[0]
            for k in range(bin_count)
        ]
        y_cor = [
            (node_e[1] - node_s[1]) * k / (bin_count - 1) + node_s[1]
            for k in range(bin_count)
        ]

        x_coor = x_cor + deflection_g[:, 0] * x_fac
        y_coor = y_cor + deflection_g[:, 1] * x_fac

        if valid_flags[i]:
            axes.plot(
                x_coor,
                y_coor,
                "b",
                lw=members_area[i],
            )
        else:
            axes.plot(
                x_coor,
                y_coor,
                "r",
                lw=members_area[i],
            )
    axes.set_xlabel("Distance (m)")
    axes.set_ylabel("Distance (m)")
    axes.set_title("Deflected shape")
    axes.grid()
    plt.savefig(f"{path}/{fname}.png", bbox_inches="tight")
    plt.close()
    # plt.show()


def zero_pad_image(image, N):
    original_height, original_width = image.shape

    # Calculate the padding required to make it N*N
    pad_height = max(0, N - original_height)
    pad_width = max(0, N - original_width)

    # Add zero padding to the image
    padded_image = np.pad(
        image,
        (
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
        ),
        mode="constant",
        constant_values=0,
    )

    # Resize the padded image to N*N
    # resized_image = cv2.resize(padded_image, (N, N))
    return padded_image
