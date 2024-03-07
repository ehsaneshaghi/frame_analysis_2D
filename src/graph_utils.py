import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
import json
import pickle
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import math
import visualization as vs
import stiffness as st


class GraphHandler:
    def __init__(self, conf):
        self.num_cols = conf["num_cols"]
        self.num_rows = conf["num_rows"]
        self.transfer_row = conf["transfer_row"]
        self.horizontal_scale = conf["horizontal_scale"]
        self.vertical_scale = conf["vertical_scale"]
        self.possible_columns_up = conf.get(
            "possible_columns_up", np.arange(self.num_cols)
        )
        self.possible_columns_down = conf.get(
            "possible_columns_down", np.arange(self.num_cols)
        )
        self.distance_lower_bound = conf.get("distance_lower_bound", 4)
        self.distance_upper_bound = conf.get("distance_upper_bound", 10)

    def generate_graph(self, mode="train", analysis="vertical"):
        G = nx.grid_2d_graph(self.num_cols, self.num_rows)
        self.set_coo(G)
        self.set_pos(G)
        if mode == "train":
            sample_columns_up = self.sample_cols_train(self.possible_columns_up)
            sample_columns_down = self.sample_cols_train(self.possible_columns_down)
        else:
            sample_columns_up = self.sample_cols_test(self.possible_columns_up)
            sample_columns_down = self.sample_cols_test(self.possible_columns_down)

        self.remove_cols(G, sample_columns_up, sample_columns_down)
        self.remove_foundation_beams(G)
        self.remove_disconnected_nodes(G)
        G = GraphHandler.node_tuple_2_index(
            G
        )  # relabel nodes only after that unnecessary nodes are removed
        self.set_edge_size(G)
        self.set_DOF(G)
        self.set_foundation_flag(G)
        self.set_loads(G, analysis)
        self.set_rotation(G)
        self.set_column_flag(G)
        self.set_distance(G)
        self.set_cant_flag(G)
        self.set_step_size(G)
        return G

    def simplify_graph(self, G, analysis="vertical"):
        Gs = self.unify_edges(G.copy())
        Gs = self.node_tuple_2_index(Gs)  # relabel nodes before adding hops
        if analysis == "V" or analysis == "C":
            self.add_hop_edges(Gs)
        self.set_rotation(Gs)
        self.set_column_flag(Gs)
        self.set_distance(Gs)
        self.set_edge_size(Gs)
        self.set_foundation_flag(Gs)
        self.set_transfer_flag(Gs)
        self.set_cant_flag(Gs)
        self.set_step_size(Gs)
        self.set_level(Gs)
        self.set_column_l2r_ratio(Gs)
        return Gs

    def set_pos(self, G):
        # Ensure all nodes have positions defined
        for node in G.nodes():
            (x, y) = node
            G.nodes[node]["pos"] = [x * self.horizontal_scale, y * self.vertical_scale]

    def set_edge_size(self, G):
        for edge in G.edges():
            (x1, y1) = G.nodes[edge[0]]["coo"]
            (x2, y2) = G.nodes[edge[1]]["coo"]
            if y1 == y2 and y1 == self.transfer_row:
                G[edge[0]][edge[1]]["D"] = 0.8
                G[edge[0]][edge[1]]["W"] = 0.4

            elif y1 == y2 and y1 > self.transfer_row:
                G[edge[0]][edge[1]]["D"] = 0.25
                G[edge[0]][edge[1]]["W"] = 0.4

            elif y1 == y2 and y1 < self.transfer_row:
                G[edge[0]][edge[1]]["D"] = 0.25
                G[edge[0]][edge[1]]["W"] = 0.4

            elif x1 == x2 and max(y1, y2) > 2:
                G[edge[0]][edge[1]]["D"] = 0.8
                G[edge[0]][edge[1]]["W"] = 0.3
            else:
                G[edge[0]][edge[1]]["D"] = 1
                G[edge[0]][edge[1]]["W"] = 0.3

    def modify_edge_size(self, G, row_noise, column_noise, col_compressible=False):
        for edge in G.edges():
            (x1, y1) = G.nodes[edge[0]]["coo"]
            (x2, y2) = G.nodes[edge[1]]["coo"]
            if y1 == y2:
                if y1 < self.transfer_row:
                    G[edge[0]][edge[1]]["D"] = (
                        G[edge[0]][edge[1]]["D"]
                        + G[edge[0]][edge[1]]["D"] * row_noise[0, 0]
                    )
                    G[edge[0]][edge[1]]["W"] = (
                        G[edge[0]][edge[1]]["W"]
                        + G[edge[0]][edge[1]]["W"] * row_noise[0, 1]
                    )
                elif y1 == self.transfer_row:
                    G[edge[0]][edge[1]]["D"] = (
                        G[edge[0]][edge[1]]["D"]
                        + G[edge[0]][edge[1]]["D"] * row_noise[1, 0]
                    )
                    G[edge[0]][edge[1]]["W"] = (
                        G[edge[0]][edge[1]]["W"]
                        + G[edge[0]][edge[1]]["W"] * row_noise[1, 1]
                    )
                else:
                    G[edge[0]][edge[1]]["D"] = (
                        G[edge[0]][edge[1]]["D"]
                        + G[edge[0]][edge[1]]["D"] * row_noise[2, 0]
                    )
                    G[edge[0]][edge[1]]["W"] = (
                        G[edge[0]][edge[1]]["W"]
                        + G[edge[0]][edge[1]]["W"] * row_noise[2, 1]
                    )
            elif col_compressible:
                if x1 == x2 and min(y1, y2) >= self.transfer_row:
                    G[edge[0]][edge[1]]["D"] = (
                        G[edge[0]][edge[1]]["D"]
                        + G[edge[0]][edge[1]]["D"] * column_noise[0, 0]
                    )
                    G[edge[0]][edge[1]]["W"] = (
                        G[edge[0]][edge[1]]["W"]
                        + G[edge[0]][edge[1]]["W"] * column_noise[0, 1]
                    )

                elif x1 == x2 and min(y1, y2) < self.transfer_row:
                    G[edge[0]][edge[1]]["D"] = (
                        G[edge[0]][edge[1]]["D"]
                        + G[edge[0]][edge[1]]["D"] * column_noise[1, 0]
                    )
                    G[edge[0]][edge[1]]["W"] = (
                        G[edge[0]][edge[1]]["W"]
                        + G[edge[0]][edge[1]]["W"] * column_noise[1, 1]
                    )
            else:
                G[edge[0]][edge[1]]["D"] = 5
                G[edge[0]][edge[1]]["W"] = 5

        return G

    def set_coo(self, G):
        for node in G.nodes():
            G.nodes[node]["coo"] = node

    def set_DOF(self, G):
        for node in G.nodes():
            x, y = G.nodes[node]["coo"]
            if y == 0:
                G.nodes[node]["DOF"] = [0, 0, 0]
                G.nodes[node]["free"] = [0]
            else:
                G.nodes[node]["DOF"] = [1, 1, 1]
                G.nodes[node]["free"] = [1]

    def set_loads(self, G, analysis):
        if analysis == "V":
            for node in G.nodes():
                G.nodes[node]["load"] = [
                    0,
                    -(80 / self.num_cols) * 1e3,
                    0,
                ]  # 8000 KN gravity load distributed
        elif analysis == "L":  # Lateral analysis
            for node in G.nodes():
                x, y = G.nodes[node]["pos"]
                if x == 0 and y > self.transfer_row:
                    G.nodes[node]["load"] = [
                        7 * 1e3 * (y - self.transfer_row),
                        0,
                        0,
                    ]  # 8000 KN gravity load distributed
                else:
                    G.nodes[node]["load"] = [0, 0, 0]
        else:
            for node in G.nodes():
                x, y = G.nodes[node]["pos"]
                if x == 0 and y > self.transfer_row:
                    G.nodes[node]["load"] = [
                        7 * 1e3 * (y - self.transfer_row),
                        -(80 / self.num_cols) * 1e3,
                        0,
                    ]  # 8000 KN gravity load distributed
                else:
                    G.nodes[node]["load"] = [0, -(80 / self.num_cols) * 1e3, 0]

    def set_rotation(self, G):
        for u, v in G.edges():
            pos_u = G.nodes[u]["pos"]
            pos_v = G.nodes[v]["pos"]

            # Calculate the direction vector (dx, dy) from u to v
            dx = pos_v[0] - pos_u[0]
            dy = pos_v[1] - pos_u[1]

            # Calculate the rotation angle (in radians) using atan2
            rotation_angle = math.atan2(dy, dx)
            G[u][v]["rotation"] = rotation_angle

    def add_hop_edges(self, G):
        # Update edge attributes for all edges in the graph
        for edge in G.edges():
            if "real" not in G[edge[0]][edge[1]]:
                G[edge[0]][edge[1]]["real"] = True

        for node_i in G.nodes():
            x1, y1 = G.nodes[node_i]["coo"]
            if y1 == self.transfer_row:
                for node_j in G.nodes():
                    x2, y2 = G.nodes[node_j]["coo"]
                    if (
                        y2 > (self.transfer_row + 1)
                        and x1 == x2
                        and G.degree(node_j) > 2
                    ):
                        G.add_edge(node_i, node_j, real=False)

    def set_foundation_flag(self, G):
        for u, v in G.edges():
            free_u = G.nodes[u]["free"]
            free_v = G.nodes[v]["free"]
            if free_u == [0] or free_v == [0]:
                G[u][v]["foundation"] = 1
            else:
                G[u][v]["foundation"] = 0

    def set_level(self, G):
        for u, v in G.edges():
            level_u = G.nodes[u]["coo"][1]
            level_v = G.nodes[v]["coo"][1]
            G[u][v]["level"] = (level_u + level_v) / 2
            G[u][v]["level_ratio"] = (max(level_u, level_v) - self.transfer_row) / (
                self.num_rows - self.transfer_row
            )

    def set_column_l2r_ratio(self, G):
        for u, v in G.edges():
            col_u = G.nodes[u]["coo"][0]
            col_v = G.nodes[v]["coo"][0]
            G[u][v]["col_l2r_ratio"] = (col_u + col_v) / self.num_cols

    def set_transfer_flag(self, G):
        for u, v in G.edges():
            x1, y1 = G.nodes[u]["coo"]
            x2, y2 = G.nodes[v]["coo"]
            if y1 == y2 and y1 == self.transfer_row:
                G[u][v]["transfer"] = 1
            else:
                G[u][v]["transfer"] = 0

    def set_cant_flag(self, G):
        for u, v in G.edges():
            if (not G[u][v]["column"]) and (np.minimum(G.degree[u], G.degree[v]) == 1):
                G[u][v]["cant"] = 1
            else:
                G[u][v]["cant"] = 0

    def set_step_size(self, G):
        for u, v in G.edges():
            if G[u][v]["cant"]:
                G[u][v]["step_size"] = 0.05
            else:
                G[u][v]["step_size"] = 0.05

    def set_column_flag(self, G):
        for u, v in G.edges():
            if np.abs(np.sin(G[u][v]["rotation"])) > 0.9:
                G[u][v]["column"] = True
            else:
                G[u][v]["column"] = False

    def set_distance(self, G):
        for u, v in G.edges():
            pos_u = G.nodes[u]["pos"]
            pos_v = G.nodes[v]["pos"]
            dx = pos_v[0] - pos_u[0]
            dy = pos_v[1] - pos_u[1]
            G[u][v]["dist"] = np.sqrt(dx**2 + dy**2)

    @staticmethod
    def node_tuple_2_index(G):
        mapping = {node: index for index, node in enumerate(G.nodes())}
        # node_mapping = {(i, j): i * self.num_cols + j for i in range(self.num_cols) for j in range(self.num_rows)}
        G_relabel = nx.relabel_nodes(G, mapping)
        return G_relabel

    def calc_col_distance(self, cols):
        cols = np.sort(cols)

        a = np.diff(cols, prepend=0)
        b = np.diff(cols, append=self.num_cols)
        max_diff = np.maximum(a, b).astype("float")

        a = np.diff(cols, prepend=-1000)
        b = np.diff(cols, append=1000)
        min_diff = np.minimum(a, b).astype("float")

        max_diff[0] = max_diff[0] + 3 / self.horizontal_scale
        max_diff[-1] = max_diff[-1] + 3 / self.horizontal_scale
        return max_diff * self.horizontal_scale, min_diff * self.horizontal_scale

    def sample_cols_train(self, cols):
        while True:
            max_dist, min_dist = self.calc_col_distance(cols)
            distances = np.sqrt(max_dist * min_dist)
            new_cols = np.sort(cols)
            while True:
                if min(min_dist) >= self.distance_lower_bound:
                    if np.random.rand() > (self.distance_lower_bound * 1.2) / min(
                        min_dist
                    ):
                        break
                keep_prob = np.exp(distances) / sum(np.exp(distances))
                new_cols = np.sort(
                    np.random.choice(
                        new_cols, size=len(new_cols) - 1, replace=False, p=keep_prob
                    )
                )
                if len(new_cols) < 2:
                    break
                max_dist, min_dist = self.calc_col_distance(new_cols)
                distances = np.sqrt(max_dist * min_dist)
            if max(max_dist) < self.distance_upper_bound:
                break

        return new_cols

    def sample_cols_test(self, cols):
        while True:
            max_dist, min_dist = self.calc_col_distance(cols)
            distances = np.sqrt(max_dist * min_dist)
            new_cols = np.sort(cols)
            while True:
                if min(min_dist) >= self.distance_lower_bound:
                    if np.random.rand() > (self.distance_lower_bound * 1.3) / min(
                        min_dist
                    ):
                        break
                keep_prob = np.exp(distances) / sum(np.exp(distances))
                new_cols = np.sort(
                    np.random.choice(
                        new_cols, size=len(new_cols) - 1, replace=False, p=keep_prob
                    )
                )
                if len(new_cols) < 2:
                    break
                max_dist, min_dist = self.calc_col_distance(new_cols)
                distances = np.sqrt(max_dist * min_dist)
            if max(max_dist) < self.distance_upper_bound:
                break

        return new_cols

    def remove_cols(self, G, random_cols_up, random_cols_down):
        edges_to_remove = []
        for edge in G.edges():
            (x1, y1) = G.nodes[edge[0]]["coo"]
            (x2, y2) = G.nodes[edge[1]]["coo"]
            if y1 != y2:
                if (y1 >= self.transfer_row and x1 not in random_cols_up) or (
                    y1 < self.transfer_row and x1 not in random_cols_down
                ):
                    edges_to_remove.append(edge)

        G.remove_edges_from(edges_to_remove)
        return G

    def unify_edges(self, G):
        while True:
            degree_two_nodes = [
                node
                for node in G.nodes()
                if G.degree(node) == 2
                and G.nodes()[node]["coo"][0] != 0
                and G.nodes()[node]["coo"][0] != self.num_cols - 1
            ]
            if not degree_two_nodes:
                break

            for node in degree_two_nodes:
                neighbors = list(G.neighbors(node))
                if len(neighbors) == 2:
                    y, z = neighbors
                    if G.nodes()[y]["coo"][1] == G.nodes()[z]["coo"][1]:
                        G.remove_node(node)  # Remove node x
                        if not G.has_edge(y, z):
                            G.add_edge(
                                y, z
                            )  # Add edge (y, z) if it doesn't exist already

        return G

    def remove_disconnected_nodes(self, G):
        # # Identify nodes with degree 0
        nodes_to_remove = [node for node in G.nodes() if G.degree(node) == 0]
        # # Remove nodes with degree 0
        G.remove_nodes_from(nodes_to_remove)
        return G

    def remove_foundation_beams(self, G):
        edges_to_remove = []
        for edge in G.edges():
            (x1, y1) = G.nodes[edge[0]]["coo"]
            (x2, y2) = G.nodes[edge[1]]["coo"]
            if y1 == 0 and y2 == 0:
                edges_to_remove.append(edge)

        G.remove_edges_from(edges_to_remove)
        return G

    def set_nodal_displacement(self, G, UG):
        UG = UG.reshape(-1, 3)
        for node in G.nodes():
            G.nodes[node]["U"] = UG[node]
        return G

    def transfer_nodal_displacement(self, G, Gs):
        for node_s in Gs.nodes():
            x1, y1 = Gs.nodes[node_s]["coo"]
            for node in G.nodes():
                x2, y2 = G.nodes[node]["coo"]

                if x1 == x2 and y1 == y2:
                    Gs.nodes[node_s]["U"] = G.nodes[node]["U"]
        return Gs

    def set_member_deflection(self, G, Gs):
        for us, vs in Gs.edges():
            x1, y1 = Gs.nodes[us]["coo"]
            x2, y2 = Gs.nodes[vs]["coo"]
            Gs[us][vs]["deflection"] = np.zeros((0, 3))
            for u in G.nodes():
                x3, y3 = G.nodes[u]["coo"]
                if (min(x1, x2) <= x3 <= max(x1, x2)) and (
                    min(y1, y2) <= y3 <= max(y1, y2)
                ):
                    Gs[us][vs]["deflection"] = np.vstack(
                        (Gs[us][vs]["deflection"], G.nodes[u]["U"])
                    )
        return Gs

    def max_distance_to_line(self, x, y):
        # Calculate the slope and intercept of the line connecting the two endpoints
        slope = (y[-1] - y[0]) / (x[-1] - x[0])
        intercept = y[0] - slope * x[0]

        # Calculate the distance of each point to the line using the formula for distance
        distances = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)
        # Return the maximum distance
        return np.max(distances)

    def max_distance_to_cant(self, x, y, slope_l, slop_r):
        # Calculate the slope and intercept of the line connecting the two endpoints
        # slope = (y[1] - y[0]) / (x[1] - x[0])
        slope = slope_l
        intercept = y[0] - slope * x[0]

        # Calculate the distance of each point to the line using the formula for distance
        distances = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)
        dist1 = np.max(distances)

        # slope = (y[-2] - y[-1]) / (x[-2] - x[-1])
        slope = slop_r
        intercept = y[-1] - slope * x[-1]

        # Calculate the distance of each point to the line using the formula for distance
        distances = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)
        dist2 = np.max(distances)

        return np.maximum(dist1, dist2)

    def calc_ver_deflection(self, Gs):
        for u, v in Gs.edges():
            if "valid" not in Gs[u][v]:
                Gs[u][v]["valid"] = True
            if "normal_deflection" not in Gs[u][v]:
                Gs[u][v]["normal_deflection"] = 0
            if np.abs(np.cos(Gs[u][v]["rotation"])) > 0.9:  # slab
                if Gs.nodes[u]["coo"][0] < Gs.nodes[v]["coo"][0]:
                    x = np.arange(
                        0,
                        Gs[u][v]["dist"],
                        Gs[u][v]["dist"] / Gs[u][v]["deflection"].shape[0],
                    )
                else:
                    print("RIDI")
                    x = np.arange(0, Gs[u][v]["dist"], -Gs[u][v]["step_size"])
                a = Gs[u][v]["deflection"]
                x = x + Gs[u][v]["deflection"][:, 0]
                y = Gs[u][v]["deflection"][:, 1]
                theta = Gs[u][v]["deflection"][:, 2]
                if Gs[u][v]["cant"]:
                    Gs[u][v]["normal_deflection"] = (
                        self.max_distance_to_cant(x, y, theta[0], theta[-1])
                        / Gs[u][v]["dist"]
                    )
                else:
                    Gs[u][v]["normal_deflection"] = (
                        self.max_distance_to_line(x, y) / Gs[u][v]["dist"]
                    )
                Gs[u][v]["valid"] = Gs[u][v]["normal_deflection"] < 1 / 2e3

        return Gs

    def calc_drift(self, Gs):
        for us, vs in Gs.edges():
            if "valid" not in Gs[us][vs]:
                Gs[us][vs]["valid"] = True
            if "drift" not in Gs[us][vs]:
                Gs[us][vs]["drift"] = 0
            if np.cos(Gs[us][vs]["rotation"]) < 0.1:
                Gs[us][vs]["drift"] = (
                    np.abs(Gs.nodes[us]["U"][0] - Gs.nodes[vs]["U"][0])
                    / Gs[us][vs]["dist"]
                )
                Gs[us][vs]["valid"] = Gs[us][vs]["drift"] < 1 / 5e2
        return Gs

    def visualize(self, G):
        # Extract node positions for visualization
        pos = nx.get_node_attributes(G, "pos")
        # Visualize the graph
        fig = plt.figure(figsize=(5, 5))
        nx.draw(
            G, pos, with_labels=True, node_size=5, node_color="lightblue", font_size=0
        )
        plt.title(f"Multi-story frame")
        plt.savefig(f"G.png", bbox_inches="tight")

    @staticmethod
    def graph_to_array(G):

        v = {}
        # Create an array to store the degree of freedom for each node
        v["dof_array"] = np.array([G.nodes[node]["DOF"] for node in G.nodes()])
        v["restrained_dof"] = np.where(v["dof_array"].reshape(-1, 1) == 0)[0]

        # Create a list of edges as tuples of node indices
        v["edges"] = np.array([(u, v) for u, v in G.edges()])

        v["rotations"] = np.array([G[u][v]["rotation"] for u, v in G.edges()])
        v["lenghts"] = np.array([G[u][v]["dist"] for u, v in G.edges()])

        v["depths"] = np.array([G[u][v]["D"] for u, v in G.edges()])

        v["widths"] = np.array([G[u][v]["W"] for u, v in G.edges()])

        v["node_positions"] = np.array([G.nodes[node]["pos"] for node in G.nodes()])

        loads = np.array([G.nodes[node]["load"] for node in G.nodes()])
        v["loads"] = loads.reshape(-1, 1).ravel()

        v["step_sizes"] = np.array([G[u][v]["step_size"] for u, v in G.edges()])

        return v

    def save_graph(self, G, directory):
        file_name = G.graph["name"]
        file_name = f"{file_name}.pkl"

        file_path = os.path.join(directory, file_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "wb") as file:
            pickle.dump(G, file)

    def save_pyg_graphs(self, nx_graphs, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = dir + "pyg_graphs.pkl"

        pyg_data_list = []
        for G in nx_graphs:
            # Convert NetworkX graph to PyG Data
            G = G.to_directed()
            edge_index = torch.tensor(list(G.edges())).T
            G = nx.node_link_data(G)

            nodes = G["nodes"]
            edges = G["links"]

            # Extract edge-level attributes and assign them to PyG Data
            edge_attr = torch.tensor(
                [
                    [
                        edge["rotation"] < 0.01,
                        edge["D"] ** 3 * edge["W"],
                        edge["D"] * edge["W"],
                        edge["dist"],
                    ]
                    for edge in edges
                ],
                dtype=torch.float,
            )

            # Concatenate node attributes (positions and DOF) into a single tensor
            x = torch.tensor(
                [[node["free"][0], node["pos"][1]] for node in nodes], dtype=torch.float
            )

            # Global score (label)
            y = torch.tensor([edge["valid"] for edge in edges])
            edge_attr[:, 1:] = (edge_attr[:, 1:] - edge_attr[:, 1:].mean(axis=0)) / (
                edge_attr[:, 1:].std(axis=0)
            )
            x = (x - x.mean(axis=0)) / (x.std(axis=0))

            data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=y)

            pyg_data_list.append(data)

        # Save the PyG Data objects
        with open(path, "wb") as f:
            pickle.dump(pyg_data_list, f)

    @staticmethod
    def save_pyg_line_graphs_vertical(nx_graphs, dir, name, has_label=True):
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = dir + name

        pyg_data_list = []
        for G in nx_graphs:
            # Convert NetworkX graph to PyG Data
            G = G.to_directed()
            edge_index = torch.tensor(list(G.edges())).T
            G = nx.node_link_data(G)

            nodes = G["nodes"]

            # Extract edge-level attributes and assign them to PyG Data
            x = torch.tensor(
                [
                    [
                        node["rotation"] < 0.01,
                        node["cant"],
                        node["foundation"],
                        node["transfer"],
                        node["real"],
                        node["D"] ** 3 * node["W"] * np.cos(node["rotation"]),
                        node["dist"] * np.cos(node["rotation"]),
                    ]
                    for node in nodes
                ],
                dtype=torch.float,
            )
            weight = sum(
                [
                    (
                        node["D"] * node["W"] * node["dist"]
                        if node["rotation"] < 0.01
                        else 0.3 * node["dist"] * int(node["real"])
                    )
                    for node in nodes
                ]
            )
            if has_label:
                y = torch.tensor([node["valid"] for node in nodes])
                slab_def = torch.tensor(
                    [node["def_perp_diff"] / node["dist"] for node in nodes]
                )
                data = Data(
                    x=x, edge_index=edge_index, y=y, weight=weight, slab_def=slab_def
                )
            else:
                data = Data(x=x, edge_index=edge_index, weight=weight)

            pyg_data_list.append(data)

        # Save the PyG Data objects
        with open(path, "wb") as f:
            pickle.dump(pyg_data_list, f)

    @staticmethod
    def save_pyg_line_graphs_lateral(nx_graphs, dir, name, has_label=True):
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = dir + name

        pyg_data_list = []
        for G in nx_graphs:
            # Convert NetworkX graph to PyG Data
            G = G.to_directed()
            edge_index = torch.tensor(list(G.edges())).T
            G = nx.node_link_data(G)

            nodes = G["nodes"]

            # Extract edge-level attributes and assign them to PyG Data
            x = torch.tensor(
                [
                    [
                        node["rotation"] > 0.01,
                        node["level"],
                        node["level_ratio"],
                        node["col_l2r_ratio"] * np.sin(node["rotation"]),
                        node["D"] ** 3 * node["W"] * np.sin(node["rotation"]),
                        node["D"] ** 3 * node["W"] * np.cos(node["rotation"]),
                        node["D"] * node["W"] * np.cos(node["rotation"]),
                        node["dist"] * np.cos(node["rotation"]),
                    ]
                    for node in nodes
                ],
                dtype=torch.float,
            )
            weight = sum([node["D"] * node["W"] * node["dist"] for node in nodes])
            if has_label:
                y = torch.tensor([node["valid"] for node in nodes])
                drift = torch.tensor([node["drift"] / node["dist"] for node in nodes])
                data = Data(x=x, edge_index=edge_index, y=y, weight=weight, drift=drift)
            else:
                data = Data(x=x, edge_index=edge_index, weight=weight)

            pyg_data_list.append(data)

        for data in pyg_data_list:
            data.x
        # Save the PyG Data objects
        with open(path, "wb") as f:
            pickle.dump(pyg_data_list, f)

    @staticmethod
    def save_pyg_line_graphs_combined(nx_graphs, dir, name, has_label=True):
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = dir + name

        pyg_data_list = []
        for G in nx_graphs:
            # Convert NetworkX graph to PyG Data
            G = G.to_directed()
            edge_index = torch.tensor(list(G.edges())).T
            G = nx.node_link_data(G)

            nodes = G["nodes"]

            # Extract edge-level attributes and assign them to PyG Data
            x = torch.tensor(
                [
                    [
                        node["rotation"] > 0.01,
                        node["cant"],
                        node["foundation"],
                        node["transfer"],
                        node["real"],
                        node["level_ratio"],
                        node["col_l2r_ratio"] * np.sin(node["rotation"]),
                        node["level"],
                        node["D"] ** 3 * node["W"] * np.sin(node["rotation"]),
                        node["D"] * node["W"] * np.sin(node["rotation"]),
                        node["D"] ** 3 * node["W"] * np.cos(node["rotation"]),
                        node["D"] * node["W"] * np.cos(node["rotation"]),
                        node["dist"] * np.cos(node["rotation"]),
                    ]
                    for node in nodes
                ],
                dtype=torch.float,
            )
            weight = sum(
                [
                    node["D"] * node["W"] * node["dist"] * int(node["real"])
                    for node in nodes
                ]
            )
            if has_label:
                y = torch.tensor([node["valid"] for node in nodes])
                drift = torch.tensor([node["drift"] for node in nodes])
                normal_def = torch.tensor([node["normal_deflection"] for node in nodes])
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    weight=weight,
                    drift=drift,
                    normal_def=normal_def,
                    name=G["graph"]["name"],
                )
            else:
                data = Data(
                    x=x, edge_index=edge_index, weight=weight, name=G["graph"]["name"]
                )

            pyg_data_list.append(data)

        for data in pyg_data_list:
            data.x
        # Save the PyG Data objects
        with open(path, "wb") as f:
            pickle.dump(pyg_data_list, f)

    @staticmethod
    def load_all_graphs(directory):
        graphs = []

        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                file_path = os.path.join(directory, filename)
                with open(file_path, "rb") as file:
                    graph = pickle.load(file)
                    if isinstance(graph, nx.Graph):
                        graphs.append(graph)

        return graphs

    def load_graphs(directory, file_names):
        graphs = []
        for name in file_names:
            file_path = os.path.join(directory, name)
            with open(file_path, "rb") as file:
                graph = pickle.load(file)
                if isinstance(graph, nx.Graph):
                    graphs.append(graph)

        return graphs

    def draw_graph(Gs, path, conf):
        v = GraphHandler.graph_to_array(Gs)
        name = Gs.graph["name"]
        valid_flags = [Gs[u][v]["valid"] for u, v in Gs.edges()]
        deflections = [Gs[u][v]["deflection"][:, :2] for u, v in Gs.edges()]
        members_real = [Gs[u][v]["real"] for u, v in Gs.edges()]
        vs.plot_deflection(
            v["edges"],
            v["node_positions"],
            v["rotations"],
            v["lenghts"],
            deflections,
            v["depths"],
            v["widths"],
            members_real,
            100,
            path,
            name,
            conf,
            valid_flags,
        )
