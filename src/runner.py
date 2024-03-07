import numpy as np
from environment import StructEnvironment
import nodal_reactions as nr
import graph_utils as gu
import json
import networkx as nx


env = StructEnvironment()
num_episodes = 3
top_stability_count = 20
top_weight_count = 3
mode = "test_3"
analysis = "C"
FEA = True
visualize = True
with open("config.json", "r") as jsonfile:
    conf = json.load(jsonfile)[mode]

conf["possible_columns_up"] = np.array(conf["possible_columns_up"])
conf["possible_columns_down"] = np.array(conf["possible_columns_down"])


episode = 0
line_graphs = []
graph_handler = gu.GraphHandler(conf)


while episode < num_episodes:
    if mode == "train":
        graph_handler.num_rows = conf["num_rows"] + np.random.randint(low=-3, high=3)

    G = graph_handler.generate_graph(mode=mode, analysis=analysis)
    Gs = graph_handler.simplify_graph(G, analysis=analysis)

    row_noise = np.random.beta(4, 4, size=(3, 2))
    column_noise = np.random.beta(4, 4, size=(2, 2))

    G = graph_handler.modify_edge_size(
        G, row_noise, column_noise, col_compressible=analysis != "V"
    )
    Gs = graph_handler.modify_edge_size(
        Gs, row_noise, column_noise, col_compressible=analysis != "V"
    )

    env_variables_dict = gu.GraphHandler.graph_to_array(G)
    env.set_attributes(env_variables_dict)

    # ---------------- analysis part ----------------------
    if FEA:
        UG = env.analyse()
        if UG is None or UG.max() > 0.2:
            print("invalid")
            continue
        G = graph_handler.set_nodal_displacement(G, UG)
        Gs = graph_handler.transfer_nodal_displacement(G, Gs)
        Gs = graph_handler.set_member_deflection(G, Gs)

        if analysis == "V":
            Gs = graph_handler.calc_ver_deflection(Gs)
        elif analysis == "L":
            Gs = graph_handler.calc_drift(Gs)
        else:
            Gs = graph_handler.calc_ver_deflection(Gs)
            Gs = graph_handler.calc_drift(Gs)
    # ---------------- analysis part ----------------------

    L = nx.line_graph(Gs)
    # Propagate node and edge attributes from the original graph to the line graph
    L.add_nodes_from((node, Gs.edges[node]) for node in L)
    L = gu.GraphHandler.node_tuple_2_index(L)

    print("step: ", episode)

    G.graph["name"] = str(episode)
    Gs.graph["name"] = str(episode)
    L.graph["name"] = str(episode)
    line_graphs.append(L)
    graph_handler.save_graph(G, f"data/{mode}/{analysis}/raw/graphs/")
    graph_handler.save_graph(Gs, f"data/{mode}/{analysis}/simplified/graphs/")
    graph_handler.save_graph(L, f"data/{mode}/{analysis}/line_simplified/graphs/")
    Gs = graph_handler.node_tuple_2_index(Gs)
    if visualize:
        gu.GraphHandler.draw_graph(Gs, "images/", conf)
    episode += 1

if analysis == "V":
    gu.GraphHandler.save_pyg_line_graphs_vertical(
        line_graphs,
        f"data/{mode}/{analysis}/line_simplified/dataset/",
        "pyg_line_graphs.pkl",
    )
elif analysis == "L":
    gu.GraphHandler.save_pyg_line_graphs_lateral(
        line_graphs,
        f"data/{mode}/{analysis}/line_simplified/dataset/",
        "pyg_line_graphs.pkl",
    )
else:
    gu.GraphHandler.save_pyg_line_graphs_combined(
        line_graphs,
        f"data/{mode}/{analysis}/line_simplified/dataset/",
        "pyg_line_graphs.pkl",
        has_label=FEA,
    )

# import torch
# import pandas as pd
# import torch.nn as nn
# import torch.nn.functional as F

# import numpy as np
# import pandas as pd

# import numpy as np
# import torch
# import json
# import pickle

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GraphConv
# from environment import StructEnvironment
# import graph_utils as gu

# if mode != "train":
#     test_name = mode

#     class SharedMPNN(nn.Module):
#         def __init__(
#             self, num_features, hidden_dim, output_dim, num_message_passing_steps
#         ):
#             super(SharedMPNN, self).__init__()

#             self.conv0 = nn.Linear(num_features, hidden_dim)
#             self.conv1 = GraphConv(hidden_dim, hidden_dim)
#             self.fc2 = nn.Linear(hidden_dim, output_dim)
#             self.num_message_passing_steps = num_message_passing_steps

#         def forward(self, data):
#             x, edge_index = data.x, data.edge_index

#             x = self.conv0(x)
#             x = F.relu(x)

#             for _ in range(self.num_message_passing_steps):
#                 x = self.conv1(x, edge_index)
#                 x = F.relu(x)

#             # Apply the final linear layer
#             x = self.fc2(x)

#             return x

#     dataset_path = f"./data/{test_name}/C/line_simplified/dataset/pyg_line_graphs.pkl"
#     with open(dataset_path, "rb") as f:
#         val_data = pickle.load(f)

#     # Create a DataLoader
#     batch_size = 32  # Set your desired batch size

#     mask_mean = torch.tensor([0.8893, 0.5544])
#     mask_std = torch.tensor([0.4109, 0.0996])

#     not_mask_mean = torch.tensor([0.2172, 0.3118, 4.2756])
#     not_mask_std = torch.tensor([0.4307, 0.1982, 1.9530])

#     def normalize(data_list):
#         for data in data_list:
#             mask = data.x[:, 0] == 1
#             data.x[mask, 8:10] = (data.x[mask, 8:10] - mask_mean) / mask_std
#             data.x[~mask, 10:] = (data.x[~mask, 10:] - not_mask_mean) / not_mask_std
#         return data_list

#     val_data = normalize(val_data)

#     # Initialize the model
#     num_node_features = val_data[0].x.shape[1]
#     hidden_dim = 18
#     output_dim = 1

#     model = SharedMPNN(num_node_features, hidden_dim, output_dim, 3)
#     model.load_state_dict(torch.load("./best_model_combined.pth"))

#     model.eval()
#     S = np.zeros((len(val_data), 3))
#     for i in range(len(val_data)):
#         S[i, 0] = val_data[i].name
#         S[i, 1] = torch.sigmoid(model(val_data[i])).min()
#         S[i, 2] = int(val_data[i].weight)

#     df = pd.DataFrame(data=S, columns=["name", "valid_ratio_prediction", "weight"])

#     df["rank_prediction"] = df["valid_ratio_prediction"].rank(
#         ascending=False, method="min"
#     )

#     df = df.sort_values(by=["rank_prediction", "weight"], ascending=True).iloc[
#         :top_stability_count
#     ]
#     df = df.sort_values(by=["weight"], ascending=True).iloc[:top_weight_count]

#     graph_path = f"./data/{test_name}/C/simplified/graphs/"

#     top_graph_indices = df["name"].values.astype("int")
#     top_graph_names = top_graph_indices.astype("str")
#     top_graph_names = [t + ".pkl" for t in top_graph_names]
#     top_simplified_graphs = gu.GraphHandler.load_graphs(graph_path, top_graph_names)

#     graph_path = f"./data/{test_name}/C/raw/graphs/"
#     top_raw_graphs = gu.GraphHandler.load_graphs(graph_path, top_graph_names)
#     analysis = "C"
#     FEA = True
#     with open("config.json", "r") as jsonfile:
#         conf = json.load(jsonfile)[test_name]

#     def FEA(G, Gs):
#         graph_handler = gu.GraphHandler(conf)

#         env = StructEnvironment()
#         (
#             node_positions,
#             node_restrained_dof,
#             node_loads,
#             edges,
#             edge_rotations,
#             edge_lenghts,
#             edge_depths,
#             edge_widths,
#             step_sizes,
#         ) = gu.GraphHandler.graph_to_array(G)

#         env.set_attributes(
#             node_positions,
#             node_restrained_dof,
#             node_loads,
#             edges,
#             edge_rotations,
#             edge_lenghts,
#             edge_depths,
#             edge_widths,
#             step_sizes,
#         )
#         UG, deflections, K = env.analyse()
#         Gs = graph_handler.set_member_deflection(G, Gs, deflections)
#         Gs = graph_handler.set_d_theta(Gs, UG)
#         Gs = graph_handler.calc_ver_deflection(Gs)
#         Gs = graph_handler.calc_drift(G, Gs, UG)
#         return Gs

#     for i in range(len(top_raw_graphs)):
#         G = top_raw_graphs[i]
#         Gs = top_simplified_graphs[i]
#         Gs = FEA(G, Gs)
#         Gs = gu.GraphHandler.node_tuple_2_index(Gs)
#         gu.GraphHandler.draw_graph(Gs, f"top_structs/{test_name}/")
#         weight = df["weight"].iloc[i]
#         name = df["name"].iloc[i]
#         print(f"name: {name} weight: {weight}")
