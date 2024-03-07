import stiffness as st
import numpy as np
import math
import geometry as geo
import member_reactions as mr
import nodal_reactions as nr
import stiffness as st
import visualization as vs


class StructEnvironment:
    def __init__(self):
        # Constants
        self.E = 32800 * 10**6  # (N/m^2) Young's modulus
        self.current_step = 0

    def calc_A(self, d, w):
        return d * w

    def calc_I(self, d, w):
        return (w * d**3) / 12

    def analyse(self):
        members_area = self.calc_A(self.members_depth, self.members_width)
        members_moment = self.calc_I(self.members_depth, self.members_width)

        K = st.build_K(
            self.members,
            members_area,
            self.rotations,
            self.lengths,
            self.E,
            members_moment,
        )
        U_free = nr.calc_displacement(K, self.point_loads, self.restrained_DOF)
        if U_free is None:
            return None

        UG = nr.assemble_UG(U_free, self.DOF, self.restrained_DOF)
        return UG

    def calculate_cost(self, def_members):
        return sum(def_members**2)

    def reset(self):
        self.current_step += 1
        members_area = np.random.normal(
            loc=self.A, scale=self.A / 3, size=len(self.members)
        )
        members_area[members_area < 1e-4] = 1e-4
        members_area = self.normalize_area(members_area)
        return members_area

    def normalize_area(self, members_area):
        members_area = members_area * (
            self.total_area / sum(members_area * self.lengths)
        )
        return members_area

    def update_area(self, members_area):
        random_element = np.random.randint(self.element_count)
        members_area[random_element] = members_area[random_element] + np.random.normal(
            loc=0, scale=self.A / 5
        )
        members_area[members_area < 1e-4] = 1e-4
        members_area = self.normalize_area(members_area)
        return members_area

    def step(self, members_area):
        F_members, def_members, UG, del_K__del_area = self.analyse(members_area)
        cost = self.calculate_cost(def_members)
        return cost, F_members, UG

    def render(self, members_depth, members_width, delections, x_fac, fname):
        vs.plot_deflection(
            self.members,
            self.nodes,
            self.rotations,
            self.lengths,
            delections,
            members_depth,
            members_width,
            x_fac,
            fname,
        )

    def init_graph(self, maximum_size):
        S = (
            np.ones(
                (
                    maximum_size,
                    maximum_size,
                )
            )
            * 0.5
        )
        num_nodes = S.shape[0] * S.shape[1]
        candidate_columns = np.random.choice(
            a=np.arange(num_nodes), size=int(num_nodes * 0.7), replace=False
        )

        return S, candidate_columns

    def random_structure(self, S, candidate_columns):
        num_stories = S.shape[0] - 1
        resolution = S.shape[1]
        num_nodes = S.shape[0] * S.shape[1]
        nodes = np.zeros((num_nodes, 2))
        floor_length = 1
        floor_heigth = 1
        members = []
        members_radius = []
        self.DOF = num_nodes * 3
        point_loads = np.array([np.zeros(self.DOF)]).T.ravel()

        selected_columns = np.random.choice(
            a=candidate_columns, size=int(len(candidate_columns) * 0.8), replace=False
        )
        S_n = S.copy()

        for i in range(num_stories + 1):
            for j in range(resolution):
                nodes[i * resolution + j] = (j * floor_length, i * floor_heigth)

        for i in range(num_stories):
            for j in range(resolution):
                if i * resolution + j not in candidate_columns:
                    S_n[i, j] = 1

                if i * resolution + j not in selected_columns:
                    S[i, j] = 0
                    point_loads[(i * resolution + j) * 3 + 1] = -10 * 1e3
                else:
                    members.append((i * resolution + j, (i + 1) * resolution + j))
                    members_radius.append(S[i, j])
                    point_loads[(i * resolution + j) * 3 + 1] = -11 * 1e3

        for i in range(
            0, num_stories + 1
        ):  # for i in range(1, num_stories + 1): # start from 1 to not to add slab for the foundation
            for j in range(resolution - 1):
                members.append((i * resolution + j, i * resolution + j + 1))
                members_radius.append(0.5)

        restrained_DOF = np.arange(resolution * 3)
        members_radius = np.array(members_radius)
        return (
            S_n,
            members_radius,
            np.array(nodes),
            np.array(members),
            restrained_DOF,
            point_loads,
        )

    def set_attributes(self, v):
        self.nodes = v["node_positions"]
        self.members = v["edges"]
        self.restrained_DOF = v["restrained_dof"]
        self.DOF = st.calc_DOF(v["edges"])
        self.element_count = len(v["edges"])
        self.point_loads = v["loads"]
        self.rotations = v["rotations"]
        self.lengths = v["lenghts"]
        self.members_depth = v["depths"]
        self.members_width = v["widths"]
        self.step_sizes = v["step_sizes"]

    def calc_drift(self, U, num_cols, num_rows):
        U = U.reshape(num_cols, num_rows, 3)
        drift = np.abs(np.diff(U[:, :, 0])).max(axis=0)
        return drift
