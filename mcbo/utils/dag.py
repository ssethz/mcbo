from typing import List, Optional


class DAG(object):
    """
    Defines a DAG based upon a list of the parents of each node.
    """

    def __init__(self, parent_nodes: List[List[Optional[int]]]):
        self.parent_nodes = parent_nodes
        self.n_nodes = len(parent_nodes)
        self.root_nodes = []
        for k in range(self.n_nodes):
            if len(parent_nodes[k]) == 0:
                self.root_nodes.append(k)

    def get_n_nodes(self):
        return self.n_nodes

    def get_parent_nodes(self, k):
        return self.parent_nodes[k]

    def get_root_nodes(self):
        return self.root_nodes


class FNActionInput(object):
    """
    Defines the action targets based upon a list of the action indices effecting
    each graph node. Used only for function networks.
    """

    def __init__(self, active_input_indices: List[List[Optional[int]]]):
        self.active_input_indices = active_input_indices
        # input dim is the highest action index
        self.input_dim = (
            max([0 if len(x) == 0 else max(x) for x in active_input_indices]) + 1
        )

    def get_input_dim(self):
        return self.input_dim

    def get_active_input_indices(self):
        return self.active_input_indices
