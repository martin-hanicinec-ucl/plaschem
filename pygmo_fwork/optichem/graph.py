from typing import Dict, Set, Tuple, Callable, List, Union

import numpy as np
import pandas as pd
from pygmo_fwork.pygmol.model_parameters import ModelParameters

from plaschem.chemistry import Chemistry

from pygmo_fwork.optichem.exceptions import GraphInitError


# noinspection DuplicatedCode
class Graph(object):
    """A very simple graph representation base class.
    Edges and Nodes have arbitrary attributes, 'weight' attribute must be a positive number!
    This class should be abstract class suitable for any graph, not exclusive to any given application
    (such as chemistry networks).
    """
    def __init__(self):
        """Initializer for the Graph base class.
        """
        self.nodes: Set[str] = set([])  # full of nodes
        self.node_attributes: Dict[str, Dict[str, float]] = {}  # keyed by nodes, full of dicts
        self.edges: Set[Tuple[str, str]] = set([])  # all edges as tuples (node1, node2)
        self.edge_attributes: Dict[Tuple[str, str], Dict[str, float]] = {}  # dict of attributes under (u, v) keys
        self.successors: Dict[str, Set[str]] = {}  # self.adj[u] = set(v1, v2, ..., vN) for all (u, vi)
        self.predecessors: Dict[str, Set[str]] = {}  # self.adj[v] = set(u1, u2, ..., uN) for all (ui, v)
        self.weights_distributed: bool = False

    def get_nodes(self) -> Set[str]:
        """Getter method for all the nodes. Nodes are str instances.
        Returns the set object which is mutable and also the stored instance attributes - careful with modifying this.
        :return: Set of all the nodes.
        """
        return self.nodes

    def update_node_attributes(self, node: str, **attributes: float) -> None:
        """A method to save attributes to any given node. The node needs to exist in the graph.
        :param node: (str) node
        :param attributes: kwarg=value, values are numbers.
        :return: None
        """
        self.node_attributes[node].update(attributes)

    def get_node_attributes(self, node: str) -> Dict[str, float]:
        """Getter method for the dict of attributes for any given existing node.
        :param node: (str) node
        :return: Dict of attributes keyed by str keys with float values. These need first be set.
        """
        return self.node_attributes[node]

    def add_edge(self, u: str, v: str) -> None:
        """Method adding an edge to the graph. If the nodes u or v are not present, these are automatically added.
        :param u: (str) tail node of the (u, v) edge
        :param v: (str) head node of the (u, v) edge
        :return: None
        """
        for node in (u, v):
            if node not in self.nodes:
                self.nodes.add(node)
                self.node_attributes[node] = {}
                self.successors[node] = set([])
                self.predecessors[node] = set([])
        if (u, v) not in self.edges:
            self.successors[u].add(v)
            self.predecessors[v].add(u)
            self.edges.add((u, v))
            self.edge_attributes[(u, v)] = {}  # so far empty

    def has_edge(self, u: str, v: str) -> bool:
        """Method returning if the edge (u, v) exists in the graph.
        :param u: (str) tail node of the edge
        :param v: (str) head node of the edge
        :return: (bool) (u, v) edge in the graph
        """
        return (u, v) in self.edges

    def get_edges(self) -> Set[Tuple[str, str]]:
        """Getter method for all the edges present in the graph. Returns a mutable structure, which is also an instance
        attribute. Careful with modifying this!
        :return: (set) of edges
        """
        return self.edges

    def update_edge_attributes(self, u: str, v: str, **attributes: float) -> None:
        """A method to save attributes to any given edge. The edge needs to exist in the graph.
        :param u: (str) tail node of the edge
        :param v: (str) head node of the edge
        :param attributes: str: float attributes for the (u, v) edge.
        :return: None
        """
        # guard against non-positive weights - they would indicate problem with weight distribution method!
        if 'weight' in attributes:
            assert attributes['weight'] > 0, 'Edge weight must be positive for each edge!'
        self.edge_attributes[(u, v)].update(attributes)

    def get_edge_attributes(self, u: str, v: str) -> Dict[str, float]:
        """Getter method for the dict of attributes for any given existing node. Returns mutable dict which is also
        a instance attribute! Careful!
        :param u: (str) tail node of the (u, v) edge
        :param v: (str) head node of the (u, v) edge
        :return: (dict) of the edge attributes
        """
        return self.edge_attributes[(u, v)]

    # *************************************** GRAPH THEORY ALGOS ***************************************************** #
    @staticmethod
    def _default_weighting_functions(
            weighting_func: Callable = None, inverse: Callable = None) -> Tuple[Callable, Callable]:
        """Get default weighting function and it's inverse, if these are not defined. Defaults are just x to x mapping.
        Also checks that if one is defined, the other also needs to be.
        :param weighting_func: (func) x: x mapping
        :param inverse: (func) x: x mapping
        :return: (func, func)
        """
        if weighting_func is not None:
            assert inverse is not None, 'Both weighting function and its inverse need to be defined!'
        if inverse is not None:
            assert weighting_func is not None, 'Both weighting function and its inverse need to be defined!'

        if weighting_func is None:
            return lambda x: x, lambda x: x
        else:
            return weighting_func, inverse

    def find_max_bottleneck_paths(
            self, target_node: str, edge_attribute: str = 'weight', weighting_func: Callable = None,
            inverse: Callable = None) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """For each edge, takes weight = weighting_func(edge[edge_attribute]) and treat it as a edge 'width'. Finds
        the widest paths (the maximum bottleneck paths) from each node to the one target node. Weights already need
        to have been distributed and each edge needs to have the 'edge_attribute' attribute. Returns the tuple of dicts,
        first is all the paths keyed by the source nodes, second is all the values inverse(max_bottleneck) for each
        path.
        If weighting_func and inverse not passed, they are not used (simple x: x lambdas are used as default.)
        :param target_node: (str)
        :param edge_attribute: (str)
        :param weighting_func: (func - optional)
        :param inverse: (func - optional)
        :return: (dict, dict) first is dict of found paths for each source node (each path is a list of nodes) and
                 the second is a dict of found paths widths keyed by each source node.
        """
        assert self.weights_distributed, 'Cannot run any graph theory algorithms - weights not yet distributed!'
        weighting_func, inverse = self._default_weighting_functions(weighting_func, inverse)

        # initialise widths:
        widths = {node: float('inf') if node == target_node else 0. for node in self.get_nodes()}
        # initialise successor relationships:
        successors = {node: None for node in self.get_nodes()}
        unrelaxed = set(self.get_nodes())

        # modified Dijkstra:
        while len(unrelaxed):
            priority_queue = sorted(unrelaxed, key=lambda x: widths[x], reverse=True)
            node = priority_queue[0]  # pop the item with largest width
            unrelaxed.remove(node)
            for predecessor in self.predecessors[node]:
                if predecessor in unrelaxed:
                    edge_width = weighting_func(self.get_edge_attributes(predecessor, node)[edge_attribute])
                    assert edge_width >= 0, 'Algo works only with non-negative edge weights!'
                    alt_width = min(widths[node], edge_width)
                    if alt_width > widths[predecessor]:
                        # relaxation
                        widths[predecessor] = alt_width
                        successors[predecessor] = node

        # at this point, I have correct widths and successors relationships, only need to reconstruct the paths
        # and apply inverse function to the result maximum bottlenecks, if defined:
        paths = {source: [source] for source in self.get_nodes()}
        for source in self.get_nodes():
            while paths[source][-1] != target_node and paths[source][-1] is not None:
                paths[source].append(successors[paths[source][-1]])
        widths = {node: inverse(widths[node]) for node in widths}

        return paths, widths

    def find_shortest_paths(self, target_node: str, edge_attribute: str = 'weight', weighting_func: Callable = None,
                            inverse: Callable = None) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """For each edge, takes weight = weighting_func(edge[edge_attribute]) and treat it as a edge 'length'. Finds
        the shortest paths from each node to the one target node. Weights already need
        to have been distributed and each edge needs to have the 'edge_attribute' attribute. Returns the tuple of dicts,
        first is all the paths keyed by the source nodes, second is all the values inverse(path_length) for each
        path.
        If weighting_func and inverse not passed, they are not used (simple x: x lambdas are used as default.)
        :param target_node: (str)
        :param edge_attribute: (str)
        :param weighting_func: (func - optional)
        :param inverse: (func - optional)
        :return: (dict, dict) first is dict of found paths for each source node (each path is a list of nodes) and
                 the second is a dict of found paths lengths keyed by each source node.
        """
        assert self.weights_distributed, 'Cannot run any graph theory algorithms - weights not yet distributed!'
        weighting_func, inverse = self._default_weighting_functions(weighting_func, inverse)

        # initialise lengths:
        lengths = {node: 0 if node == target_node else float('inf') for node in self.get_nodes()}
        # initialise successor relationships:
        successors = {node: None for node in self.get_nodes()}
        unrelaxed = set(self.get_nodes())

        # Dijkstra:
        while len(unrelaxed):
            priority_queue = sorted(unrelaxed, key=lambda x: lengths[x])
            node = priority_queue[0]  # pop the item with shortest distance
            unrelaxed.remove(node)
            for predecessor in self.predecessors[node]:
                if predecessor in unrelaxed:
                    edge_length = weighting_func(self.get_edge_attributes(predecessor, node)[edge_attribute])
                    assert edge_length >= 0, 'Algo works only with non-negative edge weights!'
                    alt_length = lengths[node] + edge_length
                    if alt_length < lengths[predecessor]:
                        # relaxation
                        lengths[predecessor] = alt_length
                        successors[predecessor] = node

        # at this point, I have correct length and successors relationships, only need to reconstruct the paths
        # and apply inverse function to the result path lengths, if defined:
        paths = {source: [source] for source in self.get_nodes()}
        for source in self.get_nodes():
            while paths[source][-1] != target_node and paths[source][-1] is not None:
                paths[source].append(successors[paths[source][-1]])
        lengths = {node: inverse(lengths[node]) if lengths[node] > 0 else np.nan for node in lengths}

        return paths, lengths

    def get_max_flow(self, source_node: str, target_node: str, edge_attribute: str = 'weight',
                     weighting_func: Callable = None, inverse: Callable = None) -> float:
        """For each edge, takes weight = weighting_func(edge[edge_attribute]) and treat it as a edge 'capacity'. Finds
        the maximum "flows" the one source_node node to the one target node. Weights already need
        to have been distributed and each edge needs to have the 'edge_attribute' attribute. Returns the maximum flow
        or if inverse supplied, then inverse(max_flow).
        If weighting_func and inverse not passed, they are not used (simple x: x lambdas are used as default.)
        :param source_node: (str)
        :param target_node: (str)
        :param edge_attribute: (str)
        :param weighting_func: (func - optional)
        :param inverse: (func - optional)
        :return: (float)
        """
        assert self.weights_distributed, 'Cannot run any graph theory algorithms - weights not yet distributed!'
        weighting_func, inverse = self._default_weighting_functions(weighting_func, inverse)

        # first build the matrix of capacities and plugging zero where edges do not exist:
        capacity = pd.DataFrame(index=self.get_nodes(), columns=self.get_nodes())
        for u in self.get_nodes():
            for v in self.get_nodes():
                if self.has_edge(u, v):
                    edge_capacity = weighting_func(self.get_edge_attributes(u, v)[edge_attribute])
                    assert edge_capacity >= 0, 'Algo works only with non-negative edge weights!'
                    capacity.at[u, v] = edge_capacity
        capacity = capacity.fillna(0.)

        # Ford-Fulkerson algorithm to find the maximum flow from source_node to target_node:
        max_flow = 0
        while True:
            # perform a BFS to find a path from source to target in the capacity matrix and find it's bottleneck:
            predecessor = {}  # dict defining the predecessor relationship to reconstruct the path from BFS
            visited = {source_node, }
            priority_queue = [source_node, ]
            # standard BFS loop:
            path_exists = False
            while len(priority_queue) and not path_exists:
                u = priority_queue.pop(0)
                for v in capacity.loc[u].index:
                    remaining_cap = capacity.at[u, v]
                    if v not in visited and remaining_cap > 0:
                        priority_queue.append(v)
                        visited.add(v)
                        predecessor[v] = u
                        if v == target_node:
                            path_exists = True
                            break
            # at the end of this, I have a flag if path exists with some remaining capacity and the predecessor
            # relationship dict defines the path, if it exists.
            if path_exists:
                # find the maximum flow through the path found by the BFS
                path_flow = float("Inf")
                s = target_node
                while s != source_node:
                    path_flow = min(path_flow, capacity.at[predecessor[s], s])
                    s = predecessor[s]

                # Add path flow to overall flow
                max_flow += path_flow

                # update residual capacities of the edges (and reverse edges) along the path
                v = target_node
                while v != source_node:
                    u = predecessor[v]
                    capacity.at[u, v] -= path_flow
                    capacity.at[v, u] += path_flow
                    v = predecessor[v]
            else:
                break  # break the main Ford-Furkenson cycle
        return inverse(max_flow)


class ChemistryGraph(object):
    def __init__(self,
                 chemistry: Chemistry, model_params: Union[dict, ModelParameters], rates_frames: pd.DataFrame,
                 wall_fluxes_frames: pd.DataFrame) -> None:
        """
        :param chemistry: (Chemistry instance)
        :param model_params: (dict-like) must have 'radius' and 'length' keys, all in [SI]
        :param rates_frames: (pd.DataFrame) index irrelevant, columns must be ['t'] + list(int(reactions ids)), with
                             each row for one important time. All in [SI]
        :param wall_fluxes_frames: (pd.DataFrame) index irrelevant, columns must be ['t'] + list(str(species_names)),
                                   with each row for one important time. All ins [SI]
        """
        self.assert_input_consistency(chemistry, rates_frames, wall_fluxes_frames)

        self.r = model_params['radius']
        self.z = model_params['length']
        self.vol = np.pi*self.r**2 * self.z  # in [m3]
        self.area = 2*np.pi*self.r * (self.r + self.z)  # in [m2]

        self.special_species = np.array([sp.get_name() for sp in chemistry.get_special_species()])
        self.species = np.array(chemistry.get_species_name(special=True))  # species names
        self.reactions = np.array(chemistry.get_reactions_id())  # reactions ids
        self.important_times = np.array(rates_frames['t'])  # all timestamps for all rows in rates_frames

        self.stick_coefs = chemistry.get_species_stick_coef(special=True).fillna(0.).values  # fill in 0 for e, M
        self.ret_matrix = chemistry.get_return_matrix(special=True).fillna(0.).values  # fill in 0 for e, M

        # rates [m-3/s] matrix - each row for one important time and each column for one reaction
        self.rates = rates_frames.drop(columns='t').values  # np.array
        # wall fluxes [SI] matrix - each row for one important time and columns are species names (str) incl special
        self.wall_fluxes = pd.DataFrame(index=wall_fluxes_frames.index, columns=['t']+list(self.species)).fillna(0.)
        self.wall_fluxes.loc[wall_fluxes_frames.index, wall_fluxes_frames.columns] = \
            wall_fluxes_frames.loc[wall_fluxes_frames.index, wall_fluxes_frames.columns]
        self.wall_fluxes = self.wall_fluxes.drop(columns='t').values  # np.array
        # stoichiometric matrices - each row for one reaction, each column for one species
        self.stoichiomatrix_lhs = chemistry.get_stoichiomatrix(special=True, method='lhs').values  # np.array
        self.stoichiomatrix_rhs = chemistry.get_stoichiomatrix(special=True, method='rhs').values  # np.array
        self.stoichiomatrix_net = self.stoichiomatrix_rhs - self.stoichiomatrix_lhs  # np.array
        # delta_ji[j, i] is 0, if i-th species does not figure in j-th reaction, and 1 otherwise.
        self.delta_ji = np.array((self.stoichiomatrix_lhs + self.stoichiomatrix_rhs) > 0, dtype=int)

        self.graphs = []  # placeholder - list holding the Graph instances for each imp. time.

    @staticmethod
    def assert_input_consistency(
            chemistry: Chemistry, rates_frames: pd.DataFrame, wall_fluxes_frames: pd.DataFrame) -> None:
        """Method checking the mutual consistency of the inputs.
        :param chemistry:
        :param rates_frames:
        :param wall_fluxes_frames:
        :return: None
        """
        stoichiomatrix_lhs = chemistry.get_stoichiomatrix(special=True, method='lhs')
        stoichiomatrix_rhs = chemistry.get_stoichiomatrix(special=True, method='rhs')
        species_ids = chemistry.get_species_id(special=True)
        reactions_ids = chemistry.get_reactions_id()
        # consistency between rates and wall fluxes:
        if list(rates_frames['t']) != list(wall_fluxes_frames['t']):
            raise GraphInitError('Rates and wall fluxes have inconsistent time stamps!')
        # consistency between species and stoichiomatrix:
        if list(stoichiomatrix_lhs.columns) != list(species_ids):
            raise GraphInitError('stoichiomatrix from chemistry inconsistent with species from chemistry!')
        if list(stoichiomatrix_rhs.columns) != list(species_ids):
            raise GraphInitError('stoichiomatrix from chemistry inconsistent with species from chemistry!')
        # consistency between reactions and stoichiomatrix:
        if list(stoichiomatrix_lhs.index) != list(reactions_ids):
            raise GraphInitError('stoichiomatrix inconsistent with reactions!')
        if list(stoichiomatrix_rhs.index) != list(reactions_ids):
            raise GraphInitError('stoichiomatrix inconsistent with reactions!')
        # consistency between reactions and rates:
        if list(rates_frames.columns) != ['t'] + list(reactions_ids):
            raise GraphInitError('Rates frames inconsistent with reactions!')
        # consistency between wall fluxes and species:
        # wall fluxes are not defined for special species:
        if list(wall_fluxes_frames.columns) != ['t'] + list(chemistry.get_species_name(special=False)):
            raise GraphInitError('Wall fluxes frames inconsistent with species!')
        # consistency between sticking coefficients, return matrix and species:
        if list(chemistry.get_return_matrix(special=True).index) != list(species_ids):
            raise GraphInitError('Return matrix inconsistent with species!')
        if list(chemistry.get_return_matrix(special=True).columns) != list(species_ids):
            raise GraphInitError('Return matrix inconsistent with species!')
        if list(chemistry.get_species_stick_coef(special=True).index) != list(species_ids):
            raise GraphInitError('Sticking coefficients inconsistent with species!')

    def init_graphs(self) -> List[Graph]:
        """Build an array with empty Graph instances each for one important time. Without distributed weights/edges.
        :return: List of Graph instances - empty ones before distributing weights etc.
        """
        return [Graph() for _ in self.important_times]  # list holding the Graph instances for each imp. time.

    def distribute_weights(self, weighting_method: str = 'DRG', surface: bool = True) -> None:
        """Method to distribute weights to all the edges in the ChemistryGraph - according to one of the competing
        different schemes: DRG, DRG1, DRG2, DRGEP. The default 'DRG' has been identified as the best of the four.
        :param weighting_method: (str) from {'DRG', 'DRG1', 'DRG2', 'DRGEP'}.
                       DRG: The coupling coefficients B -> A are the total consumption AND production
                            rate of A by all reactions involving B, normalised by total consumption AND production of A
                       DRG1: The coupling coefficients B -> A are the total consumption AND production
                             rate of A by all reactions involving B, normalised by absolute value of NET
                             production/consumption of A
                       DRGEP: The coupling coefficients B -> A are the total NET consumption/production in absolute
                              values, normalised by max{abs(tot_cons, tot_prod)}
        :param surface: (bool) if to include surface rates into the coupling coefficients or not.
        :return: None
        """
        # re-initialise the graphs list (each for one important time)
        self.graphs = self.init_graphs()
        # surface rates corrections:
        if surface:
            # contributions to species coupling coefficients from surface sink/sources - first axis is times.
            # sticking rates - axis - 0: important times, 1: sticking rates [m-3/s]
            stick_rates = self.wall_fluxes * self.area/self.vol
            # return rates - axis - 0: important times, 1: return species, 2: sticking species. rates in [m-3/s]
            ret_rates = -self.ret_matrix[np.newaxis, :, :] * stick_rates[:, np.newaxis, :]  # is it correct?
            # need to swap axes - not matrix is mirrored compared to r!
            ret_rates = np.swapaxes(ret_rates, 1, 2)
            # surface rates = sum of sticking rates and return rates
            surf_rates = np.zeros(ret_rates.shape)
            surf_rates[:, np.arange(surf_rates.shape[1]), np.arange(surf_rates.shape[2])] += stick_rates  # diagonals
            surf_rates += ret_rates
        else:
            surf_rates = np.zeros((len(self.important_times), len(self.species), len(self.species)))

        # build the species coupling coefficients:
        if weighting_method in {'DRG', 'DRG1'}:
            # total consumption AND production rate of a_i-th species by all reactions involving b_i-th species for
            # t_i-th important time will be r[t_i, b_i, a_i]:
            r = (abs(self.stoichiomatrix_net[np.newaxis, :, np.newaxis, :]) *
                 self.rates[:, :, np.newaxis, np.newaxis] *
                 self.delta_ji[np.newaxis, :, :, np.newaxis]).sum(axis=1)
            r += abs(surf_rates)
            # axis=0: important times, axis=2: species, axis=3: species
        elif weighting_method in {'DRG2', 'DRGEP'}:
            # total NET production/consumption of a_i-th species by all reactions involving b_i-th species for
            # t_i-th important time will be r[t_i, b_i, a_i]:
            r = (self.stoichiomatrix_net[np.newaxis, :, np.newaxis, :] *
                 self.rates[:, :, np.newaxis, np.newaxis] *
                 self.delta_ji[np.newaxis, :, :, np.newaxis]).sum(axis=1)
            r += surf_rates
            r = abs(r)
        else:
            raise NotImplementedError('Unrecognised weighting method: {}'.format(weighting_method))
        r[r == 0] = np.nan

        # normalisation factors:
        if weighting_method == 'DRG':  # normalising by total consumption + production
            norm = \
                (abs(self.stoichiomatrix_net[np.newaxis, :, :]) * self.rates[:, :, np.newaxis]).sum(axis=1) + \
                np.sum(abs(surf_rates), axis=1)
            # rows are important times and columns are species
        elif weighting_method in {'DRG1', 'DRG2'}:  # normalising by absolute of NET consumption/production
            norm = \
                (self.stoichiomatrix_net[np.newaxis, :, :] * self.rates[:, :, np.newaxis]).sum(axis=1) + \
                surf_rates.sum(axis=1)
            norm = abs(norm)
            # rows are important times and columns are species
        elif weighting_method == 'DRGEP':  # normalising by maximum of net consumption against net production
            stoich_net_prod = self.stoichiomatrix_net.copy()
            stoich_net_prod[stoich_net_prod < 0] = 0  # cap negative values, only interested in production
            surf_rates_prod = surf_rates.copy()
            surf_rates_prod[surf_rates_prod < 0] = 0
            stoich_net_cons = -self.stoichiomatrix_net.copy()
            stoich_net_cons[stoich_net_cons < 0] = 0  # cap negative, only interested in consumption
            surf_rates_cons = -surf_rates.copy()
            surf_rates_cons[surf_rates_cons < 0] = 0
            tot_prod = \
                (stoich_net_prod[np.newaxis, :, :] * self.rates[:, :, np.newaxis]).sum(axis=1) + \
                surf_rates_prod.sum(axis=1)
            tot_cons = \
                (stoich_net_cons[np.newaxis, :, :] * self.rates[:, :, np.newaxis]).sum(axis=1) + \
                surf_rates_cons.sum(axis=1)
            # some sanity checks:
            assert (tot_prod >= 0).all()
            assert (tot_cons >= 0).all()
            # build the normalisation array (N_times x N_species):
            norm = np.zeros(tot_prod.shape)
            norm[tot_prod > tot_cons] = tot_prod[tot_prod > tot_cons]
            norm[tot_cons > tot_prod] = tot_cons[tot_cons > tot_prod]
            # rows are important times and columns are species
        else:
            raise NotImplementedError('Unrecognised weighting method: {}'.format(weighting_method))
        norm[norm == 0] = np.nan  # so I can divide by it...
        # normalise:
        r /= norm[:, np.newaxis, :]

        # distribute the weights for all the graph for each one important time:
        for t_i, t in enumerate(self.important_times):
            graph = self.graphs[t_i]
            for a_i, a in enumerate(self.species):
                for b_i, b in enumerate(self.species):
                    if a != b:
                        weight = r[t_i, b_i, a_i]
                        if not np.isnan(weight):  # in not np.nan
                            graph.add_edge(b, a)
                            graph.update_edge_attributes(b, a, weight=weight)
            graph.weights_distributed = True

    def get_ranking(self, important_species: List[str], search_method: str = 'shortest_path') -> pd.Series:
        """Method to get ranking of all species reflecting their importance for modeling the set of important_species.
        Different competing search methods can be used: 'shortest_path', 'max_bottleneck', 'max_product' and 'max_flow'.
        The default 'shortest_path' has been identified as the best of the four.
        The ranking also is affected by what weighting scheme has been used. The self.distribute_weights needs to be
        run before this method, otherwise a ValueError will be raised.
        WARNING:
        *   If max_bottleneck is used, there may be many different source species with the same maximum bottleneck
            values towards the target species. In the pathological case, the target species is connected to the rest of
            the graph by only one edge with smallest width of all edges in the graph - in that case all the source
            species will get the same ranking score and the order will be somewhat arbitrary. Also in the case of the
            actual max_bottleneck path being utilised (and not just the value), there are likely to be many different
            paths with the same maximum bottleneck value! The path itself is likely to be very ambiguous.
        *   If max_product is used, note that it uses Dijkstra, which does not handle 0-value paths (it omits them),
            which is not a correct behaviour, since 0 in that context means edge with weight 1.0, which is a valid edge
            paths including it should be searched. Bellman-Forth algorithm might fix that.
        :param important_species: (list) of all the important species
        :param search_method: (str) one of {'shortest_path', 'max_bottleneck', 'max_product', 'max_flow'}
        :return: (pd.Series) indexed by species names and with values of ranking scores.
        """
        if not len(self.graphs):
            raise ValueError('Chemistry graphs have not been built yet, distribute_weights() must be called first!')
        static_species = set(important_species) | set(self.special_species)
        # static species stay as they are, no need to compute rankings for them...
        assessed_species = self.species[[sp not in static_species for sp in self.species]]
        # frame with rows for all important outputs and times and columns only species I'm ranking (not special or imp.)
        # the values are dependent on the search method.
        ranking_all = pd.DataFrame(columns=assessed_species)  # only a seed...
        for important_sp in important_species:
            for t_i, important_time in enumerate(self.important_times):
                graph = self.graphs[t_i]
                index = '{} ({:.3e})'.format(important_sp, important_time)
                if search_method == 'max_bottleneck':
                    paths, scores = graph.find_max_bottleneck_paths(target_node=important_sp, edge_attribute='weight')
                elif search_method == 'shortest_path':
                    paths, scores = graph.find_shortest_paths(target_node=important_sp, edge_attribute='weight',
                                                              weighting_func=lambda x: 1/x, inverse=lambda x: 1/x)
                elif search_method == 'max_product':
                    paths, scores = graph.find_shortest_paths(target_node=important_sp, edge_attribute='weight',
                                                              weighting_func=lambda x: np.log(1/x),
                                                              inverse=lambda x: 1/np.exp(x))
                elif search_method == 'max_flow':
                    scores = {}
                    for sp in assessed_species:
                        max_flow = graph.get_max_flow(source_node=sp, target_node=important_sp, edge_attribute='weight')
                        scores[sp] = max_flow
                else:
                    raise ValueError('Unrecognised search method: {}'.format(search_method))
                ranking_all.loc[index, :] = scores

        ranking = ranking_all.max(axis=0)
        # sort values (also first sorting by keys, so order for species with the same scores is consistent)
        ranking = ranking[sorted(ranking.index)]
        ranking = ranking.sort_values(ascending=True)
        return ranking
