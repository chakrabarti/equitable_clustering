import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import *
from enum import IntEnum
from typing import *

import numpy as np

from timer import Timer

# we return None when things are infeasible

Point = int

verbose = False


class Algorithm(IntEnum):
    AG = 1
    PP = 2
    Pseudo_PoF = 3


class ClusteringProblem:
    def __init__(
        self,
        k: int,
        dist_matrix: np.array,
    ):
        """
        Returns an object to which stochastic constraints can be added to represent the clustering problem we care about.
        Args:
            k (int): The number of clusters.
            dist_matrix (np.array): Distance matrix corresponding to distances between each of the points in original_points (this doesn't have to be the Euclidean metric and will be used for k-center objective calculation).
        """
        self._k = k
        self._dist_matrix = dist_matrix
        self._similarity_sets = defaultdict(set)
        self._R_f = None
        self._R_f_index = None
        self._hs_phi = None
        self._sorted_distances = np.unique(self.dist_matrix)
        self._R_j = None
        self._R_j_array = None

    def construct_similarity_sets(self):
        assert self._R_f is not None, "R_f not found yet"
        if self._R_j_array is None:
            R_j_array = np.random.uniform(0, 2*self.R_f, self.num_points)
            self._R_j_array = R_j_array
        else:
            R_j_array = self._R_j_array
        for point in range(self.num_points):
            indices = set((self.dist_matrix[point] <= R_j_array[point]).nonzero()[0])
            self._similarity_sets[point].update(indices)
        max_R_j = np.max(R_j_array)
        index = np.searchsorted(self._sorted_distances, max_R_j)
        if index < len(self._sorted_distances) - 1:
            index += 1
        self._R_f_index = index

    def find_R_f(self):
        self._R_f, self._R_f_index, self._hs_phi = find_R_f(self, hs=True)

    def set_R_f(self, R_f, R_f_index, hs_phi):
        self._R_f, self._R_f_index, self._hs_phi = R_f, R_f_index, hs_phi

    @property
    def R_j_array(self):
        return self._R_j_array

    @R_j_array.setter
    def R_j_array(self, arr):
        self._R_j_array = arr

    @property
    def hs_phi(self):
        return self._hs_phi

    @property
    def num_points(self):
        return self._dist_matrix.shape[0]

    def permutation_of_points(self, specified_set=None, list_fmt=False):
        if not list_fmt:
            if specified_set is None:
                return np.random.permutation(self.num_points)
            else:
                return np.random.permutation(list(specified_set))
        else:
            if specified_set is None:
                return random.sample(range(self.num_points), self.num_points)
            else:
                return random.sample(list(specified_set), len(specified_set))

    @property
    def dist_matrix(self):
        return self._dist_matrix

    @property
    def k(self):
        return self._k

    @property
    def similarity_sets(self):
        return self._similarity_sets

    @property
    def R_f(self):
        return self._R_f

    @property
    def R_f_index(self):
        return self._R_f_index

    @property
    def sorted_distances(self):
        return self._sorted_distances


def hochbaum_shmoys_filtering(clustering_problem: ClusteringProblem, R: float) -> bool:
    """Runs Hochbaum-Shmoys algorithm

    Args:
        clustering_problem (ClusteringProblem): the problem instance
        R (float): the radius guess to be used for Hochbaum-Shmoys

    Returns:
        bool: returns True if a feasible clustering can be found using the provided R
    """
    k = clustering_problem.k
    dist_matrix = clustering_problem.dist_matrix
    U = clustering_problem.permutation_of_points()  # pick in an arbitrary order
    set_U = set(U)
    S = set()  # selected centers
    max_radius = None
    while len(set_U):
        c = random.sample(set_U, 1)[0]
        set_U.remove(c)
        S.add(c)
        G_c = set()
        for j in set_U:
            if dist_matrix[j, c] <= 2 * R:
                if max_radius is None:
                    max_radius = dist_matrix[j, c]
                else:
                    max_radius = max(max_radius, dist_matrix[j, c])
                G_c.add(j)
        set_U -= G_c
    if len(S) > k:
        return False
    else:
        return True, S


def choose_centers(
    clustering_problem, R
) -> Optional[Dict[Point, Tuple[Set[Point], int]]]:
    dist = clustering_problem.dist_matrix
    set_U = set(range(clustering_problem.num_points))
    G_c_dict = (
        dict()
    )  # this will be a dictionary mapping centers to the points clustered to it along with the number of things in its component
    P = [set()]
    S = set()
    k = clustering_problem.k
    while len(set_U):
        list_U = clustering_problem.permutation_of_points(
            specified_set=set_U
        )  # permute elements of set for randomness
        chosen_center = None
        if len(P[-1]):
            for c in list_U:
                if dist[c, list(P[-1])].min() <= 3 * R:
                    chosen_center = c
                    G_c_dict[chosen_center] = set()
                    P[-1].add(chosen_center)
                    break
        if chosen_center is None:
            chosen_center = random.sample(set_U, 1)[0]
            G_c_dict[chosen_center] = set()
            if len(P[0]):  # we're not in the first center case
                P.append({chosen_center})
            else:  # just for first run
                P[0] = {chosen_center}
        G_c_dict[chosen_center].update(
            list_U[(dist[chosen_center, list_U] <= 2 * R).nonzero()]
        )
        set_U -= G_c_dict[chosen_center]
        assert chosen_center not in set_U
        S.add(chosen_center)
    if len(S) > k:
        return None  # this is the infeasible case
    else:
        seen = set()
        for component in P:
            for center in component:
                assert center not in seen, f"Center {center} has been seen before!"
                seen.add(center)
                G_c_dict[center] = (G_c_dict[center], len(component))
        return G_c_dict


def nonisolated_and_isolated(clustering_problem, G_c_dict):
    isolated_map = {
        k: v[0] for k, v in G_c_dict.items() if v[1] == 1
    }  # checking if the P_t that the center is in has length 1
    non_isolated_map = {
        k: v[0] for k, v in G_c_dict.items() if v[1] > 1
    }  # checking if the P_t that the center is in has length more than 1
    reverse_ni_map = {v_elem: k for k, v in non_isolated_map.items() for v_elem in v}

    return isolated_map, non_isolated_map, reverse_ni_map


def assignment_points_non_isolated(
    clustering_problem,
    R,
    non_isolated_map,
    reverse_ni_map,
):
    dist_matrix = clustering_problem.dist_matrix

    H = defaultdict(lambda: (set(), set()))
    H_map = dict()

    phi = -1 * np.ones(clustering_problem.num_points, dtype=int)

    non_isolated_centers = np.array(list(non_isolated_map.keys()), dtype=np.int)
    for j, c in reverse_ni_map.items():
        if dist_matrix[j, c] <= R:
            H[c][0].add(j)
            H_map[j] = c
        else:
            potential_c_primes = non_isolated_centers[
                (dist_matrix[j, non_isolated_centers] <= R).nonzero()[0]
            ]
            if len(potential_c_primes):
                c_prime = np.random.choice(potential_c_primes)  # choose randomly
                H[c_prime][0].add(j)
                H_map[j] = c_prime
            else:
                H[c][1].add(j)
                H_map[j] = c

    for j, c in H_map.items():
        valid_c_prime = list(non_isolated_map.keys())
        assert (
            c in valid_c_prime
        ), f"{c} not in valid_c_prime, keys are {non_isolated_map.keys()}, values are {set(reverse_ni_map.values())}"
        valid_c_prime.remove(c)
        valid_c_prime = np.array(valid_c_prime)
        potential_distances = dist_matrix[j, valid_c_prime]
        if j in H[c][0]:
            best_c_prime = valid_c_prime[potential_distances.argmin()]
        else:
            assert j in H[c][1]
            argmin_c_prime = potential_distances.argmin()
            min_value = potential_distances[argmin_c_prime]
            if min_value > 2 * R:
                best_c_prime = valid_c_prime[argmin_c_prime]
            else:
                filtered_distances = potential_distances[potential_distances <= 2 * R]
                filtered_c_primes = valid_c_prime[potential_distances <= 2 * R]
                best_c_prime = filtered_c_primes[filtered_distances.argmax()]
                if dist_matrix[j, c] > dist_matrix[j, best_c_prime]:
                    assert dist_matrix[j, c] <= 2 * R
                    best_c_prime = c

        phi[j] = best_c_prime
    return phi


def assignment_isolated_ag_pp(clustering_problem, isolated_map, algorithm):
    phi = -1 * np.ones(clustering_problem.num_points, dtype=int)
    if algorithm == Algorithm.AG:
        constraint_fn = check_f1
    else:
        constraint_fn = check_f2

    for c, G_c in isolated_map.items():
        G_c_assigned = False
        for j in G_c:
            if constraint_fn(clustering_problem, j, G_c):
                for j_prime in G_c:
                    phi[j_prime] = j
                G_c_assigned = True
                break

        if not G_c_assigned:
            G_c_list = list(G_c)
            subset_distances = clustering_problem.dist_matrix[G_c_list][:, G_c_list]
            x_ind, y_ind = np.unravel_index(
                subset_distances.argmax(), subset_distances.shape
            )
            assert x_ind != y_ind
            x, y = G_c_list[x_ind], G_c_list[y_ind]

            for j in G_c_list:
                x_dist = clustering_problem.dist_matrix[j, x]
                y_dist = clustering_problem.dist_matrix[j, y]
                if x_dist > y_dist:
                    phi[j] = x
                else:
                    phi[j] = y

    return phi


def check_f1(clustering_problem, center, assigned_points, alpha=2):
    similarity_sets = clustering_problem.similarity_sets
    dist_to_center = clustering_problem.dist_matrix[center]
    for point in assigned_points:
        similar_points = similarity_sets[point]
        assert similar_points.issubset(
            assigned_points
        ), f"Similar points not subset of assigned points! {point}, {similar_points - assigned_points}"
        distances = dist_to_center[list(similar_points)]
        if dist_to_center[point] > alpha * sum(distances) / len(similar_points):
            return False
    return True


def check_f2(clustering_problem, center, assigned_points, alpha=2):
    similarity_sets = clustering_problem.similarity_sets
    dist_to_center = clustering_problem.dist_matrix[center]
    for point in assigned_points:
        similar_points = similarity_sets[point]
        assert similar_points.issubset(
            assigned_points
        ), "Similar points not subset of assigned points!"
        distances = dist_to_center[list(similar_points)]
        if dist_to_center[point] > alpha * min(distances):
            return False
    return True


def assignment_isolated_pseudo_pof(clustering_problem, isolated_map):
    dist_matrix = clustering_problem.dist_matrix
    phi = -1 * np.ones(clustering_problem.num_points, dtype=int)
    for c, G_c in isolated_map.items():
        G_c_list = list(G_c)
        subset_distances = dist_matrix[G_c_list][:, G_c_list]
        x_ind, y_ind = np.unravel_index(
            subset_distances.argmax(), subset_distances.shape
        )
        x, y = G_c_list[x_ind], G_c_list[y_ind]

        for j in G_c_list:
            x_dist = dist_matrix[j, x]
            y_dist = dist_matrix[j, y]
            if x_dist > y_dist:
                phi[j] = x
            else:
                phi[j] = y
    return phi


def general_algorithm(clustering_problem, R, algorithm):
    main_timer = Timer("general_algorithm")
    G_c_dict = choose_centers(clustering_problem, R)
    if G_c_dict is None:
        main_timer.Accumulate()
        return None
    else:
        center_dist = np.copy(
            clustering_problem.dist_matrix[list(G_c_dict.keys())][
                :, list(G_c_dict.keys())
            ]
        )
        np.fill_diagonal(center_dist, np.inf)
        assert np.all(center_dist > 2 * R)
        isolated_map, non_isolated_map, reverse_ni_map = nonisolated_and_isolated(
            clustering_problem, G_c_dict
        )

        non_isolated_dist = np.copy(
            clustering_problem.dist_matrix[list(non_isolated_map.keys())][
                :, list(non_isolated_map.keys())
            ]
        )

        np.fill_diagonal(non_isolated_dist, np.inf)
        if non_isolated_dist.size > 0:
            assert np.max(np.min(non_isolated_dist, axis=1)) <= 3 * R

        for c, G_c in isolated_map.items():
            for point in G_c:
                for other_point in range(clustering_problem.num_points):
                    if other_point not in G_c:
                        assert clustering_problem.dist_matrix[point, other_point] > R

        phi = assignment_points_non_isolated(
            clustering_problem, R, non_isolated_map, reverse_ni_map
        )

        if algorithm == Algorithm.AG or algorithm == Algorithm.PP:
            ag_pp_phi = assignment_isolated_ag_pp(
                clustering_problem,
                isolated_map,
                algorithm == Algorithm.AG,
            )
            phi = np.maximum(phi, ag_pp_phi)
        else:
            assert algorithm == Algorithm.Pseudo_PoF
            pseudo_pof_phi = assignment_isolated_pseudo_pof(
                clustering_problem, isolated_map
            )
            phi = np.maximum(phi, pseudo_pof_phi)

        if algorithm != Algorithm.Pseudo_PoF:
            if len(set(phi)) > clustering_problem.k:
                main_timer.Accumulate()
                return None
        main_timer.Accumulate()
        return phi, isolated_map, non_isolated_map


def find_R_f(clustering_problem, algorithm=Algorithm.AG, hs=True) -> float:
    sorted_distances = clustering_problem.sorted_distances
    if hs:
        func = hochbaum_shmoys_filtering
    else:
        func = general_algorithm
        if clustering_problem.R_f is None:
            clustering_problem.find_R_f()
        if not len(clustering_problem.similarity_sets):
            clustering_problem.construct_similarity_sets()

    l = 0 if hs else clustering_problem.R_f_index
    h = len(sorted_distances) - 1
    i = 0

    best_result = None

    while l < h:
        m = (l + h) // 2
        i += 1
        if verbose:
            print(f"Iteration {i} of binary search for main_problem: {not hs}")
            print(f"Current value of m is {m}")
        if hs:
            result = func(clustering_problem, sorted_distances[m])
        else:
            result = func(clustering_problem, sorted_distances[m], algorithm)

        if (not hs and result is not None) or (result and hs):
            h = m
            best_result = result
        else:
            l = m + 1

    r_f = sorted_distances[l]

    if hs:
        hs_phi = dict()
        hs_S = list(best_result[-1])
        for point in range(clustering_problem.num_points):
            hs_phi[point] = hs_S[clustering_problem.dist_matrix[point, hs_S].argmin()]

        return r_f, l, hs_phi  # minimum radius that's feasible
    else:
        return (
            r_f,
            best_result,
            clustering_problem.R_f,
            clustering_problem.hs_phi,
        )


def ChooseInitialMin(A: np.array, k: int) -> float:
    """
    Takes minimum of maximum of each row of matrix.
    """
    max_each_row = np.amax(A, axis=1)
    minmax = np.argmin(max_each_row)
    return minmax


def Rest(A: np.array, k: int, centers: np.array) -> np.array:
    """
    Determines rest of centers to be used in Gonzalez.
    """
    while k > 0:
        A[centers, :] = np.zeros((np.size(centers), np.shape(A)[1]))
        relevant = A[:, centers]
        min_each_row = np.amin(relevant, axis=1)
        maxmin = np.argmax(min_each_row)
        centers.append(maxmin)
        k -= 1
    return np.array(centers)


def GonzalezVariant(
    choose_function: Callable[[np.array, int], float], dist_matrix: np.array, k: int
) -> Tuple[np.array, float, np.array, np.array, List[int]]:
    """
    Helper for the Gonzalez algorithm.
    """
    A = deepcopy(dist_matrix)
    centers = []
    centers.append(choose_function(A, k))
    centers = Rest(A, k - 1, centers)
    relevant_distances = dist_matrix[:, centers]
    center_assignments = centers[np.argmin(relevant_distances, axis=1)]
    # radii = np.amin(relevant_distances, axis=1)
    # max_radius = np.amax(radii)
    # clusters = list(set(center_assignments))
    return center_assignments


def Gonzalez(
    dist_matrix: np.array, k: int
) -> Tuple[np.array, float, np.array, np.array, List[int]]:
    """
    Run Gonzalez algorithm.
    """
    return GonzalezVariant(ChooseInitialMin, dist_matrix, k)


def histogram_helper(counter, key):
    if key <= 2:
        counter["[0, 2]"] += 1
    elif key <= 10:
        counter["(2, 10]"] += 1
    elif key <= 100:
        counter["(10, 100]"] += 1
    else:
        counter["(100, inf)"] += 1


def run_analysis(
    clustering_problem,
    phi,
):
    dist_matrix = clustering_problem.dist_matrix
    f1_counts = dict()
    f2_counts = dict()
    similarity_sets = clustering_problem.similarity_sets

    max_f1_ratio = None
    max_f2_ratio = None
    max_distance = None
    f1_histogram = Counter()
    f2_histogram = Counter()

    for point in range(clustering_problem.num_points):
        similar_points = similarity_sets[point]
        total_distance = 0
        f2_ratio = None
        point_distance = dist_matrix[phi[point], point]
        if max_distance is None or max_distance < point_distance:
            max_distance = point_distance

        for similar_point in similar_points:
            distance = dist_matrix[phi[similar_point], similar_point]
            total_distance += distance
            if distance == 0:
                if point_distance == 0:
                    distance_ratio = 1
                else:
                    distance_ratio = np.inf
            else:
                distance_ratio = point_distance / distance
            if f2_ratio is None or distance_ratio > f2_ratio:
                f2_ratio = distance_ratio

        average_distance = total_distance / len(similar_points)

        if average_distance == 0:
            if point_distance == 0:
                f1_ratio = 1
            else:
                f1_ratio = np.inf
        else:
            f1_ratio = point_distance / average_distance

        histogram_helper(f1_histogram, f1_ratio)
        histogram_helper(f2_histogram, f2_ratio)

        f1_counts[point] = f1_ratio
        f2_counts[point] = f2_ratio
        if (max_f1_ratio is None or f1_ratio > max_f1_ratio) and (f1_ratio != np.inf):
            max_f1_ratio = f1_ratio
        if (max_f2_ratio is None or f2_ratio > max_f2_ratio) and (f2_ratio != np.inf):
            max_f2_ratio = f2_ratio

        assert max_f1_ratio != np.inf
        assert max_f2_ratio != np.inf

    analysis_dict = dict()
    analysis_dict["Max radius"] = max_distance
    analysis_dict["Max f1 ratio"] = max_f1_ratio
    analysis_dict["Max f2 ratio"] = max_f2_ratio
    analysis_dict["f1 histogram"] = f1_histogram
    analysis_dict["f2 histogram"] = f2_histogram
    analysis_dict["f1_counts"] = f1_counts
    analysis_dict["f2_counts"] = f2_counts
    if isinstance(phi, dict):
        analysis_dict["Num centers"] = len(set(phi.values()))
    else:
        analysis_dict["Num centers"] = len(set(phi))

    return analysis_dict
