from collections import deque
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import kmeans_plusplus
import time
import copy
from numba import njit

@njit(cache=True)
def RandomCentroid_assist(data, membership, ml_map, k, d):
    """
    Accumulate per-cluster sums and sum-of-squares for centroid perturbation.

    Used by CentroidUpdate_random to gather statistics needed to sample perturbed
    centroids from the empirical cluster distribution (mean and variance).

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Raw data points.
    membership : ndarray, shape (n_supernodes,)
        Cluster assignment for each super-node.
    ml_map : ndarray, shape (n,)
        Maps each raw data point to its super-node index.
    k : int
        Number of clusters.
    d : int
        Feature dimension.

    Returns
    -------
    sums : ndarray, shape (k, d)
        Sum of data vectors per cluster.
    sum_vec_sq : ndarray, shape (k, d)
        Sum of element-wise squared data vectors per cluster.
    counts : ndarray, shape (k,)
        Number of data points per cluster.
    """
    n = ml_map.size
    sums   = np.zeros((k, d), dtype=data.dtype)
    sum_vec_sq = np.zeros((k, d),   dtype=data.dtype)
    counts = np.zeros((k,),   dtype=np.int64)

    for i in range(n):
        c = membership[ml_map[i]]
        counts[c] += 1
        for j in range(d):
            sums[c, j] += data[i, j]
            sum_vec_sq[c, j] += data[i, j]**2

    return sums, sum_vec_sq, counts

@njit(cache=True)
def CentroidUpdate_assist(membership, y_values, y_values2, k, ml_count, n_supernodes, d):
    """
    Accumulate per-cluster sums and sum-of-squares over super-nodes.

    Core Numba kernel for CentroidUpdate. Iterates over super-nodes and
    accumulates the weighted sums needed to compute cluster centroids and WCSS.

    Parameters
    ----------
    membership : ndarray, shape (n_supernodes,)
        Cluster assignment for each super-node.
    y_values : ndarray, shape (n_supernodes, d)
        Aggregated feature sums for each super-node.
    y_values2 : ndarray, shape (n_supernodes,)
        Aggregated squared L2 norm for each super-node (sum of ||y_i||^2).
    k : int
        Number of clusters.
    ml_count : ndarray, shape (n_supernodes,)
        Number of raw data points in each super-node.
    n_supernodes : int
        Number of super-nodes.
    d : int
        Feature dimension.

    Returns
    -------
    sums : ndarray, shape (k, d)
        Weighted sum of super-node feature vectors per cluster.
    sum_sq : ndarray, shape (k,)
        Sum of squared norms per cluster.
    counts : ndarray, shape (k,)
        Total number of data points per cluster.
    """
    sums   = np.zeros((k, d), dtype=y_values.dtype)
    sum_sq = np.zeros((k,),   dtype=y_values2.dtype)
    counts = np.zeros((k,),   dtype=np.int64)

    for i in range(n_supernodes):
        c = membership[i]
        counts[c] += ml_count[i]
        sum_sq[c] += y_values2[i]
        for j in range(d):
            sums[c, j] += y_values[i, j]

    return sums, sum_sq, counts

@njit(cache=True, fastmath=True)
def build_MWSP_core(D, neighbors, neighbors_idx, membership_vertices, k, n_vertices, n_swaps_cap):
    """
    Enumerate candidate Kempe chain swaps and their conflict structure for MWIS.

    For each pair of clusters (i, j), this function identifies all Kempe chains
    (connected components of the subgraph induced by nodes in cluster i or j)
    and filters those with negative swap cost delta < 0 (i.e., swaps that reduce
    WCSS). The conflict graph among swaps is also built:
    - Clique constraints: a node can participate in at most one swap.
    - Cannot-link constraints: two swaps share a cannot-link edge between them.

    The result feeds directly into the MWIS solver inside KSAssignment.

    Parameters
    ----------
    D : ndarray, shape (n_vertices, k)
        Distance of each super-node to each cluster centroid.
    neighbors : ndarray
        CSR-format neighbor array (local indices within connected component).
    neighbors_idx : ndarray
        CSR-format index pointer array; neighbors of node v are neighbors[neighbors_idx[v]:neighbors_idx[v+1]].
    membership_vertices : ndarray, shape (n_vertices,)
        Current cluster label for each local vertex.
    k : int
        Number of clusters.
    n_vertices : int
        Number of vertices in this connected component.
    n_swaps_cap : int
        Maximum number of swaps to enumerate before early exit.

    Returns
    -------
    swaps_i_arr : ndarray
        Source cluster index for each enumerated swap.
    swaps_j_arr : ndarray
        Target cluster index for each enumerated swap.
    swaps_Hi_arr : ndarray
        Concatenated H_i (source-side) vertex lists across swaps.
    swaps_Hj_arr : ndarray
        Concatenated H_j (target-side) vertex lists across swaps.
    swaps_Hi_index_arr : ndarray
        CSR-style pointers into swaps_Hi_arr per swap.
    swaps_Hj_index_arr : ndarray
        CSR-style pointers into swaps_Hj_arr per swap.
    swaps_weights_arr : ndarray
        Swap cost delta (negative = improving).
    points_in_swaps_arr : ndarray, shape (n_vertices, k-1)
        For each vertex, which swap IDs contain it (clique constraints).
    points_in_swaps_arr_idx : ndarray, shape (n_vertices,)
        Number of valid entries per row of points_in_swaps_arr.
    adj_swaps_u_arr : ndarray
        First endpoint of each swap-vs-swap conflict edge (cannot-link constraint).
    adj_swaps_v_arr : ndarray
        Second endpoint of each swap-vs-swap conflict edge.
    """
    eps = 1e-6
    len_colors = np.zeros(k, dtype=np.uint32)
    color_class_arr = np.empty((k, n_vertices), dtype=np.uint32)
    for v in range(n_vertices):
        c = membership_vertices[v]
        color_class_arr[c][len_colors[c]] = v
        len_colors[c] += 1
    s_ids_arr = np.empty((k, n_swaps_cap), dtype=np.uint32)
    s_ids_arr_idx = np.zeros(k, dtype=np.uint32)
    points_in_swaps_arr = np.empty((n_vertices, k-1), dtype=np.uint32)
    points_in_swaps_arr_idx = np.zeros(n_vertices, dtype=np.uint32)
    swaps_i_arr = np.empty(n_swaps_cap, np.uint32)
    swaps_j_arr = np.empty(n_swaps_cap, np.uint32)
    swaps_Hi_arr = np.empty((k-1) * n_vertices, np.uint32)
    swaps_Hj_arr = np.empty((k-1) * n_vertices, np.uint32)
    swaps_Hi_index_arr = np.empty(n_swaps_cap + 1, np.uint32)
    swaps_Hj_index_arr = np.empty(n_swaps_cap + 1, np.uint32)
    swaps_Hi_index_arr[0] = 0
    swaps_Hj_index_arr[0] = 0
    swaps_weights_arr = np.empty(n_swaps_cap, dtype=np.float64)
    len_adj_cap = int(n_swaps_cap * (n_swaps_cap-1) / 2)
    adj_swaps_u_arr = np.empty(len_adj_cap, np.uint32)
    adj_swaps_v_arr = np.empty(len_adj_cap, np.uint32)
    len_adj = 0
    visited = np.zeros(n_vertices, dtype=np.uint32)
    q_v = np.empty(n_vertices, dtype=np.uint32)
    q_side = np.empty(n_vertices, dtype=np.bool_)
    H_i_neighbor = np.full(n_vertices, -1, dtype=np.int64)
    H_j_neighbor = np.full(n_vertices, -1, dtype=np.int64)
    stamp = 0
    s_id = 0
    len_swaps_Hi = 0
    len_swaps_Hj = 0
    for i in range(k):
        len_i = len_colors[i]
        for j in range(i + 1, k):
            len_j = len_colors[j]
            if len_i == 0:
                for v_idx in range(len_j):
                    v = color_class_arr[j, v_idx]
                    delta = -(D[v, j] - D[v, i])
                    if delta < -eps:
                        swaps_i_arr[s_id] = i
                        swaps_j_arr[s_id] = j
                        swaps_Hj_arr[len_swaps_Hj] = v
                        len_swaps_Hj += 1
                        swaps_Hi_index_arr[s_id + 1] = len_swaps_Hi
                        swaps_Hj_index_arr[s_id + 1] = len_swaps_Hj
                        swaps_weights_arr[s_id] = delta
                        neighbor_from = neighbors_idx[v]
                        neighbor_to = neighbors_idx[v+1]
                        points_in_swaps_arr[v, points_in_swaps_arr_idx[v]] = s_id
                        points_in_swaps_arr_idx[v] += 1
                        if neighbor_to > neighbor_from:
                            for t in range(neighbor_from,neighbor_to):
                                H_j_neighbor[neighbors[t]] = s_id
                            for s in s_ids_arr[i, :s_ids_arr_idx[i]]:
                                ii = swaps_i_arr[s]
                                jj = swaps_j_arr[s]
                                if (ii == i) and (j != jj):
                                    jj_from = swaps_Hj_index_arr[s]
                                    jj_to = swaps_Hj_index_arr[s+1]
                                    for u_index in range(jj_from, jj_to):
                                        u = swaps_Hj_arr[u_index]
                                        if H_j_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                                elif jj == i:
                                    ii_from = swaps_Hi_index_arr[s]
                                    ii_to = swaps_Hi_index_arr[s+1]
                                    for u_index in range(ii_from, ii_to):
                                        u = swaps_Hi_arr[u_index]
                                        if H_j_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                            s_ids_arr[i, s_ids_arr_idx[i]] = s_id
                            s_ids_arr_idx[i] += 1
                        s_id += 1
                        if s_id >= n_swaps_cap:
                            return swaps_i_arr[:s_id], swaps_j_arr[:s_id], swaps_Hi_arr[:len_swaps_Hi], swaps_Hj_arr[:len_swaps_Hj], swaps_Hi_index_arr[:s_id+1], swaps_Hj_index_arr[:s_id+1], swaps_weights_arr[:s_id], points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_arr[:len_adj], adj_swaps_v_arr[:len_adj]
            elif len_j == 0:
                for v_idx in range(len_i):
                    v = color_class_arr[i,v_idx]
                    delta = (D[v, j] - D[v, i])
                    if delta < -eps:
                        swaps_i_arr[s_id] = i
                        swaps_j_arr[s_id] = j
                        swaps_Hi_arr[len_swaps_Hi] = v
                        len_swaps_Hi += 1
                        swaps_Hi_index_arr[s_id + 1] = len_swaps_Hi
                        swaps_Hj_index_arr[s_id + 1] = len_swaps_Hj
                        swaps_weights_arr[s_id] = delta
                        neighbor_from = neighbors_idx[v]
                        neighbor_to = neighbors_idx[v+1]
                        points_in_swaps_arr[v, points_in_swaps_arr_idx[v]] = s_id
                        points_in_swaps_arr_idx[v] += 1
                        if neighbor_to > neighbor_from:
                            for t in range(neighbor_from, neighbor_to):
                                H_i_neighbor[neighbors[t]] = s_id
                            for s in s_ids_arr[j, :s_ids_arr_idx[j]]: # swaps involving j
                                ii = swaps_i_arr[s]
                                jj = swaps_j_arr[s]
                                if ii == j:
                                    jj_from = swaps_Hj_index_arr[s]
                                    jj_to = swaps_Hj_index_arr[s+1]
                                    for u_index in range(jj_from, jj_to):
                                        u = swaps_Hj_arr[u_index]
                                        if H_i_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                                elif (jj == j) and (i != ii):
                                    ii_from = swaps_Hi_index_arr[s]
                                    ii_to = swaps_Hi_index_arr[s+1]
                                    for u_index in range(ii_from, ii_to):
                                        u = swaps_Hi_arr[u_index]
                                        if H_i_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                            s_ids_arr[j, s_ids_arr_idx[j]] = s_id
                            s_ids_arr_idx[j] += 1
                        s_id += 1
                        if s_id >= n_swaps_cap:
                            return swaps_i_arr[:s_id], swaps_j_arr[:s_id], swaps_Hi_arr[:len_swaps_Hi], swaps_Hj_arr[:len_swaps_Hj], swaps_Hi_index_arr[:s_id+1], swaps_Hj_index_arr[:s_id+1], swaps_weights_arr[:s_id], points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_arr[:len_adj], adj_swaps_v_arr[:len_adj]
            else:
                stamp += 1
                for v_idx in range(len_i):
                    start = color_class_arr[i,v_idx]
                    if visited[start] == stamp:
                        continue
                    head = 0
                    tail = 0
                    q_v[tail] = start
                    q_side[tail] = True
                    tail += 1
                    visited[start] = stamp
                    delta = 0.0
                    while head < tail:
                        v = q_v[head]
                        side = q_side[head]
                        head += 1
                        if side:
                            delta += (D[v, j] - D[v, i])
                            neighbor_from = neighbors_idx[v]
                            neighbor_to = neighbors_idx[v+1]
                            for t in range(neighbor_from, neighbor_to):
                                u = neighbors[t]
                                if (membership_vertices[u] == j) and (visited[u] != stamp):
                                    visited[u] = stamp
                                    q_v[tail] = u
                                    q_side[tail] = False
                                    tail += 1
                        else:
                            delta -= (D[v, j] - D[v, i])
                            neighbor_from = neighbors_idx[v]
                            neighbor_to = neighbors_idx[v+1]
                            for t in range(neighbor_from, neighbor_to):
                                u = neighbors[t]
                                if (membership_vertices[u] == i) and (visited[u] != stamp):
                                    visited[u] = stamp
                                    q_v[tail] = u
                                    q_side[tail] = True
                                    tail += 1
                    if delta < -eps:
                        swaps_i_arr[s_id] = i
                        swaps_j_arr[s_id] = j
                        swaps_weights_arr[s_id] = delta
                        H_i_neighbor_non_empty = False
                        H_j_neighbor_non_empty = False
                        for idx in range(tail):
                            v = q_v[idx]
                            neighbor_from = neighbors_idx[v]
                            neighbor_to = neighbors_idx[v+1]
                            points_in_swaps_arr[v, points_in_swaps_arr_idx[v]] = s_id
                            points_in_swaps_arr_idx[v] += 1
                            if q_side[idx]:
                                swaps_Hi_arr[len_swaps_Hi] = v
                                len_swaps_Hi += 1
                                if neighbor_to > neighbor_from:
                                    H_i_neighbor_non_empty = True
                                    for t in range(neighbor_from, neighbor_to):
                                        H_i_neighbor[neighbors[t]] = s_id
                            else:
                                swaps_Hj_arr[len_swaps_Hj] = v
                                len_swaps_Hj += 1
                                if neighbor_to > neighbor_from:
                                    H_j_neighbor_non_empty = True
                                    for t in range(neighbor_from, neighbor_to):
                                        H_j_neighbor[neighbors[t]] = s_id
                        swaps_Hi_index_arr[s_id + 1] = len_swaps_Hi
                        swaps_Hj_index_arr[s_id + 1] = len_swaps_Hj
                        if H_i_neighbor_non_empty:
                            for s in s_ids_arr[j, :s_ids_arr_idx[j]]: # swaps involving j
                                ii = swaps_i_arr[s]
                                jj = swaps_j_arr[s]
                                if ii == j:
                                    jj_from = swaps_Hj_index_arr[s]
                                    jj_to = swaps_Hj_index_arr[s+1]
                                    for u_index in range(jj_from, jj_to):
                                        u = swaps_Hj_arr[u_index]
                                        if visited[u] == stamp:
                                            break
                                        if H_i_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                                elif (jj == j) and (i != ii):
                                    ii_from = swaps_Hi_index_arr[s]
                                    ii_to = swaps_Hi_index_arr[s+1]
                                    for u_index in range(ii_from, ii_to):
                                        u = swaps_Hi_arr[u_index]
                                        if visited[u] == stamp:
                                            break
                                        if H_i_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                            s_ids_arr[j, s_ids_arr_idx[j]] = s_id
                            s_ids_arr_idx[j] += 1
                        if H_j_neighbor_non_empty:
                            for s in s_ids_arr[i, :s_ids_arr_idx[i]]:
                                ii = swaps_i_arr[s]
                                jj = swaps_j_arr[s]
                                if (ii == i) and (j != jj):
                                    jj_from = swaps_Hj_index_arr[s]
                                    jj_to = swaps_Hj_index_arr[s+1]
                                    for u_index in range(jj_from, jj_to):
                                        u = swaps_Hj_arr[u_index]
                                        if visited[u] == stamp:
                                            break
                                        if H_j_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                                elif jj == i:
                                    ii_from = swaps_Hi_index_arr[s]
                                    ii_to = swaps_Hi_index_arr[s+1]
                                    for u_index in range(ii_from, ii_to):
                                        u = swaps_Hi_arr[u_index]
                                        if visited[u] == stamp:
                                            break
                                        if H_j_neighbor[u] == s_id:
                                            adj_swaps_u_arr[len_adj] = s
                                            adj_swaps_v_arr[len_adj] = s_id
                                            len_adj += 1
                                            break
                            s_ids_arr[i, s_ids_arr_idx[i]] = s_id
                            s_ids_arr_idx[i] += 1
                        s_id += 1
                        if s_id >= n_swaps_cap:
                            return swaps_i_arr[:s_id], swaps_j_arr[:s_id], swaps_Hi_arr[:len_swaps_Hi], swaps_Hj_arr[:len_swaps_Hj], swaps_Hi_index_arr[:s_id+1], swaps_Hj_index_arr[:s_id+1], swaps_weights_arr[:s_id], points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_arr[:len_adj], adj_swaps_v_arr[:len_adj]
                for v_idx in range(len_j):
                    v = color_class_arr[j,v_idx]
                    if visited[v] != stamp:
                        delta = -(D[v, j] - D[v, i])
                        if delta < -eps:
                            swaps_i_arr[s_id] = i
                            swaps_j_arr[s_id] = j
                            swaps_Hj_arr[len_swaps_Hj] = v
                            len_swaps_Hj += 1
                            swaps_Hi_index_arr[s_id + 1] = len_swaps_Hi
                            swaps_Hj_index_arr[s_id + 1] = len_swaps_Hj
                            swaps_weights_arr[s_id] = delta
                            points_in_swaps_arr[v, points_in_swaps_arr_idx[v]] = s_id
                            points_in_swaps_arr_idx[v] += 1
                            neighbor_from = neighbors_idx[v]
                            neighbor_to = neighbors_idx[v+1]
                            if neighbor_to > neighbor_from:
                                for t in range(neighbor_from, neighbor_to):
                                    H_j_neighbor[neighbors[t]] = s_id
                                for s in s_ids_arr[i, :s_ids_arr_idx[i]]:
                                    ii = swaps_i_arr[s]
                                    jj = swaps_j_arr[s]
                                    if (ii == i) and (j != jj):
                                        jj_from = swaps_Hj_index_arr[s]
                                        jj_to = swaps_Hj_index_arr[s+1]
                                        for u_index in range(jj_from, jj_to):
                                            u = swaps_Hj_arr[u_index]
                                            if visited[u] == stamp:
                                                break
                                            if H_j_neighbor[u] == s_id:
                                                adj_swaps_u_arr[len_adj] = s
                                                adj_swaps_v_arr[len_adj] = s_id
                                                len_adj += 1
                                                break
                                    elif jj == i:
                                        ii_from = swaps_Hi_index_arr[s]
                                        ii_to = swaps_Hi_index_arr[s+1]
                                        for u_index in range(ii_from, ii_to):
                                            u = swaps_Hi_arr[u_index]
                                            if visited[u] == stamp:
                                                break
                                            if H_j_neighbor[u] == s_id:
                                                adj_swaps_u_arr[len_adj] = s
                                                adj_swaps_v_arr[len_adj] = s_id
                                                len_adj += 1
                                                break
                                s_ids_arr[i, s_ids_arr_idx[i]] = s_id
                                s_ids_arr_idx[i] += 1
                            s_id += 1
                            if s_id >= n_swaps_cap:
                                return swaps_i_arr[:s_id], swaps_j_arr[:s_id], swaps_Hi_arr[:len_swaps_Hi], swaps_Hj_arr[:len_swaps_Hj], swaps_Hi_index_arr[:s_id+1], swaps_Hj_index_arr[:s_id+1], swaps_weights_arr[:s_id], points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_arr[:len_adj], adj_swaps_v_arr[:len_adj]
    return swaps_i_arr[:s_id], swaps_j_arr[:s_id], swaps_Hi_arr[:len_swaps_Hi], swaps_Hj_arr[:len_swaps_Hj], swaps_Hi_index_arr[:s_id+1], swaps_Hj_index_arr[:s_id+1], swaps_weights_arr[:s_id], points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_arr[:len_adj], adj_swaps_v_arr[:len_adj]

@njit(cache=True)
def dsatur_init_numba(membership, vertices, neighbors, neighbors_idx, D_vertices, k, n):
    """
    DSATUR-style initialization that assigns vertices to their nearest feasible centroid.

    Vertices are processed in DSATUR order (non-increasing degree of saturation,
    breaking ties by degree then vertex id). Each vertex is assigned to the nearest
    centroid that does not violate any cannot-link constraint with already-colored
    neighbors. This produces a lower-WCSS starting solution than classic DSATUR
    while still respecting the k-coloring constraint.

    Parameters
    ----------
    membership : ndarray, shape (n_total,)
        Global membership array to be updated in-place for vertices in this component.
    vertices : ndarray, shape (n,)
        Global indices of the vertices in this connected component.
    neighbors : ndarray
        CSR-format local neighbor array for this component.
    neighbors_idx : ndarray
        CSR-format index pointers for the neighbor array.
    D_vertices : ndarray, shape (n, k)
        Distance of each local vertex to each cluster centroid.
    k : int
        Number of clusters (colors).
    n : int
        Number of vertices in this connected component.

    Returns
    -------
    membership : ndarray
        Updated global membership array.
    fitted : int
        1 if a feasible k-coloring was found, 0 otherwise.
    """
    # colors[v] = color assigned to local vertex v, -1 = uncolored
    colors = np.full(n, -1, np.int64)

    # degree[v] = len(neighbors[v])
    degree = np.empty(n, np.int64)
    for i in range(n):
        degree[i] = neighbors_idx[i+1] - neighbors_idx[i]

    # neigh_colors[v, c] = 1 if some neighbor of v has color c
    neigh_colors = np.zeros((n, k), np.bool_)

    # cost_order[v, :] = colors in increasing D[v, c] order, with sentinel k at the end
    cost_order = np.empty((n, k + 1), np.int64)
    for i in range(n):
        # argsort over colors for vertex i
        idx = np.argsort(D_vertices[i])
        for j in range(k):
            cost_order[i, j] = idx[j]
        cost_order[i, k] = k  # sentinel (means "no color left")

    n_colored = 0

    while n_colored < n:
        # --- pick vertex v with max saturation, then max degree, then max id ---
        best_v = -1
        best_sat = -1
        best_deg = -1

        for u in range(n):
            if colors[u] != -1:
                continue  # already colored

            # compute saturation = number of distinct neighbor colors
            sat = 0
            for c in range(k):
                if neigh_colors[u, c]:
                    sat += 1

            if (sat > best_sat or
                (sat == best_sat and
                 (degree[u] > best_deg or
                  (degree[u] == best_deg and u > best_v)))):
                best_sat = sat
                best_deg = degree[u]
                best_v = u

        v = best_v
        if v == -1:
            # should not happen, but just in case
            print('dsatur error')
            return membership, 0

        # --- pick cheapest color for v not used by its neighbors ---
        used_row = neigh_colors[v]
        chosen_color = -1

        for pos in range(k + 1):
            c = cost_order[v, pos]
            if c == k:  # sentinel = "no available color"
                return membership, 0
            if not used_row[c]:
                chosen_color = c
                break

        colors[v] = chosen_color
        n_colored += 1

        # --- update neighbors' saturation info ---
        neighbor_from = neighbors_idx[v]
        neighbor_to = neighbors_idx[v+1]
        for t in range(neighbor_from, neighbor_to):
            w = neighbors[t]
            if colors[w] == -1:
                neigh_colors[w, chosen_color] = True

    # write back to global membership using vertices mapping
    for v in range(n):
        membership[vertices[v]] = colors[v]

    return membership, 1

@njit(cache=True)
def dsatur_color_numba(neighbors, neighbors_idx, n):
    """
    Classic DSATUR graph coloring with an unbounded number of colors.

    Equivalent to the Python dsatur_color(adj):
    - unbounded colors (up to n-1)
    - tie-break: max saturation, then max degree, then max vertex id
    Returns: colors (int64[n]), always feasible for simple graphs.

    Used as a fallback when the cost-aware DSATUR variant (dsatur_init_numba)
    fails to produce a valid k-coloring. The resulting coloring is then mapped
    to the k clusters via a linear assignment if the chromatic number <= k.

    Parameters
    ----------
    neighbors : ndarray
        CSR-format neighbor array.
    neighbors_idx : ndarray
        CSR-format index pointer array.
    n : int
        Number of vertices.

    Returns
    -------
    colors : ndarray, shape (n,)
        Color assigned to each vertex (0-indexed).
    """

    colors = np.full(n, -1, np.int64)

    degree = np.empty(n, np.int64)
    for i in range(n):
        degree[i] = neighbors_idx[i+1] - neighbors_idx[i]

    # neigh_colors[v, c] = 1 if v has some colored neighbor with color c
    neigh_colors = np.zeros((n, n), np.bool_)

    # maintain saturation count incrementally (exactly = number of 1s in row)
    sat = np.zeros(n, np.int64)

    n_colored = 0
    while n_colored < n:
        # pick vertex with max (sat, degree, id)
        best_v = -1
        best_sat = -1
        best_deg = -1

        for u in range(n):
            if colors[u] != -1:
                continue
            su = sat[u]
            du = degree[u]
            if (su > best_sat or
                (su == best_sat and (du > best_deg or
                                     (du == best_deg and u > best_v)))):
                best_sat = su
                best_deg = du
                best_v = u

        v = best_v
        if v == -1:
            # shouldn't happen
            print('DSATUR error')
            return colors

        # choose smallest unused color (0,1,2,...)
        chosen = -1
        for c in range(n):
            if not neigh_colors[v, c]:
                chosen = c
                break

        colors[v] = chosen
        n_colored += 1

        # update neighbors' saturation sets: neigh_colors[w].add(chosen)
        neighbor_from = neighbors_idx[v]
        neighbor_to = neighbors_idx[v+1]
        for t in range(neighbor_from, neighbor_to):
            w = neighbors[t]
            if colors[w] == -1:
                if not neigh_colors[w, chosen]:
                    neigh_colors[w, chosen] = True
                    sat[w] += 1

    return colors

@njit(cache=True)
def compute_sums_numba(data, data2, ml_list_numba, ml_list_numba_idxs, n_supernodes, d):
    """
    Aggregate raw data into super-node feature sums for distance computations.

    For each super-node (must-link equivalence class), computes the sum of
    feature vectors and the sum of squared L2 norms of its member data points.
    These aggregated values replace individual point computations in all
    centroid-distance calculations, exploiting the must-link structure.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Raw data points.
    data2 : ndarray, shape (n,)
        Element-wise squared L2 norm of each data point (||y_i||^2).
    ml_list_numba : ndarray
        Concatenated member indices for all super-nodes (CSR values array).
    ml_list_numba_idxs : ndarray, shape (n_supernodes + 1,)
        CSR-style pointer array; members of super-node i are
        ml_list_numba[ml_list_numba_idxs[i]:ml_list_numba_idxs[i+1]].
    n_supernodes : int
        Number of super-nodes.
    d : int
        Feature dimension.

    Returns
    -------
    y_values : ndarray, shape (n_supernodes, d)
        Sum of feature vectors for each super-node.
    y_values2 : ndarray, shape (n_supernodes,)
        Sum of squared L2 norms for each super-node.
    """
    y_values  = np.empty((n_supernodes, d), dtype=data.dtype)
    y_values2 = np.empty(n_supernodes, dtype=data2.dtype)

    for i in range(n_supernodes):
        idxs_from = ml_list_numba_idxs[i]
        idxs_to = ml_list_numba_idxs[i+1]
        s  = np.zeros(d, dtype=data.dtype)
        s2 = 0.0
        for idx in range(idxs_from, idxs_to):
            j = ml_list_numba[idx]
            s  += data[j]
            s2 += data2[j]
        y_values[i]  = s
        y_values2[i] = s2
    return y_values, y_values2

def f2(C):
    """
    Compute squared L2 norms of rows in C.

    Parameters
    ----------
    C : ndarray, shape (k, d)
        Matrix of k row vectors.

    Returns
    -------
    C2 : ndarray, shape (k,)
        C2[i] = ||C[i]||^2.
    """
    # shape: (k,d) to (k,)
    C2 = np.einsum('ij,ij->i', C, C)
    return C2

def InitAssign_Singletons_and_Cliques(membership, sub_adjs, D):
    """
    Initialize cluster assignments for singletons and clique components.

    Singletons are assigned to their nearest centroid. Clique components
    (complete subgraphs) are assigned via the linear sum assignment (Hungarian
    algorithm) to find the minimum-cost bijection from clique vertices to
    cluster labels, guaranteeing no two clique members share a cluster.

    Parameters
    ----------
    membership : ndarray, shape (n_supernodes,)
        Cluster assignment array to be updated in-place.
    sub_adjs : dict
        Preprocessed connected-component data with keys 'singletons', 'cliques', 'others'.
    D : ndarray, shape (n_supernodes, k)
        Distance matrix from each super-node to each centroid.

    Returns
    -------
    membership : ndarray
        Updated membership array.
    """
    verts = sub_adjs['singletons']
    D_verts = D[verts]
    best_j = np.argmin(D_verts, axis=1)
    for v, j_new in zip(verts, best_j):
        membership[v] = j_new

    for verts in sub_adjs['cliques']:
        D_verts = D[verts]
        _, best_j = linear_sum_assignment(D_verts)
        for v, j_new in zip(verts, best_j):
            membership[v] = j_new
    return membership

def Assign_Singletons_and_Cliques(membership, sub_adjs, D, n_s):
    """
    Reassign singletons and clique components to improve WCSS, tracking changes.

    Same logic as InitAssign_Singletons_and_Cliques but also counts the number
    of components that changed (n_s) for convergence detection.

    Parameters
    ----------
    membership : ndarray, shape (n_supernodes,)
        Current cluster assignments (updated in-place).
    sub_adjs : dict
        Preprocessed connected-component data.
    D : ndarray, shape (n_supernodes, k)
        Distance matrix from each super-node to each centroid.
    n_s : int
        Running count of assignment steps that produced changes.

    Returns
    -------
    membership : ndarray
        Updated membership array.
    n_s : int
        Updated count of improving assignment steps.
    """
    verts = sub_adjs['singletons']
    D_verts = D[verts]
    best_j = np.argmin(D_verts, axis=1)
    current_j = membership[verts]
    move_mask = (current_j != best_j)

    moved_verts = verts[move_mask]
    if moved_verts.size:
        membership[moved_verts] = best_j[move_mask]
        n_s += 1

    for verts in sub_adjs['cliques']:
        D_verts = D[verts]
        current_j = membership[verts]
        _, best_j = linear_sum_assignment(D_verts)
        move_mask = (current_j != best_j)
        moved_verts = verts[move_mask]
        if moved_verts.size:
            membership[moved_verts] = best_j[move_mask]
            n_s += 1
    return membership, n_s

def distance_matrix(y_values, C, C2, ml_count):
    """
    Compute the weighted squared distance from each super-node to each centroid.

    Uses the identity ||sum_y - |H|*mu||^2 = ||sum_y||^2 - 2*(sum_y . mu) + |H|*||mu||^2
    to compute distances efficiently via matrix multiplication, avoiding
    per-point iteration.

    Parameters
    ----------
    y_values : ndarray, shape (n_supernodes, d)
        Aggregated feature sum for each super-node.
    C : ndarray, shape (k, d)
        Cluster centroids.
    C2 : ndarray, shape (k,)
        Squared L2 norm of each centroid (||mu_j||^2).
    ml_count : ndarray, shape (n_supernodes,)
        Number of raw data points in each super-node.

    Returns
    -------
    D : ndarray, shape (n_supernodes, k)
        D[i, j] = ||y_i - |H_i| * mu_j||^2_F (up to a constant not depending on j).
    """
    D = - 2.0 * (y_values @ C.T) + ml_count[:, None] * C2[None, :]
    D.round(5, out=D)
    return D

def ExactInitAssignment(n_supernodes, graph_coloring_problems, sub_adjs, D, k):
    """
    Compute a feasible initial assignment by solving GCP subproblems exactly.

    Used in KSKM_E when both the cost-aware and classic DSATUR heuristics fail
    to produce a feasible k-coloring. Each non-trivial connected component's
    coloring is solved as an ILP (graph coloring problem) via Gurobi.

    Parameters
    ----------
    n_supernodes : int
        Number of super-nodes.
    graph_coloring_problems : list of (gurobipy.Model, dict)
        Pre-built Gurobi models for each non-trivial component.
    sub_adjs : dict
        Preprocessed connected-component data.
    D : ndarray, shape (n_supernodes, k)
        Distance matrix from each super-node to each centroid.
    k : int
        Number of clusters.

    Returns
    -------
    membership : ndarray, shape (n_supernodes,)
        Initial feasible cluster assignment.
    """
    membership = np.zeros(n_supernodes, dtype = np.int64)
    membership = InitAssign_Singletons_and_Cliques(membership, sub_adjs, D)

    for i in range(len(sub_adjs['others'])):
        vertices, _, _, n_vertices = sub_adjs['others'][i]
        m, x = graph_coloring_problems[i]
        D_vertices = D[vertices]
        membership = solve_graph_coloring_problem(membership, m, x, vertices, n_vertices, D_vertices, k, start_solution = False)

    return membership

def exact_assignment_given_centroids_and_Kempe_Assignment(k, graph_coloring_problems, membership, y_values, y_values2, sub_adjs, n_supernodes, d, ml_count, verbose, time_limit, explored = []):
    """
    Alternate between Kempe-swap local search and exact GCP until convergence or time limit.

    This implements the descent phase of KSKM_E: repeatedly applies KSAssignment
    to reach a Kempe-swap local optimum, then invokes Gurobi to solve the fixed-
    centroid graph coloring problem (GCP) to escape Kempe-swap local optima.
    The GCP is skipped if its objective value was already explored or time runs out.

    Parameters
    ----------
    k : int
        Number of clusters.
    graph_coloring_problems : list of (gurobipy.Model, dict)
        Pre-built Gurobi GCP models for each non-trivial component.
    membership : ndarray, shape (n_supernodes,)
        Current cluster assignments.
    y_values : ndarray, shape (n_supernodes, d)
        Aggregated feature sums for each super-node.
    y_values2 : ndarray, shape (n_supernodes,)
        Aggregated squared norms for each super-node.
    sub_adjs : dict
        Preprocessed connected-component data.
    n_supernodes : int
        Number of super-nodes.
    d : int
        Feature dimension.
    ml_count : ndarray, shape (n_supernodes,)
        Number of raw data points in each super-node.
    verbose : bool
        Whether to print objective values.
    time_limit : float
        Wall-clock time budget in seconds.
    explored : list of float
        WCSS values already seen; GCP is skipped if current WCSS is already listed.

    Returns
    -------
    membership : ndarray
        Improved cluster assignments.
    """
    start_time = time.time()
    while True:
        while True:
            D, obj = CentroidUpdate(membership, y_values, y_values2, k, ml_count, n_supernodes, d)
            membership, n_s = KempeChainMutation_target_centroids(k, membership, sub_adjs, D)
            if not n_s:
                break
        time_used = time.time() - start_time
        time_left = time_limit - time_used
        if time_left < 0 or obj in explored:
            return membership
        membership, n_s_e = ExactAssignment(k, graph_coloring_problems, membership, sub_adjs, D, time_limit = time_left, skip_phases = True)
        explored += [obj]
        if verbose:
            print(obj)
            print(f"remaining time: {round(time_left,2)}")
        if not n_s_e:
            return membership

def ExactAssignment(k, graph_coloring_problems, membership, sub_adjs, D, time_limit, skip_phases = True):
    """
    Improve cluster assignments by solving fixed-centroid GCP subproblems via Gurobi.

    For each non-trivial connected component, the current assignment is used as
    a warm start for the ILP. Gurobi terminates as soon as it finds a strictly
    improving incumbent (first-improvement callback). Only components whose
    subproblem is not infeasible or over-time are updated.

    Parameters
    ----------
    k : int
        Number of clusters.
    graph_coloring_problems : list of (gurobipy.Model, dict)
        Pre-built Gurobi GCP models.
    membership : ndarray, shape (n_supernodes,)
        Current cluster assignments (updated in-place).
    sub_adjs : dict
        Preprocessed connected-component data.
    D : ndarray, shape (n_supernodes, k)
        Distance matrix from each super-node to each centroid.
    time_limit : float
        Remaining wall-clock budget in seconds.
    skip_phases : bool
        If False, also re-assign singletons and cliques before solving GCPs.

    Returns
    -------
    membership : ndarray
        Updated cluster assignments.
    n_s : int
        Number of components where an improving solution was found.
    """
    start_time = time.time()
    n_s = 0
    if not skip_phases:
        membership, n_s = Assign_Singletons_and_Cliques(membership, sub_adjs, D, n_s)

    for i in range(len(sub_adjs['others'])):
        vertices, _, _, n_vertices = sub_adjs['others'][i]
        m, x = graph_coloring_problems[i]
        D_vertices = D[vertices]
        time_used = time.time() - start_time
        time_left = time_limit - time_used
        if time_left < 0:
            return membership, n_s
        try:
            membership, n_s_inner = solve_graph_coloring_problem(membership, m, x, vertices, n_vertices, D_vertices, k, time_limit = time_left)
        except Exception as e:
            print("An error occurred:", e)
            n_s_inner = 0
        n_s += n_s_inner

    return membership, n_s

def solve_graph_coloring_problem(membership, m, x, vertices, n_vertices, D, k, start_solution = True, time_limit = 3600):
    """
    Solve a fixed-centroid graph coloring ILP for one connected component via Gurobi.

    Two modes are supported:
    - start_solution=True (default): uses the current membership as a warm start
      and terminates as soon as a strictly improving feasible solution is found.
    - start_solution=False: finds the first feasible solution (used for initialization).

    Parameters
    ----------
    membership : ndarray, shape (n_total,)
        Global membership array; updated in-place for vertices in this component.
    m : gurobipy.Model
        Pre-built Gurobi model with coloring constraints.
    x : dict of gurobipy.Var
        Binary variables x[i, c] indicating if local vertex i is assigned color c.
    vertices : ndarray, shape (n_vertices,)
        Global indices of vertices in this component.
    n_vertices : int
        Number of vertices in this component.
    D : ndarray, shape (n_vertices, k)
        Distance from each local vertex to each centroid.
    k : int
        Number of clusters.
    start_solution : bool
        Whether to warm-start from current membership.
    time_limit : float
        Wall-clock time budget in seconds for Gurobi.

    Returns
    -------
    membership : ndarray
        Updated global membership.
    n_s : int
        1 if an improving solution was found, 0 otherwise (only when start_solution=True).
    """
    def first_improvement_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # objective value of the new incumbent
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

            # minimization: improved if strictly lower than starting objective
            if obj < model._start_obj - 1e-5:   # tolerance to avoid noise
                # model._improved_found = True
                model.terminate()
    def build_stop_at_first_incumbent_callback():
        found = {"flag": False}      # mutable container (not global)
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                if not found["flag"]:
                    found["flag"] = True
                    print("First feasible solution found. Stopping...")
                    model.terminate()
        return callback
    m.Params.TimeLimit = time_limit
    m.setObjective(
        gp.quicksum([D[i, c] * x[i, c]
                    for i in range(n_vertices)
                    for c in range(k)]),
        GRB.MINIMIZE
    )
    if start_solution:
        obj_old = 0
        for i, v in enumerate(vertices):
            for c in range(k):
                x[i, c].Start = 0
            x[i, membership[v]].Start = 1
            obj_old += D[i, membership[v]]
        obj_old = round(obj_old, 5)
        m._start_obj = obj_old
        m.optimize(first_improvement_callback)
        # m.optimize()
        for i, v in enumerate(vertices):
            for c in range(k):
                if x[i, c].X > 0.5:
                    membership[v] = c
                    break
        obj_new = round(m.objVal)
        if obj_new < obj_old - 0.00001:
            return membership, 1
        return membership, 0
    else:
        m.Params.TimeLimit = time_limit
        cb = build_stop_at_first_incumbent_callback()
        m.optimize(cb)
        for i, v in enumerate(vertices):
            for c in range(k):
                if x[i, c].X > 0.5:
                    membership[v] = c
                    break
        return membership

def build_graph_coloring_problems(sub_adjs, k):
    """
    Pre-build Gurobi ILP models for the fixed-centroid graph coloring problem.

    For each non-trivial connected component in sub_adjs['others'], constructs
    a binary program with:
    - One binary variable x[i, c] per vertex i and color c.
    - Cannot-link constraints: x[u,c] + x[v,c] <= 1 for each edge (u,v) and color c.
    - Assignment constraints: sum_c x[i,c] == 1 for each vertex i.
    The objective is set later per-iteration in solve_graph_coloring_problem.

    Parameters
    ----------
    sub_adjs : dict
        Preprocessed connected-component data (only 'others' components are used).
    k : int
        Number of clusters (colors).

    Returns
    -------
    graph_coloring_problems : list of (gurobipy.Model, dict)
        One (model, x) tuple per non-trivial component.
    """
    def build_graph_coloring_problem(n_vertices, neighbors, neighbors_idx, k):
        m = gp.Model()
        m.Params.OutputFlag = 0
        m.Params.MIPFocus = 1
        x = m.addVars(n_vertices, k, vtype=GRB.BINARY)
        for u in range(n_vertices):
            neighbor_from = neighbors_idx[u]
            neighbor_to = neighbors_idx[u+1]
            for t in range(neighbor_from, neighbor_to):
                v = neighbors[t]
                if u < v:
                    m.addConstrs(x[u,c] + x[v,c] <= 1 for c in range(k))
        m.addConstrs(x.sum(i, '*') == 1 for i in range(n_vertices))
        m.update()
        return (m, x)

    graph_coloring_problems = []
    for _, neighbors, neighbors_idx, n_vertices in sub_adjs['others']:
        graph_coloring_problems.append(build_graph_coloring_problem(n_vertices, neighbors, neighbors_idx, k))
    return graph_coloring_problems

def CentroidUpdate(membership, y_values, y_values2, k, ml_count, n_supernodes, d, reposition = False):
    """
    Recompute cluster centroids and the distance matrix from current assignments.

    Handles empty clusters by randomly repositioning their centroid to an
    existing super-node. Optionally implements centroid reposition (KSShift):
    shifts the centroid of the best-fitting cluster toward the worst-fitting
    cluster to encourage exploration.

    Parameters
    ----------
    membership : ndarray, shape (n_supernodes,)
        Current cluster assignments.
    y_values : ndarray, shape (n_supernodes, d)
        Aggregated feature sums for each super-node.
    y_values2 : ndarray, shape (n_supernodes,)
        Aggregated squared norms for each super-node.
    k : int
        Number of clusters.
    ml_count : ndarray, shape (n_supernodes,)
        Number of raw data points in each super-node.
    n_supernodes : int
        Number of super-nodes.
    d : int
        Feature dimension.
    reposition : bool
        If True, replaces the best-cluster centroid with the worst-cluster centroid
        (centroid reposition / KSShift mutation) and returns only D.

    Returns
    -------
    D : ndarray, shape (n_supernodes, k)
        Updated distance matrix.
    total_ssr : float
        Total within-cluster sum of squares (only returned when reposition=False).
    """
    C = np.empty((k, d))

    sums, sum_sq, counts = CentroidUpdate_assist(membership, y_values, y_values2, k, ml_count, n_supernodes, d)

    nonempty       = counts > 0
    nonempty_clusters = np.flatnonzero(nonempty)
    empty_clusters_size = k - nonempty_clusters.size

    C[nonempty]  = sums[nonempty] / counts[nonempty, None]
    if empty_clusters_size:
        next_idx = np.random.randint(n_supernodes, size = empty_clusters_size)
        C[~nonempty] = y_values[next_idx] / ml_count[next_idx, None]
    C2 = np.einsum('ij,ij->i', C, C)

    ssr_nonempty = sum_sq[nonempty] - counts[nonempty] * C2[nonempty]

    D = distance_matrix(y_values, C, C2, ml_count)
    if reposition:
        rank = np.argsort(ssr_nonempty)
        a = nonempty_clusters[rank[0]]
        b = nonempty_clusters[rank[-1]]
        D[:,a] = D[:,b]
        return D
    total_ssr    = float(ssr_nonempty.sum())
    return D, round(total_ssr, 5)

def CentroidUpdate_random(membership, y_values, k, ml_count, d, data, ml_map):
    """
    Compute a distance matrix based on perturbed centroids (KSPerturb / centroid perturbation).

    Samples perturbed centroids from the empirical cluster distribution using
    the multivariate t-distribution heuristic described in Section 2.2.2 of the paper.
    Clusters with higher within-cluster variance receive larger perturbations,
    encouraging exploration where fit quality is lower. Empty clusters are
    repositioned to a random data point.

    Parameters
    ----------
    membership : ndarray, shape (n_supernodes,)
        Current cluster assignments.
    y_values : ndarray, shape (n_supernodes, d)
        Aggregated feature sums for each super-node.
    k : int
        Number of clusters.
    ml_count : ndarray, shape (n_supernodes,)
        Number of raw data points in each super-node.
    d : int
        Feature dimension.
    data : ndarray, shape (n, d)
        Raw data (used to reposition empty cluster centroids).
    ml_map : ndarray, shape (n,)
        Maps each raw data point to its super-node index.

    Returns
    -------
    D : ndarray, shape (n_supernodes, k)
        Distance matrix based on perturbed centroids.
    """
    vec, vec_sq, counts = RandomCentroid_assist(data, membership, ml_map, k, d)
    nonempty       = counts > 0
    mask_ge2       = counts > 1
    vec[nonempty]  = vec[nonempty] / counts[nonempty, None]
    vec_sq[nonempty]  = vec_sq[nonempty] / counts[nonempty, None]

    C_var = np.maximum(0, vec_sq - vec**2)
    C_var[mask_ge2] = C_var[mask_ge2] / (counts[mask_ge2, None] - 1)

    C = vec + np.random.randn(k,d) * np.sqrt(C_var)
    empty_clusters_size = (~nonempty).sum()
    if empty_clusters_size:
        random_idxs = np.random.randint(ml_map.size, size = empty_clusters_size)
        C[~nonempty] = data[random_idxs]
    C2 = np.einsum('ij,ij->i', C, C)

    D = distance_matrix(y_values, C, C2, ml_count)

    return D

def KempeChainMWSP(k, membership, y_values, y_values2, sub_adjs, n_supernodes, d, ml_count, deepest = False, verbose = True):
    """
    Run Kempe-swap local search until no improving swap exists (steepest descent).

    Alternates between centroid updates and KSAssignment (or the stricter
    KempeChainMutation_target_centroids if deepest=True) until convergence.
    This is the core descent loop of KSKM.

    Parameters
    ----------
    k : int
        Number of clusters.
    membership : ndarray, shape (n_supernodes,)
        Current cluster assignments (updated in-place through iterations).
    y_values : ndarray, shape (n_supernodes, d)
        Aggregated feature sums for each super-node.
    y_values2 : ndarray, shape (n_supernodes,)
        Aggregated squared norms for each super-node.
    sub_adjs : dict
        Preprocessed connected-component data.
    n_supernodes : int
        Number of super-nodes.
    d : int
        Feature dimension.
    ml_count : ndarray, shape (n_supernodes,)
        Number of raw data points in each super-node.
    deepest : bool
        If True, use KempeChainMutation_target_centroids (exhaustive inner loop);
        otherwise use a single pass of KSAssignment per centroid update.
    verbose : bool
        Whether to print WCSS at each iteration.

    Returns
    -------
    membership : ndarray
        Locally optimal cluster assignments (no improving Kempe swap exists).
    """
    while True:
        D, obj = CentroidUpdate(membership, y_values, y_values2, k, ml_count, n_supernodes, d)
        if verbose:
            print(obj)

        if deepest:
            membership, n_s = KempeChainMutation_target_centroids(k, membership, sub_adjs, D)
        else:
            membership, n_s = KSAssignment(k, membership, sub_adjs, D)

        if not n_s:
            return membership

def KempeChainMutation_target_centroids(k, membership, sub_adjs, D):
    """
    Apply KSAssignment repeatedly until no improving Kempe swap is found (fixed centroids).

    Given a fixed distance matrix D, repeatedly calls KSAssignment until
    convergence. This implements the inner fixed-centroid descent loop used
    both during initialization and after centroid perturbations.

    Parameters
    ----------
    k : int
        Number of clusters.
    membership : ndarray, shape (n_supernodes,)
        Current cluster assignments.
    sub_adjs : dict
        Preprocessed connected-component data.
    D : ndarray, shape (n_supernodes, k)
        Fixed distance matrix.

    Returns
    -------
    membership : ndarray
        Updated cluster assignments (Kempe-swap local optimum for fixed D).
    n_s : int
        Total number of improving assignment steps performed.
    """
    n_s = 0
    while True:
        membership, n_s_inner = KSAssignment(k, membership, sub_adjs, D, skip_phases = n_s)
        n_s += n_s_inner
        if not n_s_inner:
            return membership, n_s

def KempeChainMutation_distr(k, membership, y_values, sub_adjs, d, data, ml_map, ml_count):
    """
    Perturb centroids and immediately descend via Kempe swaps (KSPerturb step).

    Samples perturbed centroids from the empirical cluster distribution
    (CentroidUpdate_random) and applies one round of Kempe-swap descent.
    This implements the neighborhood perturbation mutation of KSKM to escape
    local optima.

    Parameters
    ----------
    k : int
        Number of clusters.
    membership : ndarray, shape (n_supernodes,)
        Current cluster assignments.
    y_values : ndarray, shape (n_supernodes, d)
        Aggregated feature sums for each super-node.
    sub_adjs : dict
        Preprocessed connected-component data.
    d : int
        Feature dimension.
    data : ndarray, shape (n, d)
        Raw data points.
    ml_map : ndarray, shape (n,)
        Maps each raw data point to its super-node index.
    ml_count : ndarray, shape (n_supernodes,)
        Number of raw data points in each super-node.

    Returns
    -------
    membership : ndarray
        Cluster assignments after perturbation and Kempe-swap descent.
    """
    D = CentroidUpdate_random(membership, y_values, k, ml_count, d, data, ml_map)
    membership, _ = KempeChainMutation_target_centroids(k, membership, sub_adjs, D)
    return membership

def KSAssignment(k, membership, sub_adjs, D, skip_phases = False):
    """
    Perform one round of Kempe-swap cluster reassignment (Algorithm 2 of the paper).

    For k=2: uses a direct BFS-based Kempe chain enumeration.
    For k>2: calls build_MWSP_core to enumerate all improving Kempe chains,
    then solves a Maximum Weight Independent Set (MWIS) problem via Gurobi to
    select a compatible, maximum-weight subset of swaps.

    Optionally first reassigns singletons and cliques (skip_phases=False).

    Parameters
    ----------
    k : int
        Number of clusters.
    membership : ndarray, shape (n_supernodes,)
        Current cluster assignments (updated in-place).
    sub_adjs : dict
        Preprocessed connected-component data with 'singletons', 'cliques', 'others'.
    D : ndarray, shape (n_supernodes, k)
        Distance matrix from each super-node to each centroid.
    skip_phases : bool or int
        If falsy, also reassign singletons and cliques before processing 'others'.

    Returns
    -------
    membership : ndarray
        Updated cluster assignments.
    n_s : int
        Number of improving moves applied (0 = convergence at this centroid).
    """
    def apply_swaps(vertices, membership, chosen, swaps_i_array, swaps_j_array, swaps_Hi_array, swaps_Hj_array, swaps_Hi_index_array, swaps_Hj_index_array):
        cnt = ((swaps_Hi_index_array[chosen + 1] - swaps_Hi_index_array[chosen]).sum()
                + (swaps_Hj_index_array[chosen + 1] - swaps_Hj_index_array[chosen]).sum()
                )
        idx = np.empty(cnt, dtype=vertices.dtype)
        val = np.empty(cnt, dtype=membership.dtype)
        pos = 0
        for sid in chosen:
            i = swaps_i_array[sid]
            j = swaps_j_array[sid]
            a = swaps_Hi_index_array[sid]; b = swaps_Hi_index_array[sid+1]
            hi = swaps_Hi_array[a:b]
            vi = vertices[hi]
            L = vi.size
            idx[pos:pos+L] = vi
            val[pos:pos+L] = j
            pos += L
            a = swaps_Hj_index_array[sid]; b = swaps_Hj_index_array[sid+1]
            hj = swaps_Hj_array[a:b]
            vj = vertices[hj]
            L = vj.size
            idx[pos:pos+L] = vj
            val[pos:pos+L] = i
            pos += L
        membership[idx] = val
    def MWSP(weights,
            points_in_swaps_arr,
            points_in_swaps_arr_idx,
            adj_swaps_u_array,
            adj_swaps_v_array):
        """
        Solve the Maximum Weight Independent Set (MWIS) problem over candidate Kempe swaps.

        Selects a subset of improving Kempe swaps that are mutually compatible:
        - Clique constraints (CCswap): each super-node participates in at most one swap.
        - Cannot-link constraints (CLswap): two swaps sharing a cannot-link edge cannot both be applied.

        Parameters
        ----------
        weights : ndarray, shape (n_swaps,)
            Swap cost delta values (negative = improving; minimization objective).
        points_in_swaps_arr : ndarray, shape (n_vertices, k-1)
            For each vertex, which swap IDs include it.
        points_in_swaps_arr_idx : ndarray, shape (n_vertices,)
            Number of valid entries per row of points_in_swaps_arr.
        adj_swaps_u_array : ndarray
            First endpoint of each swap-vs-swap conflict edge.
        adj_swaps_v_array : ndarray
            Second endpoint of each swap-vs-swap conflict edge.

        Returns
        -------
        chosen : ndarray
            Indices of selected swaps forming the maximum weight independent set.
        """
        m = gp.Model("MWSP")
        m.Params.OutputFlag = 0
        m.Params.MIPFocus = 1
        m.Params.TimeLimit = 600
        # Decision vector
        x = m.addMVar(shape=weights.size, vtype=GRB.BINARY)
        m.setObjective(weights @ x, GRB.MINIMIZE)
        # (1) Packing constraints: for each point, at most one swap that contains it
        for v in range(len(points_in_swaps_arr_idx)):
            clique_len = points_in_swaps_arr_idx[v]
            if clique_len > 1:  # skip empty incidence lists
                clique = points_in_swaps_arr[v, :clique_len]
                # clique is an array of indices into x; x[clique].sum() is an MVar sum
                m.addConstr(x[clique].sum() <= 1)
        # (2) Pairwise conflicts: x[i] + x[j] <= 1
        m.addConstr(x[adj_swaps_u_array] + x[adj_swaps_v_array] <= 1)
        m.optimize()
        # Extract chosen indices
        chosen = np.flatnonzero(x.X > 0.5)
        return chosen

    n_s = 0
    if not skip_phases:
        membership, n_s = Assign_Singletons_and_Cliques(membership, sub_adjs, D, n_s)
    if k == 2:
        eps = 1e-6
        for vertices, neighbors, neighbors_idx, n_vertices in sub_adjs['others']:
            membership_vertices = membership[vertices]
            color_class_i = []
            color_class_j = []
            for local_idx, c in enumerate(membership_vertices):
                if c:
                    color_class_j.append(local_idx)
                else:
                    color_class_i.append(local_idx)
            i = 0
            j = 1
            D_c = D[vertices, j] - D[vertices, i]
            if not color_class_i:
                for v in color_class_j:
                    delta = -D_c[v]
                    if delta < -eps:
                        membership[vertices[v]] = i
            elif not color_class_j:
                for v in color_class_i:
                    delta = D_c[v]
                    if delta < -eps:
                        membership[vertices[v]] = j
            else:
                visited = np.zeros(n_vertices, dtype=np.bool_)
                q_v = np.empty(n_vertices, dtype=np.int64)
                q_side = np.empty(n_vertices, dtype=np.bool_)
                for start in color_class_i:
                    if visited[start]:
                        continue
                    head = 0
                    tail = 0
                    q_v[tail] = start
                    q_side[tail] = True
                    tail += 1
                    visited[start] = True
                    delta = 0.0
                    while head < tail:
                        v = q_v[head]
                        side = q_side[head]
                        head += 1
                        if side:
                            delta += D_c[v]
                            neighbor_from = neighbors_idx[v]
                            neighbor_to = neighbors_idx[v+1]
                            for t in range(neighbor_from, neighbor_to):
                                u = neighbors[t]
                                if (membership_vertices[u] == j) and (not visited[u]):
                                    visited[u] = True
                                    q_v[tail] = u
                                    q_side[tail] = False
                                    tail += 1
                        else:
                            delta -= D_c[v]
                            neighbor_from = neighbors_idx[v]
                            neighbor_to = neighbors_idx[v+1]
                            for t in range(neighbor_from, neighbor_to):
                                u = neighbors[t]
                                if (membership_vertices[u] == i) and (not visited[u]):
                                    visited[u] = True
                                    q_v[tail] = u
                                    q_side[tail] = True
                                    tail += 1
                    if delta < -eps:
                        membership[vertices[q_v[:tail]]] = q_side[:tail]
                for v in color_class_j:
                    if not visited[v]:
                        delta = -D_c[v]
                        if delta < -eps:
                            membership[vertices[v]] = i
    else:
        for vertices, neighbors, neighbors_idx, n_vertices in sub_adjs['others']:
            D_vertices  = D[vertices]
            membership_vertices = membership[vertices]
            n_swaps_cap = min(80000, int(n_vertices * (n_vertices - 1)/2))
            swaps_i_array, swaps_j_array, swaps_Hi_array, swaps_Hj_array, swaps_Hi_index_array, swaps_Hj_index_array, swaps_weights_array, points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_array, adj_swaps_v_array = build_MWSP_core(D_vertices, neighbors, neighbors_idx, membership_vertices, k, n_vertices, n_swaps_cap)
            if swaps_i_array.size:
                chosen = MWSP(swaps_weights_array, points_in_swaps_arr, points_in_swaps_arr_idx, adj_swaps_u_array, adj_swaps_v_array)
                apply_swaps(vertices, membership, chosen, swaps_i_array, swaps_j_array, swaps_Hi_array, swaps_Hj_array, swaps_Hi_index_array, swaps_Hj_index_array)
                n_s += chosen.size

    return membership, n_s

def KSKM(random_state, data, ml, cl, steps_mutation, k, steps_back_to_best = 3, steps_no_improvement = 10, verbose = False, time_limit = 3600):
    """
    Kempe Swap K-Means: main entry point for constrained clustering (Algorithm 1).

    Implements the full KSKM algorithm:
    1. Preprocessing: merges must-link groups into super-nodes and builds the
       cannot-link constraint graph.
    2. Initialization: k-means++ centroid initialization followed by DSATUR-based
       feasible cluster assignment.
    3. Descent: Kempe-swap local search (KempeChainMWSP) to a local optimum.
    4. Iterated local search: alternates between centroid perturbation (KSPerturb)
       and centroid reposition (KSShift) mutations followed by Kempe-swap descent,
       keeping track of the best solution found.

    Parameters
    ----------
    random_state : int
        Random seed for k-means++ initialization.
    data : ndarray, shape (n, d)
        Data points to cluster.
    ml : list of (int, int)
        Must-link constraint pairs (indices into data).
    cl : list of (int, int)
        Cannot-link constraint pairs (indices into data).
    steps_mutation : int
        Maximum number of perturbation/mutation iterations.
    k : int
        Number of clusters.
    steps_back_to_best : int
        Number of non-improving iterations before reverting to best known solution.
    steps_no_improvement : int
        Number of consecutive non-improving iterations before early termination.
    verbose : bool
        Whether to print progress information.
    time_limit : float
        Wall-clock time budget in seconds.

    Returns
    -------
    membership_final : ndarray, shape (n,)
        Cluster label for each data point (0-indexed), or [] if no feasible
        solution was found.
    """
    def KSKM_inner(k, membership, y_values, y_values2, sub_adjs, n_supernodes, d, data, ml_map, ml_count, steps_mutation, steps_back_to_best, steps_no_improvement, verbose = True, time_limit = 3600, reposition_frequency = 3):
        start_time = time.time()
        count = 0
        if verbose:
            print('descence')
        membership = KempeChainMWSP(k, membership, y_values, y_values2, sub_adjs, n_supernodes, d, ml_count, verbose = verbose)
        _, best_obj = CentroidUpdate(membership, y_values, y_values2, k, ml_count, n_supernodes, d)
        best_membership = copy.deepcopy(membership)
        for i in range(steps_mutation):
            time_used = time.time() - start_time
            if time_used > time_limit:
                return best_membership, best_obj
            if i % reposition_frequency == 0:
                if verbose:
                    print('mutate: reposition')
                D = CentroidUpdate(membership, y_values, y_values2, k, ml_count, n_supernodes, d, reposition = True)
                membership, _ = KempeChainMutation_target_centroids(k, membership, sub_adjs, D)
            else:
                if verbose:
                    print('mutate: perturbation')
                membership = KempeChainMutation_distr(k, membership, y_values, sub_adjs, d, data, ml_map, ml_count)

            if verbose:
                print('descence')
            membership = KempeChainMWSP(k, membership, y_values, y_values2, sub_adjs, n_supernodes, d, ml_count, verbose = verbose)

            _, obj = CentroidUpdate(membership, y_values, y_values2, k, ml_count, n_supernodes, d)
            if obj < best_obj:
                best_obj = obj
                if verbose:
                    print('update best obj')
                    print(best_obj)
                count = 0
                best_membership = copy.deepcopy(membership)

            else:
                count += 1
                if count > steps_back_to_best:
                    membership = copy.deepcopy(best_membership)
                    if count > steps_no_improvement:
                        break

        return best_membership, best_obj
    start_time = time.time()
    y_values, y_values2, adj, ml_map, ml_count, n_supernodes, d = preprocessing(data, cl, ml)
    sub_adjs = sub_adj_classification(adj)
    del cl, ml, adj
    C, _ = kmeans_plusplus(data, k, random_state = random_state)
    C2 = f2(C)
    D = distance_matrix(y_values, C, C2, ml_count)
    del C, C2
    membership = DSATUR_KM(n_supernodes, sub_adjs, D, k)
    if not len(membership):
        membership = DSATUR(n_supernodes, sub_adjs, k)
        if len(membership):
            if verbose:
                print('found DSATUR solution')
        else:
            print('DSATUR solution infeasible')
            return []
    membership, _ = KempeChainMutation_target_centroids(k, membership, sub_adjs, D)
    if verbose:
        print('Initialization done')
    time_lefted = time_limit - time.time() + start_time
    membership, _ = KSKM_inner(k, membership, y_values, y_values2, sub_adjs, n_supernodes, d, data, ml_map, ml_count, steps_mutation, steps_back_to_best, steps_no_improvement, verbose, time_limit = time_lefted)
    membership_final = recover_ml_from_membership(membership, ml_map)

    return membership_final

def KSKM_E(random_state, data, ml, cl, steps_mutation, k, steps_back_to_best = 3, steps_no_improvement = 10, verbose = False, time_limit = 3600):
    """
    Kempe Swap K-Means with Exact GCP refinement (KSKM-E).

    Extends KSKM by replacing the pure Kempe-swap descent with
    exact_assignment_given_centroids_and_Kempe_Assignment, which alternates
    Kempe-swap local search with Gurobi-based exact graph coloring (GCP) to
    escape Kempe-swap local optima. The GCP step is triggered only when
    Kempe swaps can no longer improve the solution, and is skipped if its
    objective value has already been explored.

    This variant is more computationally expensive than KSKM but can achieve
    better solutions when the chromatic number is close to k.

    Parameters
    ----------
    random_state : int
        Random seed for k-means++ initialization.
    data : ndarray, shape (n, d)
        Data points to cluster.
    ml : list of (int, int)
        Must-link constraint pairs (indices into data).
    cl : list of (int, int)
        Cannot-link constraint pairs (indices into data).
    steps_mutation : int
        Maximum number of perturbation/mutation iterations.
    k : int
        Number of clusters.
    steps_back_to_best : int
        Number of non-improving iterations before reverting to best known solution.
    steps_no_improvement : int
        Number of consecutive non-improving iterations before early termination.
    verbose : bool
        Whether to print progress information.
    time_limit : float
        Wall-clock time budget in seconds.

    Returns
    -------
    membership_final : ndarray, shape (n,)
        Cluster label for each data point (0-indexed).
    """
    def KSKM_inner_exact(k, graph_coloring_problems, membership, y_values, y_values2, sub_adjs, n_supernodes, d, data, ml_map, ml_count, steps_mutation, steps_back_to_best, steps_no_improvement, verbose = True, time_limit = 3600):
        start_time = time.time()
        count = 0
        if verbose:
            print('descence')
        membership = exact_assignment_given_centroids_and_Kempe_Assignment(k, graph_coloring_problems, membership, y_values, y_values2, sub_adjs, n_supernodes, d, ml_count, verbose, time_limit=time_limit)
        # membership = KempeChainMWSP(k, membership, y_values, y_values2, sub_adjs, n_supernodes, d, ml_count, verbose = verbose)
        _, best_obj = CentroidUpdate(membership, y_values, y_values2, k, ml_count, n_supernodes, d)
        explored = [round(best_obj, 5)]
        best_membership = copy.deepcopy(membership)
        for i in range(steps_mutation):
            time_used = time.time() - start_time
            time_left = time_limit - time_used
            if time_left < 0:
                return best_membership
            if verbose:
                print('mutate')
            if i % 5 == 4:
                D = CentroidUpdate(membership, y_values, y_values2, k, ml_count, n_supernodes, d, reposition = True)
                membership, _ = KempeChainMutation_target_centroids(k, membership, sub_adjs, D)
            else:
                membership = KempeChainMutation_distr(k, membership, y_values, sub_adjs, d, data, ml_map, ml_count)

            if verbose:
                print('descence')
            membership = exact_assignment_given_centroids_and_Kempe_Assignment(k, graph_coloring_problems, membership, y_values, y_values2, sub_adjs, n_supernodes, d, ml_count, verbose,time_limit=time_left, explored= explored)

            _, obj = CentroidUpdate(membership, y_values, y_values2, k, ml_count, n_supernodes, d)
            explored += [round(obj, 5)]
            if obj < best_obj:
                best_obj = obj
                if verbose:
                    print('update best obj')
                    print(best_obj)
                count = 0
                best_membership = copy.deepcopy(membership)

            else:
                count += 1
                if count > steps_back_to_best:
                    membership = copy.deepcopy(best_membership)
                    if count > steps_no_improvement:
                        break

        return best_membership
    start_time = time.time()
    y_values, y_values2, adj, ml_map, ml_count, n_supernodes, d = preprocessing(data, cl, ml)
    sub_adjs = sub_adj_classification(adj)
    del cl, ml, adj
    C, _ = kmeans_plusplus(data, k, random_state = random_state)
    C2 = f2(C)
    D = distance_matrix(y_values, C, C2, ml_count)
    del C, C2
    graph_coloring_problems = build_graph_coloring_problems(sub_adjs, k)
    membership = DSATUR_KM(n_supernodes, sub_adjs, D, k)
    if not len(membership):
        membership = DSATUR(n_supernodes, sub_adjs, k)
        if len(membership):
            if verbose:
                print('found DSATUR solution')
            membership, _ = KempeChainMutation_target_centroids(k, membership, sub_adjs, D)
            if verbose:
                print('Initialization done')

        else:
            print('DSATUR solution infeasible')

            membership = ExactInitAssignment(n_supernodes, graph_coloring_problems, sub_adjs, D, k)
            if verbose:
                print('Initialization done')

    time_lefted = time_limit - time.time() + start_time
    membership = KSKM_inner_exact(k, graph_coloring_problems, membership, y_values, y_values2, sub_adjs, n_supernodes, d, data, ml_map, ml_count, steps_mutation, steps_back_to_best, steps_no_improvement, verbose, time_limit = time_lefted)
    membership_final = recover_ml_from_membership(membership, ml_map)
    return membership_final

def DSATUR_KM(n_supernodes, sub_adjs, D, k):
    """
    Initialize cluster assignments using the cost-aware DSATUR variant (Algorithm 5).

    Assigns singletons and cliques optimally, then applies dsatur_init_numba to
    each non-trivial connected component to assign vertices in DSATUR order to
    their nearest feasible centroid. Returns an empty list if any component
    cannot be legally colored with k colors.

    Parameters
    ----------
    n_supernodes : int
        Number of super-nodes.
    sub_adjs : dict
        Preprocessed connected-component data.
    D : ndarray, shape (n_supernodes, k)
        Distance matrix from each super-node to each centroid.
    k : int
        Number of clusters.

    Returns
    -------
    membership : ndarray, shape (n_supernodes,) or list
        Feasible initial cluster assignments, or [] if infeasible.
    """
    membership = np.zeros(n_supernodes, dtype = np.int64)
    membership = InitAssign_Singletons_and_Cliques(membership, sub_adjs, D)

    for vertices, neighbors, neighbors_idx, n in sub_adjs['others']:
        D_vertices = D[vertices]
        membership, fitted = dsatur_init_numba(membership, vertices, neighbors, neighbors_idx, D_vertices, k, n)
        if not fitted:
            return []

    return membership

def DSATUR_P(n_supernodes, sub_adjs, D, k):
    """
    Initialize cluster assignments using classic DSATUR with cost-based color mapping.

    Runs the standard (unbounded) DSATUR algorithm per component to find the
    chromatic coloring, then solves a linear assignment problem to map DSATUR
    colors to the k cluster labels minimizing total distance to centroids.
    Returns [] if any component requires more than k colors.

    Parameters
    ----------
    n_supernodes : int
        Number of super-nodes.
    sub_adjs : dict
        Preprocessed connected-component data.
    D : ndarray, shape (n_supernodes, k)
        Distance matrix from each super-node to each centroid.
    k : int
        Number of clusters.

    Returns
    -------
    membership : ndarray, shape (n_supernodes,) or list
        Feasible initial cluster assignments, or [] if infeasible.
    """
    membership = np.zeros(n_supernodes, dtype = np.int64)
    membership = InitAssign_Singletons_and_Cliques(membership, sub_adjs, D)

    for vertices, neighbors, neighbors_idx, n in sub_adjs['others']:
        D_vertices = D[vertices]
        colors = dsatur_color_numba(neighbors, neighbors_idx, n)
        K = max(colors) + 1
        if K > k:
            return []
        colors_matrix = np.zeros((K,n),dtype=np.bool_)
        for i in range(n):
            c = colors[i]
            colors_matrix[c, i] = True
        d = np.zeros((K,k))
        for i in range(K):
            idxs = colors_matrix[i]
            d[i] = D_vertices[idxs].sum(axis = 0)
        _, color_map = linear_sum_assignment(d)
        for i, c in enumerate(colors):
            v = vertices[i]
            true_c = color_map[c]
            membership[v] = true_c

    return membership

def DSATUR(n_supernodes, sub_adjs, k):
    """
    Fallback feasibility-only initialization using classic DSATUR.

    Assigns clique vertices to distinct clusters (0..size-1), then applies
    the standard DSATUR algorithm (smallest available color) to each
    non-trivial component. Used when DSATUR_KM fails. Returns [] if
    any component's chromatic number exceeds k.

    Parameters
    ----------
    n_supernodes : int
        Number of super-nodes.
    sub_adjs : dict
        Preprocessed connected-component data.
    k : int
        Number of clusters.

    Returns
    -------
    membership : ndarray, shape (n_supernodes,) or list
        Feasible initial cluster assignments, or [] if infeasible.
    """
    membership = np.zeros(n_supernodes, dtype = np.int64)
    for verts in sub_adjs['cliques']:
        for i, v in enumerate(verts):
            membership[v] = i
    for vertices, neighbors, neighbors_idx, n in sub_adjs['others']:
        # Y is (n,d), Y2 is (n,1), C is (k, d)
        colors = dsatur_color_numba(neighbors, neighbors_idx, n)
        if max(colors) >= k:
            return []

        for i, c in enumerate(colors):
            v = vertices[i]
            membership[v] = c
    return membership

def sub_adj_classification(adj):
    """
    Decompose the cannot-link super-node graph into typed connected components.

    Performs BFS to find all connected components, then classifies each as:
    - Singleton: isolated node (no cannot-link neighbors).
    - Clique: complete subgraph (every pair is cannot-linked); handled by
      linear assignment instead of Kempe swaps.
    - Other: general component; stored in CSR format for efficient Numba access.

    adj: list of lists (undirected, symmetric adjacency list), nodes 0..n-1

    Returns:
        sub_adjs: dict with keys:
            - 'cliques':    list of np.ndarray (vertex indices of each clique component)
            - 'others':     list of tuples (verts, neighbors, neighbors_idx, n_vertices)
            - 'singletons': np.ndarray of all singleton vertices
    """
    n = len(adj)
    visited = [False] * n

    sub_adjs = {'cliques': [], 'others': []}
    singletons = []

    for s in range(n):
        if visited[s]:
            continue

        # --- 1) BFS to get the connected component starting from s ---
        queue = deque([s])
        visited[s] = True
        vertices = []

        while queue:
            u = queue.popleft()
            vertices.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

        n_vertices = len(vertices)

        # --- 2) Singletons ---
        if n_vertices == 1:
            singletons.append(vertices[0])
            continue

        # --- 3) Clique check (using global degrees; safe in undirected components) ---
        # In a clique of size n_vertices, every vertex has degree (n_vertices - 1)
        is_clique = True
        for u in vertices:
            if len(adj[u]) != n_vertices - 1:
                is_clique = False
                break

        verts = np.asarray(vertices, dtype=np.int64)  # shape (m,)

        if is_clique:
            # --- 4a) Clique component ---
            sub_adjs['cliques'].append(verts)
        else:
            # --- 4b) General component: build CSR-like representation ---
            # map global -> local id
            nodes_map = {u: i for i, u in enumerate(vertices)}

            neighbors_idx = np.empty(n_vertices + 1, dtype=np.uint64)
            neighbors_idx[0] = 0
            neighbors = []
            pos = 0

            # IMPORTANT: iterate local vertices in order 0..n_vertices-1
            for local_u, u in enumerate(vertices):
                # all neighbors of u are inside this component (undirected + BFS)
                neigh_local = [nodes_map[v] for v in adj[u]]
                pos += len(neigh_local)
                neighbors.extend(neigh_local)
                neighbors_idx[local_u + 1] = pos

            neighbors = np.asarray(neighbors, dtype=np.int64)

            sub_adjs['others'].append((verts, neighbors, neighbors_idx, n_vertices))

    sub_adjs['singletons'] = np.asarray(singletons, dtype=np.int64)
    return sub_adjs

def preprocessing(data, cl, ml):
    """
    Preprocess must-link and cannot-link constraints into the super-node graph.

    Steps:
    1. Merge must-link data points into equivalence classes (super-nodes) via
       BFS on the ML graph, exploiting the transitivity of must-link constraints.
    2. Remap cannot-link constraints to operate on super-nodes.
    3. Compute per-super-node aggregated feature sums and squared-norm sums
       for efficient distance computations.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Raw data points.
    cl : list of (int, int)
        Cannot-link constraint pairs.
    ml : list of (int, int)
        Must-link constraint pairs.

    Returns
    -------
    y_values : ndarray, shape (n_supernodes, d)
        Aggregated feature sum for each super-node.
    y_values2 : ndarray, shape (n_supernodes,)
        Aggregated squared L2 norm for each super-node.
    adj : list of sets
        Cannot-link adjacency list on super-nodes (length n_supernodes).
    ml_map : ndarray, shape (n,)
        Maps each raw data point index to its super-node index.
    ml_count : ndarray, shape (n_supernodes,)
        Number of raw data points in each super-node.
    n_supernodes : int
        Number of super-nodes.
    d : int
        Feature dimension.
    """
    def construct_ml_supernodes(ml, len_data):
        adj = [[] for _ in range(len_data)]
        for u, v in ml:
            adj[u].append(v)
            adj[v].append(u)
        visited = set()
        comps = []
        comps_idxs = [0]
        pos = 0
        for s in range(len_data):
            if s in visited:
                continue
            comp = []
            q = deque([s])
            visited.add(s)
            while q:
                u = q.popleft()
                comp.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            comps.extend(comp)
            pos += len(comp)
            comps_idxs.append(pos)
        comps = np.asarray(comps)
        comps_idxs = np.asarray(comps_idxs)
        return comps, comps_idxs
    def adj_from_cl_ml(cl, ml, len_data):
        ml_list_numba, ml_list_numba_idxs = construct_ml_supernodes(ml, len_data)
        n_supernodes = ml_list_numba_idxs.size - 1
        ml_count = np.zeros(n_supernodes, dtype=int)
        ml_map = np.zeros(len_data, dtype=int)
        for i in range(n_supernodes):
            idxs_from = ml_list_numba_idxs[i]
            idxs_to = ml_list_numba_idxs[i+1]
            ml_count[i] = idxs_to - idxs_from
            for idx in range(idxs_from, idxs_to):
                v = ml_list_numba[idx]
                ml_map[v] = i
        adj =  [set() for _ in range(n_supernodes)]
        for u, v in cl:
            uu = ml_map[u]
            vv = ml_map[v]
            if uu != vv:
                if not uu in adj[vv]:
                    adj[vv].add(uu)
                if not vv in adj[uu]:
                    adj[uu].add(vv)
        return adj, ml_count, ml_list_numba, ml_list_numba_idxs ,n_supernodes, ml_map
    len_data, d = data.shape
    adj, ml_count, ml_list_numba, ml_list_numba_idxs, n_supernodes, ml_map = adj_from_cl_ml(cl, ml, len_data)
    data2 = f2(data)
    y_values, y_values2 = compute_sums_numba(data, data2, ml_list_numba, ml_list_numba_idxs, n_supernodes, d)
    return y_values, y_values2, adj, ml_map, ml_count, n_supernodes, d

def recover_ml_from_membership(membership, ml_map):
    """
    Expand super-node cluster assignments back to individual data point labels.

    Inverts the must-link preprocessing by assigning every raw data point the
    cluster label of its super-node.

    Parameters
    ----------
    membership : ndarray, shape (n_supernodes,)
        Cluster assignment for each super-node.
    ml_map : ndarray, shape (n,)
        Maps each raw data point index to its super-node index.

    Returns
    -------
    membership_final : ndarray, shape (n,)
        Cluster label for each raw data point.
    """
    n = ml_map.size
    membership_final = np.empty(n, dtype=ml_map.dtype)
    for i in range(n):
        membership_final[i] = membership[ml_map[i]]
    return membership_final
