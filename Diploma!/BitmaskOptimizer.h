#ifndef BITMASK_OPTIMIZER_H
#define BITMASK_OPTIMIZER_H

#include "Graph.h"
#include <cstdint>
#include <omp.h>

#ifdef _MSC_VER
#include <intrin.h>
#define MASK_NOINLINE __declspec(noinline)
inline int count_active(uint64_t mask) { return (int)__popcnt64(mask); }
inline int get_lowest_bit(uint64_t mask) {
    unsigned long index;
    _BitScanForward64(&index, mask);
    return (int)index;
}
#else
#define MASK_NOINLINE __attribute__((noinline))
inline int count_active(uint64_t mask) { return __builtin_popcountll(mask); }
inline int get_lowest_bit(uint64_t mask) { return __builtin_ctzll(mask); }
#endif

// 42 хватает до Grid 6x7 включительно (при необходимости можно увеличить до 64)
#define MAX_NODES 42
constexpr int MAX_EDGES = MAX_NODES * (MAX_NODES - 1) / 2;

struct alignas(64) MaskAlignedDouble {
    double val;
    char pad[64 - sizeof(double)];
};

inline MaskAlignedDouble* get_mask_rel_arr() {
    alignas(64) static MaskAlignedDouble arr[128];
    return arr;
}

struct GraphMask {
    int num_nodes;
    uint64_t active_nodes;
    uint64_t adj[MAX_NODES];
    double prob[MAX_EDGES];

    GraphMask() = default;

    GraphMask(int n)
        : num_nodes(n),
        active_nodes((n >= 64) ? ~0ULL : ((1ULL << n) - 1)) {
        for (int i = 0; i < n; ++i) adj[i] = 0;
        for (int i = 0; i < MAX_EDGES; ++i) prob[i] = 0.0;
    }

    inline int edge_idx(int u, int v) const {
        int mx = (u > v) ? u : v;
        int mn = (u > v) ? v : u;
        return mx * (mx - 1) / 2 + mn;
    }

    inline double& p(int u, int v) {
        return prob[edge_idx(u, v)];
    }

    inline double p_const(int u, int v) const {
        return prob[edge_idx(u, v)];
    }

    inline void add_edge(int u, int v, double p_val) {
        adj[u] |= (1ULL << v);
        adj[v] |= (1ULL << u);
        p(u, v) = p_val;
    }
};

inline bool is_connected_mask(const GraphMask& g) {
    if (g.active_nodes == 0) return true;
    uint64_t visited = g.active_nodes & (~g.active_nodes + 1);
    uint64_t front = visited;

    while (front) {
        uint64_t next_front = 0;
        uint64_t temp = front;
        while (temp) {
            int u = get_lowest_bit(temp);
            next_front |= g.adj[u];
            temp &= temp - 1;
        }
        next_front &= ~visited;
        next_front &= g.active_nodes;
        visited |= next_front;
        front = next_front;
    }
    return visited == g.active_nodes;
}

inline void delete_edge_mask(GraphMask& g, int u, int v) {
    g.adj[u] &= ~(1ULL << v);
    g.adj[v] &= ~(1ULL << u);
}

inline void contract_edge_mask(GraphMask& g, int u, int v) {
    uint64_t v_neighbors = g.adj[v] & ~(1ULL << u);
    while (v_neighbors) {
        int neighbor = get_lowest_bit(v_neighbors);
        v_neighbors &= v_neighbors - 1;

        double p_v = g.p_const(v, neighbor);
        if (g.adj[u] & (1ULL << neighbor)) {
            double p_u = g.p_const(u, neighbor);
            g.p(u, neighbor) = 1.0 - (1.0 - p_u) * (1.0 - p_v);
        }
        else {
            g.adj[u] |= (1ULL << neighbor);
            g.adj[neighbor] |= (1ULL << u);
            g.p(u, neighbor) = p_v;
        }
        g.adj[neighbor] &= ~(1ULL << v);
    }
    g.adj[v] = 0;
    g.active_nodes &= ~(1ULL << v);
    g.adj[u] &= ~(1ULL << v);
}

inline bool simplify_mask(GraphMask& g, double& mult) {
    bool changed = true;
    while (changed) {
        changed = false;
        uint64_t active = g.active_nodes;

        while (active) {
            if ((g.active_nodes & (g.active_nodes - 1)) == 0) return true;

            int i = get_lowest_bit(active);
            active &= active - 1;

            int deg = count_active(g.adj[i]);
            if (deg == 0) {
                mult = 0.0;
                return false;
            }
            else if (deg == 1) {
                int neighbor = get_lowest_bit(g.adj[i]);
                mult *= g.p_const(i, neighbor);
                if (mult == 0.0) return false;

                g.adj[neighbor] &= ~(1ULL << i);
                g.adj[i] = 0;
                g.active_nodes &= ~(1ULL << i);
                changed = true;
            }
            else if (deg == 2) {
                int n1 = get_lowest_bit(g.adj[i]);
                int n2 = get_lowest_bit(g.adj[i] & ~(1ULL << n1));

                double p1 = g.p_const(i, n1);
                double p2 = g.p_const(i, n2);
                double denom = p1 + p2 - p1 * p2;
                if (denom <= 0.0) {
                    mult = 0.0;
                    return false;
                }

                double p_new = (p1 * p2) / denom;
                mult *= denom;

                g.adj[n1] &= ~(1ULL << i);
                g.adj[n2] &= ~(1ULL << i);
                g.adj[i] = 0;
                g.active_nodes &= ~(1ULL << i);

                if (g.adj[n1] & (1ULL << n2)) {
                    double p_old = g.p_const(n1, n2);
                    g.p(n1, n2) = 1.0 - (1.0 - p_old) * (1.0 - p_new);
                }
                else {
                    g.adj[n1] |= (1ULL << n2);
                    g.adj[n2] |= (1ULL << n1);
                    g.p(n1, n2) = p_new;
                }
                changed = true;
            }
        }
    }
    return true;
}

inline void choose_edge_mask(const GraphMask& g, int& out_u, int& out_v) {
    int min_deg_sum = 100000;
    bool best_in_triangle = false;
    out_u = -1;
    out_v = -1;

    uint64_t active = g.active_nodes;
    while (active) {
        int u = get_lowest_bit(active);
        active &= active - 1;

        int deg_u = count_active(g.adj[u]);
        uint64_t neighbors = g.adj[u] & active;

        while (neighbors) {
            int v = get_lowest_bit(neighbors);
            neighbors &= neighbors - 1;

            int deg_sum = deg_u + count_active(g.adj[v]);
            bool in_triangle = (g.adj[u] & g.adj[v]) != 0;

            if (deg_sum == 6 && in_triangle) {
                out_u = u;
                out_v = v;
                return;
            }
            if (deg_sum < min_deg_sum || (deg_sum == min_deg_sum && in_triangle && !best_in_triangle)) {
                min_deg_sum = deg_sum;
                best_in_triangle = in_triangle;
                out_u = u;
                out_v = v;
            }
        }
    }
}

inline double base_probabilities_mask(const GraphMask& g, int n_active) {
    if (n_active <= 1) return 1.0;

    int v[5];
    uint64_t mask = g.active_nodes;
    for (int i = 0; i < n_active; ++i) {
        v[i] = get_lowest_bit(mask);
        mask &= mask - 1;
    }

    auto get_q = [&](int i, int j) -> double {
        return (g.adj[v[i]] & (1ULL << v[j])) ? (1.0 - g.p_const(v[i], v[j])) : 1.0;
        };

    if (n_active == 2) {
        return (g.adj[v[0]] & (1ULL << v[1])) ? g.p_const(v[0], v[1]) : 0.0;
    }

    if (n_active == 3) {
        double p1 = (g.adj[v[0]] & (1ULL << v[1])) ? g.p_const(v[0], v[1]) : 0.0;
        double p2 = (g.adj[v[1]] & (1ULL << v[2])) ? g.p_const(v[1], v[2]) : 0.0;
        double p3 = (g.adj[v[0]] & (1ULL << v[2])) ? g.p_const(v[0], v[2]) : 0.0;
        return p1 * p2 + p1 * p3 + p2 * p3 - 2.0 * p1 * p2 * p3;
    }

    if (n_active == 4) {
        double a = get_q(0, 1);
        double f = get_q(0, 2);
        double d = get_q(0, 3);
        double b = get_q(1, 2);
        double e = get_q(1, 3);
        double c = get_q(2, 3);

        return 1.0 - (6.0 * a * b * c * d * e * f
            - 2.0 * (b * d * e * f * (a + c - 0.5)
                + a * c * e * f * (b + d - 0.5)
                + a * b * c * d * (e + f - 0.5))
            + a * b * e + a * d * f + b * c * f + c * d * e);
    }

    double a = get_q(0, 1);
    double b = get_q(0, 2);
    double c = get_q(0, 3);
    double d = get_q(0, 4);
    double e = get_q(1, 2);
    double f = get_q(1, 3);
    double q = get_q(1, 4);
    double h = get_q(2, 3);
    double k = get_q(2, 4);
    double l = get_q(3, 4);

    double k1 = 1.0 - e * (f * q + h * k);
    double k2 = 1.0 - h * (b * k + c * l);
    double k3 = 1.0 - l * (c * f + d * q);
    double k4 = 1.0 - d * (a * b + q * k);
    double k5 = 1.0 - a * (b * c + e * f);
    double k6 = (1.0 - a) * (1.0 - h) * (1.0 - k) + (1.0 - a) * (1.0 - l) * ((1.0 - h) * k + h * (1.0 - k)) + a * h * k * (1.0 - 4.0 * l);
    double k7 = (1.0 - c) * (1.0 - d) * (1.0 - e) + (1.0 - e) * (1.0 - l) * ((1.0 - c) * d + c * (1.0 - d)) + d * e * l;
    double k8 = (1.0 - a) * (1.0 - d) * (1.0 - h) + (1.0 - q) * (1.0 - h) * ((1.0 - a) * d + a * (1.0 - d)) + a * h * q;
    double k9 = (1.0 - a) * (1.0 - b) * (1.0 - l) + (1.0 - e) * (1.0 - l) * ((1.0 - a) * b + a * (1.0 - b)) + a * e * l;
    double k10 = (1.0 - e) * (1.0 - d) * (1.0 - f) + (1.0 - d) * (1.0 - h) * ((1.0 - e) * f + e * (1.0 - f)) + d * f * h;
    double k11 = (1.0 - b) * (1.0 - f) * (1.0 - q) + (1.0 - b) * (1.0 - l) * ((1.0 - f) * q + f * (1.0 - q)) + b * q * l;
    double k12 = (1.0 - c) * (1.0 - e) * (1.0 - q) + (1.0 - c) * (1.0 - k) * ((1.0 - e) * q + e * (1.0 - q)) + c * e * k;
    double k13 = (1.0 - b) * (1.0 - d) * (1.0 - f) + (1.0 - f) * (1.0 - k) * ((1.0 - b) * d + b * (1.0 - d)) + b * d * f;
    double k14 = (1.0 - b) * (1.0 - c) * (1.0 - q) + (1.0 - q) * (1.0 - h) * ((1.0 - b) * c + b * (1.0 - c)) + b * c * q;
    double k15 = (1.0 - a) * (1.0 - c) * (1.0 - k) + (1.0 - f) * (1.0 - k) * ((1.0 - a) * c + a * (1.0 - c)) + c * f * k;

    return 1.0 - (b * c * (a * d * k1 + f * e * (d * q * k6 + k * l * k8))
        + f * q * (a * e * k2 + h * k * (a * b * k7 + c * d * k9))
        + b * h * (e * k * k3 + d * l * (a * f * k12 + e * q * k15))
        + c * l * (f * h * k4 + a * q * (b * k * k10 + e * h * k13))
        + d * k * (q * l * k5 + a * e * (c * h * k11 + f * l * k14)));
}

// ---------- ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ ----------

inline double factoring_mask_rec(GraphMask& g, bool assumed_connected);

inline double factoring_mask_branch_serial(GraphMask& g) {
    int u, v;
    choose_edge_mask(g, u, v);
    if (u == -1 || v == -1) return 0.0;

    double p = g.p_const(u, v);

    GraphMask g_contract = g;
    contract_edge_mask(g_contract, u, v);
    double rel_contract = factoring_mask_rec(g_contract, true);

    delete_edge_mask(g, u, v);
    double rel_delete = factoring_mask_rec(g, false);

    return p * rel_contract + (1.0 - p) * rel_delete;
}

inline double factoring_mask_rec(GraphMask& g, bool assumed_connected) {
    if (count_active(g.active_nodes) <= 1) return 1.0;
    if (!assumed_connected && !is_connected_mask(g)) return 0.0;

    double mult = 1.0;
    if (!simplify_mask(g, mult)) return 0.0;

    int n_active = count_active(g.active_nodes);
    if (n_active <= 5) {
        return mult * base_probabilities_mask(g, n_active);
    }

    return mult * factoring_mask_branch_serial(g);
}

inline double factoring_mask(const GraphMask& g_in) {
    GraphMask g = g_in;
    return factoring_mask_rec(g, false);
}

// ---------- ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ (OpenMP no-taskwait + AlignedDouble) ----------

MASK_NOINLINE inline void factoring_mask_no_reverse_task(
    GraphMask* g_ptr,
    bool assumed_connected,
    double current_prob,
    int cutoff
) {
    if (current_prob <= 0.0) return;

    MaskAlignedDouble* rel_arr = get_mask_rel_arr();

    if (count_active(g_ptr->active_nodes) <= 1) {
        rel_arr[omp_get_thread_num()].val += current_prob;
        return;
    }

    if (!assumed_connected && !is_connected_mask(*g_ptr)) return;

    double mult = 1.0;
    if (!simplify_mask(*g_ptr, mult)) return;

    double acc_prob = current_prob * mult;
    int n_active = count_active(g_ptr->active_nodes);

    if (n_active <= 5) {
        rel_arr[omp_get_thread_num()].val += acc_prob * base_probabilities_mask(*g_ptr, n_active);
        return;
    }

    // Ниже cutoff досчитываем поддерево последовательно на стеке без создания задач
    if (n_active <= cutoff) {
        rel_arr[omp_get_thread_num()].val += acc_prob * factoring_mask_branch_serial(*g_ptr);
        return;
    }

    int u, v;
    choose_edge_mask(*g_ptr, u, v);
    if (u == -1 || v == -1) return;

    double p = g_ptr->p_const(u, v);

    // Ветка стягивания (G + e) -> в задачу OpenMP (в firstprivate только указатель и скаляры)
    double p_contract = acc_prob * p;
    if (p_contract > 0.0) {
        GraphMask* g_contract = new GraphMask(*g_ptr);
        contract_edge_mask(*g_contract, u, v);
#pragma omp task firstprivate(g_contract, p_contract, cutoff)
        {
            factoring_mask_no_reverse_task(g_contract, true, p_contract, cutoff);
            delete g_contract;
        }
    }

    // Ветка удаления (G - e) -> на месте в текущем потоке без копирования
    double p_delete = acc_prob * (1.0 - p);
    if (p_delete > 0.0) {
        delete_edge_mask(*g_ptr, u, v);
        factoring_mask_no_reverse_task(g_ptr, false, p_delete, cutoff);
    }
}

MASK_NOINLINE inline double factoring_mask_parallel(const GraphMask& g_in, int cutoff = 10) {
    MaskAlignedDouble* rel_arr = get_mask_rel_arr();
    for (int i = 0; i < 128; ++i) {
        rel_arr[i].val = 0.0;
    }

    GraphMask root_g = g_in;

#pragma omp parallel
    {
#pragma omp single
        {
            factoring_mask_no_reverse_task(&root_g, false, 1.0, cutoff);
        }
    }

    double total_rel = 0.0;
    int max_th = omp_get_max_threads();
    if (max_th > 128) max_th = 128;
    for (int i = 0; i < max_th; ++i) {
        total_rel += rel_arr[i].val;
    }
    return total_rel;
}

inline GraphMask convertToMask(math::Graph* g) {
    int n = g->getVertNumb();
    if (n > MAX_NODES) n = MAX_NODES;

    GraphMask mask(n);
    u_int* KAO = g->getKAO();
    u_int* FO = g->getFO();

    for (int i = 1; i <= n; ++i) {
        for (u_int j = KAO[i - 1]; j < KAO[i]; ++j) {
            int u = i - 1;
            int v = FO[j] - 1;
            if (v < MAX_NODES) mask.add_edge(u, v, 0.75);
        }
    }
    return mask;
}

#endif