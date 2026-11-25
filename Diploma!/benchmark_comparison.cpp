#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <omp.h>

#include "Graph.h"
#include "GraphUtil.h"
#include "BitmaskOptimizer.h"

int main() {
    omp_set_dynamic(0);
    const int num_threads = 12;
    const int task_cutoff = 10;

    // Прогрев пула потоков OpenMP
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
        volatile int tid = omp_get_thread_num();
        (void)tid;
    }

    std::cout << "====================================================================\n";
    std::cout << "    CSR (KAO-FO) vs Bitmask (Serial & Parallel OpenMP " << num_threads << " threads)\n";
    std::cout << "====================================================================\n\n";

    struct TestConfig {
        std::string name;
        math::Graph* graph;
        bool run_serial;
    };

    std::vector<TestConfig> tests;
    tests.push_back({ "Grid 3x3", math::GraphGenerator::generateGrid(2, 2), true });
    tests.push_back({ "Grid 4x4", math::GraphGenerator::generateGrid(3, 3), true });
    tests.push_back({ "Grid 5x5", math::GraphGenerator::generateGrid(4, 4), true });
    tests.push_back({ "Grid 5x6", math::GraphGenerator::generateGrid(4, 5), true });
    tests.push_back({ "Grid 6x6", math::GraphGenerator::generateGrid(5, 5), true });
    tests.push_back({ "Complete K13", math::GraphGenerator::generateCompleteGraph(13), false });
    tests.push_back({ "Grid 6x7", math::GraphGenerator::generateGrid(5, 6), false });

    std::cout << std::fixed << std::setprecision(6);

    for (auto& test : tests) {
        std::cout << "--- Test: " << test.name << " (" << test.graph->getVertNumb()
            << " nodes, " << test.graph->getEdgNumb() << " edges) ---\n";

        GraphMask mask = convertToMask(test.graph);

        double time_csr_ser = 0.0, rel_csr_ser = 0.0;
        double time_mask_ser = 0.0, rel_mask_ser = 0.0;

        if (test.run_serial) {
            auto t1 = std::chrono::steady_clock::now();
            rel_csr_ser = math::GraphFactory::factoring(
                test.graph, math::FactoringType::NO_REVERSE_REC_FACTORING, math::FactoringExecutionType::SERIAL);
            auto t2 = std::chrono::steady_clock::now();
            time_csr_ser = std::chrono::duration<double>(t2 - t1).count();

            auto t3 = std::chrono::steady_clock::now();
            rel_mask_ser = factoring_mask(mask);
            auto t4 = std::chrono::steady_clock::now();
            time_mask_ser = std::chrono::duration<double>(t4 - t3).count();

            std::cout << "  [SERIAL]   CSR      | Time: " << std::setw(10) << time_csr_ser
                << " s | Rel: " << rel_csr_ser << "\n";
            std::cout << "  [SERIAL]   Bitmask  | Time: " << std::setw(10) << time_mask_ser
                << " s | Rel: " << rel_mask_ser << "\n";
            if (time_mask_ser > 0.0) {
                std::cout << "  => Serial speedup (Bitmask vs CSR): " << std::setprecision(2)
                    << (time_csr_ser / time_mask_ser) << "x\n" << std::setprecision(6);
            }
        }

        // Параллельные прогоны (no-taskwait)
        auto p1 = std::chrono::steady_clock::now();
        double rel_csr_par = math::GraphFactory::factoring(
            test.graph, math::FactoringType::NO_REVERSE_REC_FACTORING, math::FactoringExecutionType::PARALLEL);
        auto p2 = std::chrono::steady_clock::now();
        double time_csr_par = std::chrono::duration<double>(p2 - p1).count();

        auto p3 = std::chrono::steady_clock::now();
        double rel_mask_par = factoring_mask_parallel(mask, task_cutoff);
        auto p4 = std::chrono::steady_clock::now();
        double time_mask_par = std::chrono::duration<double>(p4 - p3).count();

        std::cout << "  [PARALLEL] CSR      | Time: " << std::setw(10) << time_csr_par
            << " s | Rel: " << rel_csr_par << "\n";
        std::cout << "  [PARALLEL] Bitmask  | Time: " << std::setw(10) << time_mask_par
            << " s | Rel: " << rel_mask_par << "\n";

        if (time_mask_par > 0.0) {
            std::cout << std::setprecision(2);
            std::cout << "  => Parallel speedup (Bitmask vs CSR): " << (time_csr_par / time_mask_par) << "x\n";
            if (test.run_serial) {
                std::cout << "  => Bitmask scaling (1 -> " << num_threads << " threads): "
                    << (time_mask_ser / time_mask_par) << "x\n";
            }
            std::cout << std::setprecision(6);
        }

        if (std::fabs(rel_csr_par - rel_mask_par) > 1e-9) {
            std::cout << "  [WARNING] Reliability mismatch! diff = "
                << std::fabs(rel_csr_par - rel_mask_par) << "\n";
        }
        std::cout << "\n";

        delete test.graph;
    }

    // --- Тест масштабируемости по потокам на Grid 6x7 (Bitmask vs CSR) ---
    std::cout << "====================================================================\n";
    std::cout << "    SCALING TEST ON GRID 6x7 (Bitmask vs CSR, no-taskwait)\n";
    std::cout << "====================================================================\n";
    math::Graph* g67 = math::GraphGenerator::generateGrid(5, 6);
    GraphMask mask67 = convertToMask(g67);

    std::vector<int> thread_counts = { 1, 2, 4, 6, 8, 12 };
    double base_mask_time = 0.0;
    double base_csr_time = 0.0;

    for (int th : thread_counts) {
        omp_set_num_threads(th);

        auto s1 = std::chrono::steady_clock::now();
        double r_mask = (th == 1) ? factoring_mask(mask67) : factoring_mask_parallel(mask67, task_cutoff);
        auto e1 = std::chrono::steady_clock::now();
        double t_mask = std::chrono::duration<double>(e1 - s1).count();
        if (th == 1) base_mask_time = t_mask;

        auto s2 = std::chrono::steady_clock::now();
        double r_csr = math::GraphFactory::factoring(
            g67, math::FactoringType::NO_REVERSE_REC_FACTORING, math::FactoringExecutionType::PARALLEL);
        auto e2 = std::chrono::steady_clock::now();
        double t_csr = std::chrono::duration<double>(e2 - s2).count();
        if (th == 1) base_csr_time = t_csr;

        std::cout << "Threads: " << std::setw(2) << th
            << " | Bitmask: " << std::fixed << std::setprecision(3) << std::setw(7) << t_mask << " s ("
            << std::setprecision(2) << (base_mask_time / t_mask) << "x)"
            << " | CSR: " << std::setprecision(3) << std::setw(7) << t_csr << " s ("
            << std::setprecision(2) << (base_csr_time / t_csr) << "x)"
            << " | Bitmask faster: " << (t_csr / t_mask) << "x"
            << " | Rel diff: " << std::scientific << std::setprecision(1) << std::fabs(r_mask - r_csr) << "\n";
    }

    delete g67;
    return 0;
}