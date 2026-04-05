#pragma once

#include <vector>

#include "svec/core/types.h"
#include "svec/edge/edge_map.h"
#include "svec/error/error_estimator.h"
#include "svec/refine/adaptive_refinement.h"
#include "svec/surface/mesh.h"

namespace svec {

struct EdgeTriangleMetrics {
    TriangleId triangle_id = kInvalidIndex;
    u64 sample_count = 0;

    f64 mean_edge_strength = 0.0;
    f64 max_edge_strength = 0.0;

    [[nodiscard]] bool HasSamples() const noexcept {
        return sample_count > 0;
    }
};

struct EdgeAwareTriangleScore {
    TriangleId triangle_id = kInvalidIndex;
    f64 weighted_rmse = 0.0;
    f64 mean_edge_strength = 0.0;
    f64 max_edge_strength = 0.0;
    f64 score = 0.0;
};

struct EdgeAwareSelectionReport {
    MeshErrorSummary mesh_error{};
    std::vector<EdgeTriangleMetrics> edge_metrics;
    std::vector<EdgeAwareTriangleScore> per_triangle_score;
    TriangleId selected_triangle_id = kInvalidIndex;
    f64 selected_score = 0.0;
};

enum class EdgeAwareSelectionMode {
    FullRescan = 0,
    CachedHeap = 1
};

struct EdgeAwareSelectionOptions {
    ErrorEstimatorOptions error_options{};
    i32 edge_samples_per_axis = 2;

    f64 edge_weight = 2.5;
    f64 edge_power = 1.0;
    f64 strong_edge_threshold = 0.35;
    f64 strong_edge_bonus = 0.35;
    f64 mean_edge_bonus = 0.15;

    bool skip_degenerate_triangles = true;
    f64 inside_epsilon = 1e-7;

    EdgeAwareSelectionMode mode = EdgeAwareSelectionMode::CachedHeap;
    bool keep_full_debug_snapshots = false;
};

struct EdgeAwareRefinementOptions {
    SingleRefineStepOptions step{};
    EdgeMapOptions edge_map{};
    EdgeAwareSelectionOptions select{};

    u32 bootstrap_iterations = 0;
    u32 max_iterations = 64; // 0 = until target/budget.
    u32 max_triangle_count = 0;
    u32 max_vertex_count = 0;

    f64 target_weighted_rmse = 0.0;
    bool stop_when_no_improvement = false;
    bool precompute_edge_map_once = true;
};

struct EdgeAwareRefinementReport {
    MeshErrorSummary initial_error{};
    MeshErrorSummary final_error{};

    EdgeMap edge_map;
    std::vector<EdgeAwareSelectionReport> selections;
    std::vector<RefineStepResult> steps;

    u64 stale_heap_pops = 0;
    u64 blocked_split_attempts = 0;

    [[nodiscard]] u32 IterationsPerformed() const noexcept {
        return static_cast<u32>(steps.size());
    }
};

[[nodiscard]] EdgeTriangleMetrics ComputeTriangleEdgeMetrics(
    const Mesh& mesh,
    TriangleId triangle_id,
    const EdgeMap& edge_map,
    const EdgeAwareSelectionOptions& options = {});

[[nodiscard]] EdgeAwareSelectionReport SelectTriangleForEdgeAwareRefinement(
    const Mesh& mesh,
    const ImageOKLaba& reference,
    const EdgeMap& edge_map,
    const EdgeAwareSelectionOptions& options = {});

[[nodiscard]] EdgeAwareRefinementReport EdgeAwareRefineMesh(
    Mesh& mesh,
    const ImageOKLaba& reference,
    const EdgeAwareRefinementOptions& options = {});

} // namespace svec
