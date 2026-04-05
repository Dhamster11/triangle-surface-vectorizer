#pragma once

#include <vector>

#include "svec/core/types.h"
#include "svec/edge/edge_map.h"
#include "svec/error/hierarchical_surface_error.h"
#include "svec/pyramid/error_pyramid.h"
#include "svec/refine/adaptive_refinement.h"
#include "svec/surface/triangle_plane.h"

namespace svec {

    enum class PyramidRefinementStopReason {
        None = 0,
        MaxSplitsReached = 1,
        MaxTrianglesReached = 2,
        TargetMeanErrorReached = 3,
        MinErrorToSplitReached = 4,
        HeapExhausted = 5,
        NoSplitsPerformedInBatch = 6,
        ProgressStalled = 7
    };

    [[nodiscard]] const char* ToString(PyramidRefinementStopReason reason) noexcept;

    struct PyramidEdgeBiasOptions {
        bool enabled = true;

        // Multiplicative priority boost applied only to heap ordering.
        // The underlying composite error stays unchanged.
        f64 mean_weight = 0.75;
        f64 peak_weight = 2.90;
        f64 power = 1.45;

        f64 strong_edge_threshold = 0.11;
        f64 strong_edge_bonus = 3.10;
        f64 max_multiplier = 6.00;
    };

    struct PyramidRefinementSafetyOptions {
        bool enabled = true;

        // Region-based anti-hotspot governor.
        u32 hotspot_cell_px = 6;
        u32 max_consecutive_region_hits = 40;
        u32 region_cooldown_splits = 64;

        // Triangle-local guards.
        u32 max_triangle_depth = 128;
        u32 max_failed_attempts_per_triangle = 5;
        u32 max_low_gain_events_per_triangle = 5;

        // A split is considered low-gain if the area-weighted local error in the
        // refreshed neighborhood does not decrease by at least this ratio.
        f64 min_local_error_drop_ratio = 0.000003;

        // Global watchdog: stop if many batches pass without improving the best
        // mean composite error by at least this amount.
        bool stop_on_progress_stall = true;
        u32 stall_batch_window = 2048;
        f64 min_batch_mean_error_drop = 2e-7;

        // Soft penalties applied before hard suppression.
        f64 depth_penalty = 0.008;
        f64 region_repeat_penalty = 0.003;
        f64 low_gain_penalty = 0.005;
        f64 min_effective_score = 1e-9;
    };

    struct PyramidRefinementTelemetry {
        f64 time_build_pyramid_ms = 0.0;
        f64 time_initial_topology_ms = 0.0;
        f64 time_recolor_vertices_ms = 0.0;
        f64 time_initial_cache_ms = 0.0;
        f64 time_split_geometry_ms = 0.0;
        f64 time_topology_rebuild_ms = 0.0;
        f64 time_refresh_ms = 0.0;
        f64 time_plane_fit_ms = 0.0;
        f64 time_hier_error_ms = 0.0;
        f64 time_edge_bias_ms = 0.0;
        f64 time_final_planes_ms = 0.0;
        f64 time_final_error_ms = 0.0;

        u64 refresh_calls_total = 0;
        u64 refresh_unique_triangles = 0;
        u64 refresh_neighbor_calls = 0;
        u64 topology_rebuild_count = 0;
        u64 initial_cache_triangle_count = 0;
        u64 edge_proxy_evaluations = 0;

        u64 heap_pushes_total = 0;
        u64 heap_pops_total = 0;
        u64 heap_valid_pops = 0;
        u64 heap_rebuild_count = 0;
        u64 heap_entries_discarded_by_rebuild = 0;
        u64 heap_max_size = 0;

        u64 split_rejected_seed_too_small = 0;
        u64 split_rejected_bbox_too_small = 0;
        u64 split_rejected_edge_too_short = 0;
        u64 split_rejected_split_point_unsafe = 0;
        u64 split_rejected_neighbor_child_invalid = 0;
        u64 split_rejected_split_execution_failed = 0;

        u64 safety_suppressed_by_depth = 0;
        u64 safety_suppressed_by_cooldown = 0;
        u64 safety_suppressed_by_failed_attempts = 0;
        u64 safety_low_gain_events = 0;
        u64 safety_region_cooldowns = 0;
        u64 safety_progress_stall_batches = 0;
    };

    struct PyramidRefinementOptions {
        ErrorPyramidOptions pyramid{};
        TrianglePlaneFitOptions plane_fit{};
        HierarchicalSurfaceErrorOptions error{};
        PyramidEdgeBiasOptions edge_bias{};
        PyramidRefinementSafetyOptions safety{};
        SingleRefineStepOptions split{};

        u32 bootstrap_splits = 128;
        u32 batch_size = 16;
        u32 max_splits = 12000;
        u32 max_triangles = 0;
        f64 target_mean_error = 0.00002;
        f64 min_error_to_split = 0.000002;
        bool stop_when_heap_exhausted = true;
    };

    struct PyramidTriangleCacheEntry {
        u32 version = 0;
        bool valid = false;
        TrianglePlane plane{};
        TriangleHierarchicalError error{};

        f64 edge_mean_strength = 0.0;
        f64 edge_peak_strength = 0.0;
        f64 raw_selection_score = 0.0;
        f64 selection_score = 0.0;
    };

    struct PyramidRefinementReport {
        ErrorPyramid pyramid;
        PyramidRefinementStopReason stop_reason = PyramidRefinementStopReason::None;
        PyramidRefinementTelemetry telemetry{};

        MeshHierarchicalErrorSummary initial_error{};
        MeshHierarchicalErrorSummary final_error{};
        std::vector<TrianglePlane> final_planes;

        u32 splits_requested = 0;
        u32 splits_performed = 0;
        u32 batches_performed = 0;
        u32 blocked_split_attempts = 0;
        u64 stale_heap_pops = 0;
    };

    [[nodiscard]] PyramidRefinementReport PyramidRefineMesh(
        Mesh& mesh,
        const ImageOKLaba& reference,
        const PyramidRefinementOptions& options = {});

    [[nodiscard]] PyramidRefinementReport PyramidRefineMesh(
        Mesh& mesh,
        const ImageOKLaba& reference,
        const EdgeMap& edge_map,
        const PyramidRefinementOptions& options);

} // namespace svec
