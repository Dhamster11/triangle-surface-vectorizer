#pragma once

#include <vector>

#include "svec/core/types.h"
#include "svec/error/error_estimator.h"
#include "svec/image/image.h"
#include "svec/pyramid/error_pyramid.h"
#include "svec/surface/mesh.h"

namespace svec {

    enum class NewVertexColorMode {
        SampleReferenceBilinear = 0,
        MidpointOfEndpoints = 1
    };

    enum class RefineStepFailureReason {
        None = 0,
        SeedTriangleTooSmall = 1,
        SeedTriangleBBoxTooSmall = 2,
        SplitEdgeTooShort = 3,
        SplitPointUnsafe = 4,
        NeighborChildInvalid = 5,
        SplitExecutionFailed = 6
    };

    [[nodiscard]] const char* ToString(RefineStepFailureReason reason) noexcept;

    struct SingleRefineStepOptions {
        ErrorEstimatorOptions error_options{};

        bool split_shared_neighbor = true;
        bool rebuild_topology_after_split = true;

        NewVertexColorMode new_vertex_color_mode = NewVertexColorMode::SampleReferenceBilinear;

        f64 min_edge_length = 1e-6;
        f64 min_triangle_area = 1e-12;
        f64 min_triangle_bbox_extent = 0.35;
        f64 min_midpoint_separation = 1e-8;

        bool use_optimal_split_point = true;
        f64 split_search_min_t = 0.18;
        f64 split_search_max_t = 0.82;
        u32 split_search_candidate_count = 5;
        f64 split_search_center_penalty = 0.08;
        f64 split_search_residual_weight = 1.55;
        f64 split_search_cross_edge_weight = 1.30;
        f64 split_search_probe_radius_px = 1.5;

        // Tensor-guided anisotropic split metric.
        // The guidance itself is precomputed in ErrorPyramid and only sampled here.
        f64 tensor_target_span_px = 3.0;
        f64 tensor_min_coherence = 0.20;
        f64 tensor_min_strength = 0.05;
        f64 tensor_strength_softness = 0.15;

        // Small high-contrast details (eyebrows, eyelashes, thin strokes) should not
        // dominate the split metric as aggressively as large coherent boundaries.
        // The tensor influence ramps from 0 at tensor_min_triangle_scale_px to 1 at
        // tensor_full_triangle_scale_px.
        f64 tensor_min_triangle_scale_px = 4.25;
        f64 tensor_full_triangle_scale_px = 9.0;

        // Hard clamp on anisotropy strength so the metric cannot become too narrow.
        f64 tensor_max_anisotropy = 1.9;
        f64 tensor_edge_metric_weight = 1.10;
        f64 tensor_split_metric_weight = 1.05;
        f64 tensor_neighbor_consistency_weight = 0.9;

        // Penalize seams that travel through small strong coherent structures even if
        // the direction itself looks locally admissible.
        f64 tensor_seam_penalty_weight = 3.15;
        u32 tensor_seam_sample_count = 5;
    };

    struct RefineStepResult {
        bool split_performed = false;
        bool split_shared_neighbor = false;

        RefineStepFailureReason failure_reason = RefineStepFailureReason::None;

        bool topology_rebuild_performed = false;
        f64 topology_rebuild_ms = 0.0;


        TriangleId seed_triangle_id = kInvalidIndex;
        TriangleId seed_neighbor_triangle_id = kInvalidIndex;

        VertexId new_vertex_id = kInvalidIndex;

        u32 vertices_added = 0;
        u32 triangles_added = 0;

        f64 split_edge_length = 0.0;

        std::vector<TriangleId> touched_triangle_ids;

        MeshErrorSummary error_before{};
        MeshErrorSummary error_after{};
    };

    struct AdaptiveRefinementOptions {
        SingleRefineStepOptions step{};

        u32 max_iterations = 64;
        u32 max_triangle_count = 0;  // 0 = unlimited
        u32 max_vertex_count = 0;    // 0 = unlimited

        f64 target_weighted_rmse = 0.0;
        bool stop_when_no_improvement = false;
    };

    struct AdaptiveRefinementReport {
        MeshErrorSummary initial_error{};
        MeshErrorSummary final_error{};
        std::vector<RefineStepResult> steps;

        [[nodiscard]] u32 IterationsPerformed() const noexcept {
            return static_cast<u32>(steps.size());
        }
    };

    [[nodiscard]] ColorOKLaba SampleImageOKLabaNearest(const ImageOKLaba& image, const Vec2& p);
    [[nodiscard]] ColorOKLaba SampleImageOKLabaBilinear(const ImageOKLaba& image, const Vec2& p);

    [[nodiscard]] RefineStepResult RefineTriangleGeometryOnly(
        Mesh& mesh,
        TriangleId triangle_id,
        const ImageOKLaba& reference,
        const ErrorPyramid& guidance_pyramid,
        const SingleRefineStepOptions& options = {});

    [[nodiscard]] RefineStepResult RefineTriangleGeometryOnly(
        Mesh& mesh,
        TriangleId triangle_id,
        const ImageOKLaba& reference,
        const SingleRefineStepOptions& options = {});

    [[nodiscard]] RefineStepResult RefineTriangleOnce(
        Mesh& mesh,
        TriangleId triangle_id,
        const ImageOKLaba& reference,
        const ErrorPyramid& guidance_pyramid,
        const SingleRefineStepOptions& options = {});

    [[nodiscard]] RefineStepResult RefineTriangleOnce(
        Mesh& mesh,
        TriangleId triangle_id,
        const ImageOKLaba& reference,
        const SingleRefineStepOptions& options = {});

    [[nodiscard]] RefineStepResult RefineWorstTriangleOnce(
        Mesh& mesh,
        const ImageOKLaba& reference,
        const ErrorPyramid& guidance_pyramid,
        const SingleRefineStepOptions& options = {});

    [[nodiscard]] RefineStepResult RefineWorstTriangleOnce(
        Mesh& mesh,
        const ImageOKLaba& reference,
        const SingleRefineStepOptions& options = {});

    [[nodiscard]] AdaptiveRefinementReport AdaptiveRefineMesh(
        Mesh& mesh,
        const ImageOKLaba& reference,
        const AdaptiveRefinementOptions& options = {});

} // namespace svec
