#pragma once

#include <vector>

#include "svec/core/types.h"
#include "svec/edge/edge_map.h"
#include "svec/error/error_estimator.h"
#include "svec/image/image.h"
#include "svec/surface/mesh.h"

namespace svec {

struct EdgeFlipOptions {
    bool enabled = true;
    bool preserve_strong_edges = true;
    f64 strong_edge_threshold = 0.35;

    f64 min_quality_improvement_degrees = 0.10;
    f64 max_local_error_increase_ratio = 0.02;
    bool rebuild_topology_after_pass = true;
};

struct VertexSmoothingOptions {
    bool enabled = true;
    u32 iterations = 1;
    f64 lambda = 0.35;
    f64 max_move_distance = 4.0;

    bool skip_boundary_vertices = true;
    bool preserve_strong_edges = true;
    f64 strong_edge_threshold = 0.25;

    bool update_vertex_color_from_reference = true;
    f64 max_local_error_increase_ratio = 0.03;
    f64 max_local_min_angle_drop_degrees = 0.25;
};

struct MeshOptimizationOptions {
    ErrorEstimatorOptions error_options{};
    EdgeFlipOptions flip{};
    VertexSmoothingOptions smooth{};

    u32 outer_iterations = 1;
    bool stop_when_no_changes = true;
};

struct EdgeFlipPassReport {
    u32 candidate_edges = 0;
    u32 interior_edges = 0;
    u32 flips_applied = 0;
    u32 skipped_strong_edge = 0;
    u32 skipped_quality = 0;
    u32 skipped_error = 0;
    u32 skipped_invalid = 0;
};

struct VertexSmoothingPassReport {
    u32 vertices_considered = 0;
    u32 vertices_moved = 0;
    u32 skipped_boundary = 0;
    u32 skipped_strong_edge = 0;
    u32 skipped_geometry = 0;
    u32 skipped_error = 0;
};

struct MeshOptimizationIterationReport {
    EdgeFlipPassReport flip{};
    std::vector<VertexSmoothingPassReport> smoothing_passes;
};

struct MeshOptimizationReport {
    MeshErrorSummary initial_error{};
    MeshErrorSummary final_error{};

    f64 initial_min_triangle_angle_deg = 0.0;
    f64 final_min_triangle_angle_deg = 0.0;
    f64 initial_mean_min_angle_deg = 0.0;
    f64 final_mean_min_angle_deg = 0.0;

    std::vector<MeshOptimizationIterationReport> iterations;

    [[nodiscard]] u32 IterationsPerformed() const noexcept {
        return static_cast<u32>(iterations.size());
    }

    [[nodiscard]] u32 TotalFlipsApplied() const noexcept {
        u32 out = 0;
        for (const auto& it : iterations) {
            out += it.flip.flips_applied;
        }
        return out;
    }

    [[nodiscard]] u32 TotalVertexMovesApplied() const noexcept {
        u32 out = 0;
        for (const auto& it : iterations) {
            for (const auto& pass : it.smoothing_passes) {
                out += pass.vertices_moved;
            }
        }
        return out;
    }
};

[[nodiscard]] std::vector<std::vector<TriangleId>> BuildVertexIncidentTriangles(const Mesh& mesh);
[[nodiscard]] std::vector<std::vector<VertexId>> BuildVertexOneRingAdjacency(const Mesh& mesh);
[[nodiscard]] std::vector<bool> BuildBoundaryVertexMask(const Mesh& mesh);

[[nodiscard]] f64 ComputeTriangleMinAngleDegrees(const Mesh& mesh, TriangleId triangle_id);
[[nodiscard]] f64 ComputeMeshMinTriangleAngleDegrees(const Mesh& mesh, bool skip_degenerate = true);
[[nodiscard]] f64 ComputeMeshMeanTriangleMinAngleDegrees(const Mesh& mesh, bool skip_degenerate = true);

[[nodiscard]] EdgeFlipPassReport FlipImprovingInteriorEdges(
    Mesh& mesh,
    const ImageOKLaba& reference,
    const ErrorEstimatorOptions& error_options = {},
    const EdgeFlipOptions& options = {},
    const EdgeMap* edge_map = nullptr);

[[nodiscard]] VertexSmoothingPassReport SmoothMeshVerticesOnce(
    Mesh& mesh,
    const ImageOKLaba& reference,
    const ErrorEstimatorOptions& error_options = {},
    const VertexSmoothingOptions& options = {},
    const EdgeMap* edge_map = nullptr);

[[nodiscard]] MeshOptimizationReport OptimizeMesh(
    Mesh& mesh,
    const ImageOKLaba& reference,
    const MeshOptimizationOptions& options = {},
    const EdgeMap* edge_map = nullptr);

} // namespace svec
