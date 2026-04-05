#pragma once

#include <vector>

#include "svec/core/types.h"
#include "svec/pyramid/error_pyramid.h"
#include "svec/surface/mesh.h"
#include "svec/surface/triangle_plane.h"

namespace svec {

struct HierarchicalSurfaceErrorOptions {
    f64 color_weight = 1.0;
    f64 gradient_weight = 0.35;
    f64 detail_weight = 0.25;
    f64 structure_weight = 0.90;
    f64 peak_weight = 0.55;
    f64 alpha_weight = 0.20;

    u32 per_level_samples = 7;
    u32 max_levels_used = 4;
    bool clamp_alpha = true;
};

struct TriangleHierarchicalError {
    TriangleId triangle_id = kInvalidIndex;
    u64 sample_count = 0;

    f64 color_rmse = 0.0;
    f64 gradient_rmse = 0.0;
    f64 detail_rmse = 0.0;
    f64 structure_mean = 0.0;
    f64 peak_residual = 0.0;
    f64 composite_error = 0.0;
    f64 weighted_score = 0.0;
    f64 triangle_area = 0.0;
};

struct MeshHierarchicalErrorSummary {
    f64 mean_composite_error = 0.0;
    f64 max_composite_error = 0.0;
    TriangleId worst_triangle_id = kInvalidIndex;
    u64 sample_count = 0;
};

[[nodiscard]] TriangleHierarchicalError ComputeTriangleHierarchicalSurfaceError(
    const Mesh& mesh,
    TriangleId triangle_id,
    const ErrorPyramid& pyramid,
    const TrianglePlane& plane,
    const HierarchicalSurfaceErrorOptions& options = {});

[[nodiscard]] MeshHierarchicalErrorSummary ComputeMeshHierarchicalSurfaceError(
    const Mesh& mesh,
    const ErrorPyramid& pyramid,
    const std::vector<TrianglePlane>& planes,
    const HierarchicalSurfaceErrorOptions& options = {});

} // namespace svec
