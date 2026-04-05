#pragma once

#include <limits>
#include <vector>

#include "svec/core/types.h"
#include "svec/image/image.h"
#include "svec/math/color.h"
#include "svec/surface/mesh.h"

namespace svec {

enum class ErrorHeatmapMode {
    GrayscaleL = 0,
    RedTint = 1
};

struct ErrorEstimatorOptions {
    bool skip_degenerate_triangles = true;
    i32 samples_per_axis = 1;          // 1 = pixel-center sampling.
    f64 alpha_weight = 1.0;            // contribution of alpha to weighted score.
    f64 inside_epsilon = 1e-7;
    bool clamp_triangle_bbox_to_image = true;
};

struct TriangleErrorMetrics {
    TriangleId triangle_id = kInvalidIndex;
    u64 sample_count = 0;

    f64 mse_L = 0.0;
    f64 mse_a = 0.0;
    f64 mse_b = 0.0;
    f64 mse_alpha = 0.0;

    f64 rmse_lab = 0.0;
    f64 rmse_alpha = 0.0;
    f64 weighted_rmse = 0.0;

    f64 max_abs_L = 0.0;
    f64 max_abs_a = 0.0;
    f64 max_abs_b = 0.0;
    f64 max_abs_alpha = 0.0;
    f64 max_weighted_error = 0.0;

    [[nodiscard]] bool HasSamples() const noexcept { return sample_count > 0; }
};

struct MeshErrorSummary {
    u32 triangles_total = 0;
    u32 triangles_evaluated = 0;
    u32 triangles_skipped_degenerate = 0;
    u64 sample_count_total = 0;

    f64 mse_L = 0.0;
    f64 mse_a = 0.0;
    f64 mse_b = 0.0;
    f64 mse_alpha = 0.0;

    f64 rmse_lab = 0.0;
    f64 rmse_alpha = 0.0;
    f64 weighted_rmse = 0.0;

    TriangleId worst_triangle_id = kInvalidIndex;
    f64 worst_triangle_weighted_rmse = -1.0;

    [[nodiscard]] bool HasSamples() const noexcept { return sample_count_total > 0; }
};

struct MeshErrorReport {
    MeshErrorSummary summary;
    std::vector<TriangleErrorMetrics> per_triangle;
};

[[nodiscard]] TriangleErrorMetrics ComputeTriangleError(
    const Mesh& mesh,
    TriangleId triangle_id,
    const ImageOKLaba& reference,
    const ErrorEstimatorOptions& options = {});

[[nodiscard]] MeshErrorReport ComputeMeshError(
    const Mesh& mesh,
    const ImageOKLaba& reference,
    const ErrorEstimatorOptions& options = {});

[[nodiscard]] ImageOKLaba RenderTriangleErrorHeatmap(
    const Mesh& mesh,
    const ImageSize& output_size,
    const std::vector<TriangleErrorMetrics>& per_triangle,
    ErrorHeatmapMode mode = ErrorHeatmapMode::RedTint,
    f64 normalization_weighted_rmse = -1.0,
    bool skip_degenerate_triangles = true);

[[nodiscard]] ImageOKLaba RenderMeshErrorHeatmap(
    const Mesh& mesh,
    const ImageOKLaba& reference,
    const ErrorEstimatorOptions& options = {},
    ErrorHeatmapMode mode = ErrorHeatmapMode::RedTint,
    f64 normalization_weighted_rmse = -1.0);

} // namespace svec
