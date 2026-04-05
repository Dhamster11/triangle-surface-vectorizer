#include "svec/error/error_estimator.h"

#include <cmath>
#include <stdexcept>

#include "svec/math/geometry.h"
#include "svec/math/scalar.h"

namespace svec {
namespace {

struct SquaredErrorAccum {
    f64 sum_L2 = 0.0;
    f64 sum_a2 = 0.0;
    f64 sum_b2 = 0.0;
    f64 sum_alpha2 = 0.0;
    u64 sample_count = 0;

    f64 max_abs_L = 0.0;
    f64 max_abs_a = 0.0;
    f64 max_abs_b = 0.0;
    f64 max_abs_alpha = 0.0;
    f64 max_weighted_error = 0.0;
};

[[nodiscard]] ColorOKLaba InterpolateTriangleColor(
    const Mesh& mesh,
    const Triangle& tri,
    const Barycentric& bc) noexcept {

    const ColorOKLaba& c0 = TriangleC0(mesh, tri);
    const ColorOKLaba& c1 = TriangleC1(mesh, tri);
    const ColorOKLaba& c2 = TriangleC2(mesh, tri);

    return {
        c0.L * bc.u + c1.L * bc.v + c2.L * bc.w,
        c0.a * bc.u + c1.a * bc.v + c2.a * bc.w,
        c0.b * bc.u + c1.b * bc.v + c2.b * bc.w,
        c0.alpha * bc.u + c1.alpha * bc.v + c2.alpha * bc.w,
    };
}

[[nodiscard]] ColorOKLaba MakeHeatColor(f64 t, ErrorHeatmapMode mode) noexcept {
    t = Saturate(t);
    switch (mode) {
        case ErrorHeatmapMode::GrayscaleL:
            return {t, 0.0, 0.0, 1.0};
        case ErrorHeatmapMode::RedTint:
        default:
            return {0.25 + 0.55 * t, 0.02 + 0.24 * t, 0.02 + 0.08 * t, 1.0};
    }
}

[[nodiscard]] f64 DefaultNormalization(const std::vector<TriangleErrorMetrics>& per_triangle) noexcept {
    f64 max_v = 0.0;
    for (const auto& tri : per_triangle) {
        max_v = Max(max_v, tri.weighted_rmse);
    }
    return max_v > 0.0 ? max_v : 1.0;
}

[[nodiscard]] TriangleErrorMetrics FinalizeTriangleMetrics(
    TriangleId triangle_id,
    const SquaredErrorAccum& acc,
    f64 alpha_weight) noexcept {

    TriangleErrorMetrics out{};
    out.triangle_id = triangle_id;
    out.sample_count = acc.sample_count;
    out.max_abs_L = acc.max_abs_L;
    out.max_abs_a = acc.max_abs_a;
    out.max_abs_b = acc.max_abs_b;
    out.max_abs_alpha = acc.max_abs_alpha;
    out.max_weighted_error = acc.max_weighted_error;

    if (acc.sample_count == 0) {
        return out;
    }

    const f64 inv_n = 1.0 / static_cast<f64>(acc.sample_count);
    out.mse_L = acc.sum_L2 * inv_n;
    out.mse_a = acc.sum_a2 * inv_n;
    out.mse_b = acc.sum_b2 * inv_n;
    out.mse_alpha = acc.sum_alpha2 * inv_n;

    out.rmse_lab = std::sqrt(out.mse_L + out.mse_a + out.mse_b);
    out.rmse_alpha = std::sqrt(out.mse_alpha);
    out.weighted_rmse = std::sqrt(out.mse_L + out.mse_a + out.mse_b + alpha_weight * out.mse_alpha);
    return out;
}

[[nodiscard]] Vec2 SamplePointInsidePixel(i32 x, i32 y, i32 sx, i32 sy, i32 samples_per_axis) noexcept {
    if (samples_per_axis <= 1) {
        return {static_cast<f64>(x), static_cast<f64>(y)};
    }

    const f64 inv = 1.0 / static_cast<f64>(samples_per_axis);
    const f64 ox = (static_cast<f64>(sx) + 0.5) * inv - 0.5;
    const f64 oy = (static_cast<f64>(sy) + 0.5) * inv - 0.5;
    return {static_cast<f64>(x) + ox, static_cast<f64>(y) + oy};
}

void AccumulateTriangleError(
    SquaredErrorAccum& acc,
    const ColorOKLaba& predicted,
    const ColorOKLaba& reference,
    f64 alpha_weight) noexcept {

    const f64 dL = predicted.L - reference.L;
    const f64 da = predicted.a - reference.a;
    const f64 db = predicted.b - reference.b;
    const f64 dAlpha = predicted.alpha - reference.alpha;

    acc.sum_L2 += dL * dL;
    acc.sum_a2 += da * da;
    acc.sum_b2 += db * db;
    acc.sum_alpha2 += dAlpha * dAlpha;
    ++acc.sample_count;

    acc.max_abs_L = Max(acc.max_abs_L, std::abs(dL));
    acc.max_abs_a = Max(acc.max_abs_a, std::abs(da));
    acc.max_abs_b = Max(acc.max_abs_b, std::abs(db));
    acc.max_abs_alpha = Max(acc.max_abs_alpha, std::abs(dAlpha));

    const f64 weighted = std::sqrt(dL * dL + da * da + db * db + alpha_weight * dAlpha * dAlpha);
    acc.max_weighted_error = Max(acc.max_weighted_error, weighted);
}

} // namespace

TriangleErrorMetrics ComputeTriangleError(
    const Mesh& mesh,
    TriangleId triangle_id,
    const ImageOKLaba& reference,
    const ErrorEstimatorOptions& options) {

    if (!reference.IsValid()) {
        throw std::runtime_error("ComputeTriangleError: reference image is invalid.");
    }
    if (!mesh.IsValidTriangleId(triangle_id)) {
        throw std::runtime_error("ComputeTriangleError: triangle id out of range.");
    }

    const Triangle& tri = mesh.triangles[triangle_id];
    if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
        return TriangleErrorMetrics{triangle_id};
    }

    const Vec2& p0 = TriangleP0(mesh, tri);
    const Vec2& p1 = TriangleP1(mesh, tri);
    const Vec2& p2 = TriangleP2(mesh, tri);

    i32 min_x = static_cast<i32>(std::floor(Min(p0.x, Min(p1.x, p2.x))));
    i32 max_x = static_cast<i32>(std::ceil (Max(p0.x, Max(p1.x, p2.x))));
    i32 min_y = static_cast<i32>(std::floor(Min(p0.y, Min(p1.y, p2.y))));
    i32 max_y = static_cast<i32>(std::ceil (Max(p0.y, Max(p1.y, p2.y))));

    if (options.clamp_triangle_bbox_to_image) {
        min_x = Clamp(min_x, 0, reference.Width() - 1);
        max_x = Clamp(max_x, 0, reference.Width() - 1);
        min_y = Clamp(min_y, 0, reference.Height() - 1);
        max_y = Clamp(max_y, 0, reference.Height() - 1);
    }

    SquaredErrorAccum acc{};
    const i32 spa = Max(options.samples_per_axis, 1);

    for (i32 y = min_y; y <= max_y; ++y) {
        for (i32 x = min_x; x <= max_x; ++x) {
            for (i32 sy = 0; sy < spa; ++sy) {
                for (i32 sx = 0; sx < spa; ++sx) {
                    const Vec2 p = SamplePointInsidePixel(x, y, sx, sy, spa);
                    const auto bc = ComputeBarycentric(p, p0, p1, p2);
                    if (!bc.has_value() || !bc->IsInside(options.inside_epsilon)) {
                        continue;
                    }

                    const ColorOKLaba predicted = InterpolateTriangleColor(mesh, tri, *bc);
                    const ColorOKLaba& truth = reference.At(x, y);
                    AccumulateTriangleError(acc, predicted, truth, options.alpha_weight);
                }
            }
        }
    }

    return FinalizeTriangleMetrics(triangle_id, acc, options.alpha_weight);
}

MeshErrorReport ComputeMeshError(
    const Mesh& mesh,
    const ImageOKLaba& reference,
    const ErrorEstimatorOptions& options) {

    if (!reference.IsValid()) {
        throw std::runtime_error("ComputeMeshError: reference image is invalid.");
    }

    std::string error;
    if (!ValidateMeshIndices(mesh, &error)) {
        throw std::runtime_error("ComputeMeshError: invalid mesh indices: " + error);
    }

    MeshErrorReport report{};
    report.summary.triangles_total = static_cast<u32>(mesh.triangles.size());
    report.per_triangle.reserve(mesh.triangles.size());

    SquaredErrorAccum global_acc{};

    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        const Triangle& tri = mesh.triangles[ti];
        if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
            ++report.summary.triangles_skipped_degenerate;
            report.per_triangle.push_back(TriangleErrorMetrics{ti});
            continue;
        }

        TriangleErrorMetrics tri_metrics = ComputeTriangleError(mesh, ti, reference, options);
        report.per_triangle.push_back(tri_metrics);
        ++report.summary.triangles_evaluated;

        if (!tri_metrics.HasSamples()) {
            continue;
        }

        global_acc.sample_count += tri_metrics.sample_count;
        global_acc.sum_L2 += tri_metrics.mse_L * static_cast<f64>(tri_metrics.sample_count);
        global_acc.sum_a2 += tri_metrics.mse_a * static_cast<f64>(tri_metrics.sample_count);
        global_acc.sum_b2 += tri_metrics.mse_b * static_cast<f64>(tri_metrics.sample_count);
        global_acc.sum_alpha2 += tri_metrics.mse_alpha * static_cast<f64>(tri_metrics.sample_count);
        global_acc.max_abs_L = Max(global_acc.max_abs_L, tri_metrics.max_abs_L);
        global_acc.max_abs_a = Max(global_acc.max_abs_a, tri_metrics.max_abs_a);
        global_acc.max_abs_b = Max(global_acc.max_abs_b, tri_metrics.max_abs_b);
        global_acc.max_abs_alpha = Max(global_acc.max_abs_alpha, tri_metrics.max_abs_alpha);
        global_acc.max_weighted_error = Max(global_acc.max_weighted_error, tri_metrics.max_weighted_error);

        if (tri_metrics.weighted_rmse > report.summary.worst_triangle_weighted_rmse) {
            report.summary.worst_triangle_weighted_rmse = tri_metrics.weighted_rmse;
            report.summary.worst_triangle_id = ti;
        }
    }

    report.summary.sample_count_total = global_acc.sample_count;
    if (global_acc.sample_count > 0) {
        const f64 inv_n = 1.0 / static_cast<f64>(global_acc.sample_count);
        report.summary.mse_L = global_acc.sum_L2 * inv_n;
        report.summary.mse_a = global_acc.sum_a2 * inv_n;
        report.summary.mse_b = global_acc.sum_b2 * inv_n;
        report.summary.mse_alpha = global_acc.sum_alpha2 * inv_n;
        report.summary.rmse_lab = std::sqrt(report.summary.mse_L + report.summary.mse_a + report.summary.mse_b);
        report.summary.rmse_alpha = std::sqrt(report.summary.mse_alpha);
        report.summary.weighted_rmse = std::sqrt(report.summary.mse_L + report.summary.mse_a + report.summary.mse_b + options.alpha_weight * report.summary.mse_alpha);
    }

    return report;
}

ImageOKLaba RenderTriangleErrorHeatmap(
    const Mesh& mesh,
    const ImageSize& output_size,
    const std::vector<TriangleErrorMetrics>& per_triangle,
    ErrorHeatmapMode mode,
    f64 normalization_weighted_rmse,
    bool skip_degenerate_triangles) {

    if (!output_size.IsValid()) {
        throw std::runtime_error("RenderTriangleErrorHeatmap: output_size is invalid.");
    }
    if (per_triangle.size() != mesh.triangles.size()) {
        throw std::runtime_error("RenderTriangleErrorHeatmap: per_triangle size must match triangle count.");
    }

    ImageOKLaba out(output_size, {0.0, 0.0, 0.0, 1.0});
    const f64 norm = normalization_weighted_rmse > 0.0 ? normalization_weighted_rmse : DefaultNormalization(per_triangle);

    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        const Triangle& tri = mesh.triangles[ti];
        if (skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
            continue;
        }

        const Vec2& p0 = TriangleP0(mesh, tri);
        const Vec2& p1 = TriangleP1(mesh, tri);
        const Vec2& p2 = TriangleP2(mesh, tri);

        const i32 min_x = Clamp(static_cast<i32>(std::floor(Min(p0.x, Min(p1.x, p2.x)))), 0, out.Width() - 1);
        const i32 max_x = Clamp(static_cast<i32>(std::ceil (Max(p0.x, Max(p1.x, p2.x)))), 0, out.Width() - 1);
        const i32 min_y = Clamp(static_cast<i32>(std::floor(Min(p0.y, Min(p1.y, p2.y)))), 0, out.Height() - 1);
        const i32 max_y = Clamp(static_cast<i32>(std::ceil (Max(p0.y, Max(p1.y, p2.y)))), 0, out.Height() - 1);

        const f64 t = per_triangle[ti].weighted_rmse / norm;
        const ColorOKLaba color = MakeHeatColor(t, mode);

        for (i32 y = min_y; y <= max_y; ++y) {
            for (i32 x = min_x; x <= max_x; ++x) {
                const Vec2 p{static_cast<f64>(x), static_cast<f64>(y)};
                const auto bc = ComputeBarycentric(p, p0, p1, p2);
                if (!bc.has_value() || !bc->IsInside(1e-7)) {
                    continue;
                }
                out.At(x, y) = color;
            }
        }
    }

    return out;
}

ImageOKLaba RenderMeshErrorHeatmap(
    const Mesh& mesh,
    const ImageOKLaba& reference,
    const ErrorEstimatorOptions& options,
    ErrorHeatmapMode mode,
    f64 normalization_weighted_rmse) {

    const MeshErrorReport report = ComputeMeshError(mesh, reference, options);
    return RenderTriangleErrorHeatmap(mesh, reference.Size(), report.per_triangle, mode, normalization_weighted_rmse, options.skip_degenerate_triangles);
}

} // namespace svec
