#include "svec/error/hierarchical_surface_error.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "svec/math/geometry.h"
#include "svec/math/scalar.h"
#include "svec/refine/adaptive_refinement.h"

namespace svec {
namespace {

[[nodiscard]] std::vector<Vec2> GenerateTriangleSamplePoints(const Mesh& mesh, const Triangle& tri, u32 approx_count) {
    std::vector<Vec2> pts;
    pts.reserve(16);
    const Vec2 p0 = TriangleP0(mesh, tri);
    const Vec2 p1 = TriangleP1(mesh, tri);
    const Vec2 p2 = TriangleP2(mesh, tri);

    pts.push_back(p0);
    pts.push_back(p1);
    pts.push_back(p2);
    pts.push_back(Midpoint(p0, p1));
    pts.push_back(Midpoint(p1, p2));
    pts.push_back(Midpoint(p2, p0));
    pts.push_back(TriangleCentroid(p0, p1, p2));

    if (approx_count > pts.size()) {
        const u32 n = 4;
        for (u32 iy = 1; iy < n; ++iy) {
            for (u32 ix = 1; ix + iy < n; ++ix) {
                const f64 u = static_cast<f64>(ix) / static_cast<f64>(n);
                const f64 v = static_cast<f64>(iy) / static_cast<f64>(n);
                const f64 w = 1.0 - u - v;
                if (w <= 0.0) continue;
                pts.push_back(p0 * u + p1 * v + p2 * w);
            }
        }
    }
    return pts;
}

[[nodiscard]] f64 ColorResidual2(const ColorOKLaba& ref, const ColorOKLaba& fit, f64 alpha_weight) noexcept {
    const f64 dL = ref.L - fit.L;
    const f64 da = ref.a - fit.a;
    const f64 db = ref.b - fit.b;
    const f64 dAlpha = ref.alpha - fit.alpha;
    return dL * dL + da * da + db * db + alpha_weight * dAlpha * dAlpha;
}

[[nodiscard]] f64 GradientResidual2(const TrianglePlane& plane, f64 ref_edge_strength) noexcept {
    const f64 fit_grad = std::sqrt(
        plane.L.cx * plane.L.cx + plane.L.cy * plane.L.cy +
        plane.a.cx * plane.a.cx + plane.a.cy * plane.a.cy +
        plane.b.cx * plane.b.cx + plane.b.cy * plane.b.cy +
        0.25 * (plane.alpha.cx * plane.alpha.cx + plane.alpha.cy * plane.alpha.cy));
    const f64 d = fit_grad - ref_edge_strength;
    return d * d;
}

[[nodiscard]] f64 ComputeBBoxStructureMean(const ErrorPyramidLevel& level, const Vec2& min_p, const Vec2& max_p) {
    if (!level.gradient_energy_integral.IsValid()) {
        return 0.0;
    }
    const i32 x0 = Clamp(static_cast<i32>(std::floor(min_p.x / level.scale)), 0, level.image.Width() - 1);
    const i32 y0 = Clamp(static_cast<i32>(std::floor(min_p.y / level.scale)), 0, level.image.Height() - 1);
    const i32 x1 = Clamp(static_cast<i32>(std::ceil(max_p.x / level.scale)) + 1, 1, level.image.Width());
    const i32 y1 = Clamp(static_cast<i32>(std::ceil(max_p.y / level.scale)) + 1, 1, level.image.Height());
    return level.gradient_energy_integral.MeanRect(x0, y0, x1, y1);
}

[[nodiscard]] f64 SampleCentroidEdgeStrength(const ErrorPyramidLevel& level, const Vec2& centroid) {
    if (!level.edge_map.IsValid()) {
        return 0.0;
    }
    const Vec2 q = centroid / level.scale;
    return SampleEdgeMapBilinear(level.edge_map, q);
}

} // namespace

TriangleHierarchicalError ComputeTriangleHierarchicalSurfaceError(
    const Mesh& mesh,
    TriangleId triangle_id,
    const ErrorPyramid& pyramid,
    const TrianglePlane& plane,
    const HierarchicalSurfaceErrorOptions& options) {

    if (!mesh.IsValidTriangleId(triangle_id)) {
        throw std::runtime_error("ComputeTriangleHierarchicalSurfaceError: triangle id out of range.");
    }
    if (!pyramid.IsValid()) {
        throw std::runtime_error("ComputeTriangleHierarchicalSurfaceError: pyramid is invalid.");
    }

    const Triangle& tri = mesh.triangles.at(triangle_id);
    TriangleHierarchicalError out{};
    out.triangle_id = triangle_id;
    out.triangle_area = ComputeTriangleArea(mesh, tri);
    if (IsDegenerate(mesh, tri) || out.triangle_area <= 1e-12) {
        return out;
    }

    const Vec2 p0 = TriangleP0(mesh, tri);
    const Vec2 p1 = TriangleP1(mesh, tri);
    const Vec2 p2 = TriangleP2(mesh, tri);
    const Vec2 centroid = TriangleCentroid(p0, p1, p2);
    const Vec2 min_p{Min(p0.x, Min(p1.x, p2.x)), Min(p0.y, Min(p1.y, p2.y))};
    const Vec2 max_p{Max(p0.x, Max(p1.x, p2.x)), Max(p0.y, Max(p1.y, p2.y))};

    const std::vector<Vec2> base_samples = GenerateTriangleSamplePoints(mesh, tri, options.per_level_samples);
    if (base_samples.empty()) {
        return out;
    }

    f64 accum_color2 = 0.0;
    f64 accum_grad2 = 0.0;
    f64 accum_detail2 = 0.0;
    f64 accum_structure = 0.0;
    f64 peak_residual2 = 0.0;

    // Cheap extra statistics used to make the metric stricter on specular highlights
    // and soft non-linear tone changes without adding new sampling passes.
    f64 accum_luma_residual2 = 0.0;
    f64 peak_luma_residual2 = 0.0;
    f64 accum_luma = 0.0;
    f64 accum_luma2 = 0.0;
    u64 accum_samples = 0;

    const u32 level_count = Min<u32>(options.max_levels_used, static_cast<u32>(pyramid.levels.size()));
    for (u32 level_index = 0; level_index < level_count; ++level_index) {
        const auto& level = pyramid.levels[level_index];
        const f64 level_weight = 1.0 / (1.0 + 0.75 * static_cast<f64>(level_index));

        const f64 bbox_structure = ComputeBBoxStructureMean(level, min_p, max_p);
        const f64 centroid_edge = SampleCentroidEdgeStrength(level, centroid);
        const f64 structure_root = 0.35 * std::sqrt(Max(bbox_structure, 0.0)) + 0.65 * centroid_edge;
        accum_structure += level_weight * structure_root * structure_root;

        for (const Vec2& p : base_samples) {
            const Vec2 q = p / level.scale;
            const ColorOKLaba ref = SampleImageOKLabaBilinear(level.image, q);
            const ColorOKLaba fit = EvaluateTrianglePlane(plane, p.x, p.y, options.clamp_alpha);
            const f64 residual2 = ColorResidual2(ref, fit, options.alpha_weight);
            accum_color2 += level_weight * residual2;
            peak_residual2 = Max(peak_residual2, residual2);

            const f64 dL = ref.L - fit.L;
            const f64 luma_residual2 = dL * dL;
            accum_luma_residual2 += level_weight * luma_residual2;
            peak_luma_residual2 = Max(peak_luma_residual2, luma_residual2);
            accum_luma += level_weight * ref.L;
            accum_luma2 += level_weight * ref.L * ref.L;

            const f64 ref_edge = centroid_edge;
            accum_grad2 += level_weight * GradientResidual2(plane, ref_edge);
            if (level_index > 0) {
                const ColorOKLaba base_ref = SampleImageOKLabaBilinear(pyramid.Base().image, p);
                const f64 dL = base_ref.L - ref.L;
                const f64 da = base_ref.a - ref.a;
                const f64 db = base_ref.b - ref.b;
                accum_detail2 += level_weight * (dL * dL + da * da + db * db);
            }
            ++accum_samples;
        }
    }

    out.sample_count = accum_samples;
    if (accum_samples > 0) {
        const f64 sample_inv = 1.0 / static_cast<f64>(accum_samples);
        out.color_rmse = std::sqrt(accum_color2 * sample_inv);
        out.gradient_rmse = std::sqrt(accum_grad2 * sample_inv);
        out.detail_rmse = std::sqrt(accum_detail2 * sample_inv);
        out.structure_mean = std::sqrt(Max(accum_structure / static_cast<f64>(level_count), 0.0));
        out.peak_residual = std::sqrt(peak_residual2);

        const f64 luma_rmse = std::sqrt(accum_luma_residual2 * sample_inv);
        const f64 luma_peak_residual = std::sqrt(peak_luma_residual2);
        const f64 luma_mean = accum_luma * sample_inv;
        const f64 luma_variance = Max(accum_luma2 * sample_inv - luma_mean * luma_mean, 0.0);
        const f64 luma_std = std::sqrt(luma_variance);

        // Highlight / curved-tone booster:
        // - luma_focus grows when a localized bright residual is much stronger than the average
        //   luminance mismatch inside the triangle (classic specular highlight case);
        // - tone_nonlinearity grows when the triangle spans a noticeable luminance spread and
        //   the linear plane still leaves a meaningful luminance error.
        const f64 luma_focus = Max(luma_peak_residual - 0.65 * luma_rmse, 0.0);
        const f64 tone_nonlinearity = luma_rmse * (0.30 + 0.70 * Saturate(luma_std / 0.12));
        const f64 quality_boost = 0.70 * luma_focus + 0.45 * tone_nonlinearity;

        out.composite_error =
            options.color_weight * out.color_rmse +
            options.gradient_weight * out.gradient_rmse +
            options.detail_weight * out.detail_rmse +
            options.structure_weight * out.structure_mean +
            options.peak_weight * out.peak_residual +
            quality_boost;
        const f64 area_factor = std::sqrt(Max(out.triangle_area, 1e-6));
        const f64 heap_tie_break = 1e-9 * static_cast<f64>(triangle_id + 1u);
        const f64 strict_quality_term = 1.0 + 0.75 * out.structure_mean + 0.50 * out.peak_residual + 0.90 * quality_boost;
        out.weighted_score =
            out.composite_error * area_factor * strict_quality_term +
            heap_tie_break;
    }

    return out;
}

MeshHierarchicalErrorSummary ComputeMeshHierarchicalSurfaceError(
    const Mesh& mesh,
    const ErrorPyramid& pyramid,
    const std::vector<TrianglePlane>& planes,
    const HierarchicalSurfaceErrorOptions& options) {

    MeshHierarchicalErrorSummary out{};
    if (mesh.triangles.empty()) {
        return out;
    }
    f64 sum = 0.0;
    u64 count = 0;
    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        const TriangleHierarchicalError e = ComputeTriangleHierarchicalSurfaceError(mesh, ti, pyramid, planes.at(ti), options);
        sum += e.composite_error;
        out.sample_count += e.sample_count;
        if (e.composite_error > out.max_composite_error) {
            out.max_composite_error = e.composite_error;
            out.worst_triangle_id = ti;
        }
        ++count;
    }
    out.mean_composite_error = count > 0 ? sum / static_cast<f64>(count) : 0.0;
    return out;
}

} // namespace svec
