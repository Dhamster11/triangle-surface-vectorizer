#include "svec/surface/triangle_plane.h"

#include <stdexcept>
#include <vector>

#include "svec/math/geometry.h"
#include "svec/math/scalar.h"
#include "svec/refine/adaptive_refinement.h"

namespace svec {
namespace {

struct SamplePoint {
    Vec2 p{};
    ColorOKLaba c{};
};

void CollectTriangleSamples(
    const Mesh& mesh,
    const Triangle& tri,
    const ImageOKLaba& reference,
    const TrianglePlaneFitOptions& options,
    std::vector<SamplePoint>& out_samples) {

    const Vec2 p0 = TriangleP0(mesh, tri);
    const Vec2 p1 = TriangleP1(mesh, tri);
    const Vec2 p2 = TriangleP2(mesh, tri);

    auto push_unique = [&](const Vec2& p) {
        for (const auto& s : out_samples) {
            if (DistanceSquared(s.p, p) <= 1e-12) {
                return;
            }
        }
        out_samples.push_back({p, SampleImageOKLabaBilinear(reference, p)});
    };

    if (options.include_vertices) {
        push_unique(p0);
        push_unique(p1);
        push_unique(p2);
    }
    if (options.include_edge_midpoints) {
        push_unique(Midpoint(p0, p1));
        push_unique(Midpoint(p1, p2));
        push_unique(Midpoint(p2, p0));
    }
    if (options.include_centroid) {
        push_unique(TriangleCentroid(p0, p1, p2));
    }

    if (options.interior_barycentric_samples > 0) {
        const u32 n = options.interior_barycentric_samples + 3u;
        for (u32 iy = 1; iy + 1 < n; ++iy) {
            for (u32 ix = 1; ix + iy + 1 < n; ++ix) {
                const f64 u = static_cast<f64>(ix) / static_cast<f64>(n);
                const f64 v = static_cast<f64>(iy) / static_cast<f64>(n);
                const f64 w = 1.0 - u - v;
                if (w <= 0.0) continue;
                push_unique(p0 * u + p1 * v + p2 * w);
            }
        }
    }
}

Plane1 FitSingleChannelPlaneSamples(const std::vector<SamplePoint>& samples, f64 (*getter)(const ColorOKLaba&)) {
    Plane1 out{};
    if (samples.size() < 3) {
        if (!samples.empty()) out.c = getter(samples.front().c);
        return out;
    }

    f64 s1 = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
    f64 sz = 0.0, sxz = 0.0, syz = 0.0;
    for (const auto& s : samples) {
        const f64 x = s.p.x;
        const f64 y = s.p.y;
        const f64 z = getter(s.c);
        s1 += 1.0; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
        sz += z; sxz += x * z; syz += y * z;
    }

    const f64 det = s1 * (sxx * syy - sxy * sxy)
                  - sx * (sx * syy - sy * sxy)
                  + sy * (sx * sxy - sy * sxx);
    if (std::abs(det) <= 1e-12) {
        out.c = sz / Max(s1, 1.0);
        return out;
    }

    const f64 det_c = sz * (sxx * syy - sxy * sxy)
                    - sx * (sxz * syy - syz * sxy)
                    + sy * (sxz * sxy - syz * sxx);
    const f64 det_x = s1 * (sxz * syy - syz * sxy)
                    - sz * (sx * syy - sy * sxy)
                    + sy * (sx * syz - sy * sxz);
    const f64 det_y = s1 * (sxx * syz - sxy * sxz)
                    - sx * (sx * syz - sy * sxz)
                    + sz * (sx * sxy - sy * sxx);

    out.c = det_c / det;
    out.cx = det_x / det;
    out.cy = det_y / det;
    if (!std::isfinite(out.c) || !std::isfinite(out.cx) || !std::isfinite(out.cy)) {
        out = {};
        out.c = sz / Max(s1, 1.0);
    }
    return out;
}

Plane1 FitSingleChannelPlaneMoments(f64 n, f64 sx, f64 sy, f64 sxx, f64 syy, f64 sxy, f64 sz, f64 sxz, f64 syz) {
    Plane1 out{};
    if (n < 3.0) {
        out.c = n > 0.0 ? sz / n : 0.0;
        return out;
    }

    const f64 det = n * (sxx * syy - sxy * sxy)
                  - sx * (sx * syy - sy * sxy)
                  + sy * (sx * sxy - sy * sxx);
    if (std::abs(det) <= 1e-12) {
        out.c = sz / Max(n, 1.0);
        return out;
    }

    const f64 det_c = sz * (sxx * syy - sxy * sxy)
                    - sx * (sxz * syy - syz * sxy)
                    + sy * (sxz * sxy - syz * sxx);
    const f64 det_x = n * (sxz * syy - syz * sxy)
                    - sz * (sx * syy - sy * sxy)
                    + sy * (sx * syz - sy * sxz);
    const f64 det_y = n * (sxx * syz - sxy * sxz)
                    - sx * (sx * syz - sy * sxz)
                    + sz * (sx * sxy - sy * sxx);

    out.c = det_c / det;
    out.cx = det_x / det;
    out.cy = det_y / det;
    if (!std::isfinite(out.c) || !std::isfinite(out.cx) || !std::isfinite(out.cy)) {
        out = {};
        out.c = sz / Max(n, 1.0);
    }
    return out;
}

f64 GetL(const ColorOKLaba& c) { return c.L; }
f64 Geta(const ColorOKLaba& c) { return c.a; }
f64 Getb(const ColorOKLaba& c) { return c.b; }
f64 GetAlpha(const ColorOKLaba& c) { return c.alpha; }

TrianglePlane FitTrianglePlaneFallback(
    const Mesh& mesh,
    TriangleId triangle_id,
    const ImageOKLaba& reference,
    const TrianglePlaneFitOptions& options) {

    const Triangle& tri = mesh.triangles.at(triangle_id);
    std::vector<SamplePoint> samples;
    samples.reserve(16);
    CollectTriangleSamples(mesh, tri, reference, options, samples);

    TrianglePlane out{};
    out.triangle_id = triangle_id;
    out.fit_sample_count = static_cast<u32>(samples.size());
    out.L = FitSingleChannelPlaneSamples(samples, GetL);
    out.a = FitSingleChannelPlaneSamples(samples, Geta);
    out.b = FitSingleChannelPlaneSamples(samples, Getb);
    out.alpha = FitSingleChannelPlaneSamples(samples, GetAlpha);

    f64 se = 0.0;
    for (const auto& s : samples) {
        const ColorOKLaba p = EvaluateTrianglePlane(out, s.p.x, s.p.y, options.clamp_alpha);
        const f64 dL = p.L - s.c.L;
        const f64 da = p.a - s.c.a;
        const f64 db = p.b - s.c.b;
        const f64 dAlpha = p.alpha - s.c.alpha;
        se += dL * dL + da * da + db * db + dAlpha * dAlpha;
    }
    out.fit_rmse = samples.empty() ? 0.0 : std::sqrt(se / static_cast<f64>(samples.size()));
    return out;
}

} // namespace

TrianglePlane FitTrianglePlane(
    const Mesh& mesh,
    TriangleId triangle_id,
    const ImageOKLaba& reference,
    const TrianglePlaneFitOptions& options) {
    return FitTrianglePlane(mesh, triangle_id, reference, nullptr, options);
}

TrianglePlane FitTrianglePlane(
    const Mesh& mesh,
    TriangleId triangle_id,
    const ImageOKLaba& reference,
    const ScanlineIntegralStats* stats,
    const TrianglePlaneFitOptions& options) {

    if (!mesh.IsValidTriangleId(triangle_id)) {
        throw std::runtime_error("FitTrianglePlane: triangle id out of range.");
    }
    const Triangle& tri = mesh.triangles.at(triangle_id);
    if (IsDegenerate(mesh, tri)) {
        return {triangle_id, {}, {}, {}, {}, 0.0, 0};
    }

    if (stats != nullptr && options.prefer_scanline_integral_stats && stats->IsValid()) {
        const TriangleScanlineMoments m = stats->AccumulateTriangle(TriangleP0(mesh, tri), TriangleP1(mesh, tri), TriangleP2(mesh, tri));
        if (!m.IsEmpty()) {
            TrianglePlane out{};
            out.triangle_id = triangle_id;
            out.fit_sample_count = static_cast<u32>(m.count);
            out.L = FitSingleChannelPlaneMoments(m.count, m.sum_x, m.sum_y, m.sum_x2, m.sum_y2, m.sum_xy, m.sum_L, m.sum_xL, m.sum_yL);
            out.a = FitSingleChannelPlaneMoments(m.count, m.sum_x, m.sum_y, m.sum_x2, m.sum_y2, m.sum_xy, m.sum_a, m.sum_xa, m.sum_ya);
            out.b = FitSingleChannelPlaneMoments(m.count, m.sum_x, m.sum_y, m.sum_x2, m.sum_y2, m.sum_xy, m.sum_b, m.sum_xb, m.sum_yb);
            out.alpha = FitSingleChannelPlaneMoments(m.count, m.sum_x, m.sum_y, m.sum_x2, m.sum_y2, m.sum_xy, m.sum_alpha, m.sum_xalpha, m.sum_yalpha);

            const auto channel_rmse = [&](const Plane1& p, f64 sumz, f64 sumxz, f64 sumyz, f64 sumz2) {
                const f64 a = p.c, b = p.cx, c = p.cy;
                f64 sse = sumz2
                        - 2.0 * a * sumz - 2.0 * b * sumxz - 2.0 * c * sumyz
                        + a * a * m.count + 2.0 * a * b * m.sum_x + 2.0 * a * c * m.sum_y
                        + b * b * m.sum_x2 + 2.0 * b * c * m.sum_xy + c * c * m.sum_y2;
                sse = Max(0.0, sse);
                return sse;
            };

            const f64 sse = channel_rmse(out.L, m.sum_L, m.sum_xL, m.sum_yL, m.sum_L2)
                          + channel_rmse(out.a, m.sum_a, m.sum_xa, m.sum_ya, m.sum_a2)
                          + channel_rmse(out.b, m.sum_b, m.sum_xb, m.sum_yb, m.sum_b2)
                          + channel_rmse(out.alpha, m.sum_alpha, m.sum_xalpha, m.sum_yalpha, m.sum_alpha2);
            out.fit_rmse = std::sqrt(sse / Max(m.count, 1.0));
            return out;
        }
    }

    return FitTrianglePlaneFallback(mesh, triangle_id, reference, options);
}

std::vector<TrianglePlane> FitAllTrianglePlanes(
    const Mesh& mesh,
    const ImageOKLaba& reference,
    const TrianglePlaneFitOptions& options) {
    return FitAllTrianglePlanes(mesh, reference, nullptr, options);
}

std::vector<TrianglePlane> FitAllTrianglePlanes(
    const Mesh& mesh,
    const ImageOKLaba& reference,
    const ScanlineIntegralStats* stats,
    const TrianglePlaneFitOptions& options) {

    std::vector<TrianglePlane> out;
    out.reserve(mesh.triangles.size());
    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        out.push_back(FitTrianglePlane(mesh, ti, reference, stats, options));
    }
    return out;
}

ColorOKLaba EvaluateTrianglePlane(const TrianglePlane& plane, f64 x, f64 y, bool clamp_alpha) noexcept {
    ColorOKLaba out{
        plane.L.Evaluate(x, y),
        plane.a.Evaluate(x, y),
        plane.b.Evaluate(x, y),
        plane.alpha.Evaluate(x, y)
    };
    if (clamp_alpha) {
        out.alpha = Saturate(out.alpha);
    }
    return out;
}

} // namespace svec
