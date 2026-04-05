#include "svec/render/surface_plane_renderer.h"

#include <stdexcept>
#include <string>

#include "svec/math/geometry.h"
#include "svec/math/scalar.h"

namespace svec {
namespace {

[[nodiscard]] f64 DistancePointToSegment(const Vec2& p, const Vec2& a, const Vec2& b) noexcept {
    const Vec2 ab = b - a;
    const f64 ab_len2 = ab.LengthSquared();
    if (ab_len2 <= kEpsilon) {
        return Distance(p, a);
    }
    const f64 t = Clamp(Dot(p - a, ab) / ab_len2, 0.0, 1.0);
    const Vec2 q = a + ab * t;
    return Distance(p, q);
}

[[nodiscard]] Vec2 TransformSurfaceToTarget(const Vec2& p, const SurfaceRenderTransform& transform) noexcept {
    return {
        p.x * transform.scale_x + transform.offset_x,
        p.y * transform.scale_y + transform.offset_y
    };
}

[[nodiscard]] Vec2 TransformTargetToSurface(const Vec2& p, const SurfaceRenderTransform& transform) noexcept {
    return {
        (p.x - transform.offset_x) / transform.scale_x,
        (p.y - transform.offset_y) / transform.scale_y
    };
}

[[nodiscard]] bool CoversPixelCenter(const Vec2& p0, const Vec2& p1, const Vec2& p2, i32 x, i32 y) {
    const Vec2 p{static_cast<f64>(x), static_cast<f64>(y)};
    const auto bc = ComputeBarycentric(p, p0, p1, p2, 1e-9);
    return bc.has_value() && bc->IsInside(1e-7);
}

[[nodiscard]] ColorOKLaba TintFitError(const TrianglePlane& plane) noexcept {
    const f64 t = Saturate(plane.fit_rmse * 4.0);
    return {0.25 + 0.55 * t, 0.05 + 0.25 * t, 0.05 + 0.18 * (1.0 - t), 1.0};
}

} // namespace

SurfacePlaneRenderResult RenderMeshPlaneSurface(
    const Mesh& mesh,
    const std::vector<TrianglePlane>& planes,
    const ImageSize& output_size,
    const SurfacePlaneRenderOptions& options) {

    if (!output_size.IsValid()) {
        throw std::runtime_error("RenderMeshPlaneSurface: invalid output_size.");
    }
    if (planes.size() != mesh.triangles.size()) {
        throw std::runtime_error("RenderMeshPlaneSurface: planes.size() must equal mesh.triangles.size().");
    }
    if (options.transform.scale_x <= kEpsilon || options.transform.scale_y <= kEpsilon) {
        throw std::runtime_error("RenderMeshPlaneSurface: render transform scale must be positive.");
    }

    std::string error;
    if (!ValidateMeshIndices(mesh, &error)) {
        throw std::runtime_error("RenderMeshPlaneSurface: invalid mesh indices: " + error);
    }

    SurfacePlaneRenderResult out{};
    out.image.Resize(output_size, options.clear_color);
    out.stats.triangles_total = static_cast<u32>(mesh.triangles.size());

    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        const Triangle& tri = mesh.triangles[ti];
        if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
            ++out.stats.triangles_skipped_degenerate;
            continue;
        }

        const Vec2 p0 = TransformSurfaceToTarget(TriangleP0(mesh, tri), options.transform);
        const Vec2 p1 = TransformSurfaceToTarget(TriangleP1(mesh, tri), options.transform);
        const Vec2 p2 = TransformSurfaceToTarget(TriangleP2(mesh, tri), options.transform);
        const i32 min_x = Clamp(static_cast<i32>(std::floor(Min(p0.x, Min(p1.x, p2.x)))), 0, output_size.width - 1);
        const i32 min_y = Clamp(static_cast<i32>(std::floor(Min(p0.y, Min(p1.y, p2.y)))), 0, output_size.height - 1);
        const i32 max_x = Clamp(static_cast<i32>(std::ceil(Max(p0.x, Max(p1.x, p2.x)))), 0, output_size.width - 1);
        const i32 max_y = Clamp(static_cast<i32>(std::ceil(Max(p0.y, Max(p1.y, p2.y)))), 0, output_size.height - 1);

        bool rasterized = false;
        for (i32 y = min_y; y <= max_y; ++y) {
            for (i32 x = min_x; x <= max_x; ++x) {
                if (!CoversPixelCenter(p0, p1, p2, x, y)) {
                    continue;
                }
                const Vec2 target_p{static_cast<f64>(x), static_cast<f64>(y)};
                const Vec2 surface_p = TransformTargetToSurface(target_p, options.transform);
                ColorOKLaba color = options.mode == SurfacePlaneRenderMode::PlaneShaded
                    ? EvaluateTrianglePlane(planes[ti], surface_p.x, surface_p.y, true)
                    : TintFitError(planes[ti]);

                if (options.overlay_wireframe) {
                    const Vec2 p{static_cast<f64>(x), static_cast<f64>(y)};
                    const f64 d = Min(
                        DistancePointToSegment(p, p0, p1),
                        Min(DistancePointToSegment(p, p1, p2), DistancePointToSegment(p, p2, p0)));
                    if (d <= options.wire_half_width_px) {
                        color = options.wire_color;
                        ++out.stats.wire_pixels_shaded;
                    }
                }
                out.image.At(x, y) = color;
                ++out.stats.pixels_shaded;
                rasterized = true;
            }
        }
        if (rasterized) {
            ++out.stats.triangles_rasterized;
        }
    }

    return out;
}

} // namespace svec
