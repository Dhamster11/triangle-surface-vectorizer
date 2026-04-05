#include "svec/render/reference_renderer.h"

#include <cmath>
#include <stdexcept>

#include "svec/math/geometry.h"
#include "svec/math/scalar.h"
#include "svec/math/vec2.h"

namespace svec {
namespace {

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
        c0.alpha * bc.u + c1.alpha * bc.v + c2.alpha * bc.w
    };
}

[[nodiscard]] u32 Hash32(u32 x) noexcept {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

[[nodiscard]] f64 HashToUnit(u32 seed) noexcept {
    return static_cast<f64>(Hash32(seed) & 0x00ffffffu) / static_cast<f64>(0x00ffffffu);
}

[[nodiscard]] ColorOKLaba TriangleIdDebugColor(TriangleId id) noexcept {
    const f64 h0 = HashToUnit(id * 3u + 1u);
    const f64 h1 = HashToUnit(id * 3u + 2u);
    const f64 h2 = HashToUnit(id * 3u + 3u);

    return {
        0.45 + 0.35 * h0,
        -0.18 + 0.36 * h1,
        -0.18 + 0.36 * h2,
        1.0
    };
}

[[nodiscard]] f64 DistancePointToSegment(const Vec2& p, const Vec2& a, const Vec2& b) noexcept {
    const Vec2 ab = b - a;
    const f64 denom = ab.LengthSquared();
    if (denom <= kEpsilon) {
        return Distance(p, a);
    }

    const f64 t = Saturate(Dot(p - a, ab) / denom);
    const Vec2 q = a + ab * t;
    return Distance(p, q);
}

[[nodiscard]] bool IsPointNearTriangleEdge(
    const Vec2& p,
    const Vec2& a,
    const Vec2& b,
    const Vec2& c,
    f64 half_width_px) noexcept {

    return DistancePointToSegment(p, a, b) <= half_width_px ||
           DistancePointToSegment(p, b, c) <= half_width_px ||
           DistancePointToSegment(p, c, a) <= half_width_px;
}

void RasterizeTriangleFill(
    const Mesh& mesh,
    TriangleId triangle_id,
    const Triangle& tri,
    ImageOKLaba& target,
    const ReferenceRenderOptions& options,
    ReferenceRenderStats& stats) {

    const Vec2& p0 = TriangleP0(mesh, tri);
    const Vec2& p1 = TriangleP1(mesh, tri);
    const Vec2& p2 = TriangleP2(mesh, tri);

    const f64 min_x_f = Min(p0.x, Min(p1.x, p2.x));
    const f64 max_x_f = Max(p0.x, Max(p1.x, p2.x));
    const f64 min_y_f = Min(p0.y, Min(p1.y, p2.y));
    const f64 max_y_f = Max(p0.y, Max(p1.y, p2.y));

    const i32 min_x = Clamp(static_cast<i32>(std::floor(min_x_f)), 0, target.Width() - 1);
    const i32 max_x = Clamp(static_cast<i32>(std::ceil (max_x_f)), 0, target.Width() - 1);
    const i32 min_y = Clamp(static_cast<i32>(std::floor(min_y_f)), 0, target.Height() - 1);
    const i32 max_y = Clamp(static_cast<i32>(std::ceil (max_y_f)), 0, target.Height() - 1);

    for (i32 y = min_y; y <= max_y; ++y) {
        for (i32 x = min_x; x <= max_x; ++x) {
            const Vec2 p{static_cast<f64>(x), static_cast<f64>(y)};
            const auto bc = ComputeBarycentric(p, p0, p1, p2);
            if (!bc.has_value() || !bc->IsInside(1e-7)) {
                continue;
            }

            ColorOKLaba color{};
            switch (options.mode) {
                case ReferenceRenderMode::InterpolatedColor:
                    color = InterpolateTriangleColor(mesh, tri, *bc);
                    break;
                case ReferenceRenderMode::TriangleIdFlat:
                    color = TriangleIdDebugColor(triangle_id);
                    break;
                default:
                    color = InterpolateTriangleColor(mesh, tri, *bc);
                    break;
            }

            target.At(x, y) = color;
            ++stats.pixels_shaded;
        }
    }
}

void RasterizeTriangleWireframe(
    const Mesh& mesh,
    const Triangle& tri,
    ImageOKLaba& target,
    const ReferenceRenderOptions& options,
    ReferenceRenderStats& stats) {

    const Vec2& p0 = TriangleP0(mesh, tri);
    const Vec2& p1 = TriangleP1(mesh, tri);
    const Vec2& p2 = TriangleP2(mesh, tri);

    const f64 grow = Max(options.wire_half_width_px, 0.0);

    const f64 min_x_f = Min(p0.x, Min(p1.x, p2.x)) - grow;
    const f64 max_x_f = Max(p0.x, Max(p1.x, p2.x)) + grow;
    const f64 min_y_f = Min(p0.y, Min(p1.y, p2.y)) - grow;
    const f64 max_y_f = Max(p0.y, Max(p1.y, p2.y)) + grow;

    const i32 min_x = Clamp(static_cast<i32>(std::floor(min_x_f)), 0, target.Width() - 1);
    const i32 max_x = Clamp(static_cast<i32>(std::ceil (max_x_f)), 0, target.Width() - 1);
    const i32 min_y = Clamp(static_cast<i32>(std::floor(min_y_f)), 0, target.Height() - 1);
    const i32 max_y = Clamp(static_cast<i32>(std::ceil (max_y_f)), 0, target.Height() - 1);

    for (i32 y = min_y; y <= max_y; ++y) {
        for (i32 x = min_x; x <= max_x; ++x) {
            const Vec2 p{static_cast<f64>(x), static_cast<f64>(y)};
            const auto bc = ComputeBarycentric(p, p0, p1, p2);
            if (!bc.has_value() || !bc->IsInside(1e-7)) {
                continue;
            }

            if (!IsPointNearTriangleEdge(p, p0, p1, p2, options.wire_half_width_px)) {
                continue;
            }

            target.At(x, y) = options.wire_color;
            ++stats.wire_pixels_shaded;
        }
    }
}

} // namespace

ReferenceRenderResult RenderMeshReference(
    const Mesh& mesh,
    const ImageSize& output_size,
    const ReferenceRenderOptions& options) {

    ReferenceRenderResult result;
    result.image.Resize(output_size, options.clear_color);
    RenderMeshReferenceTo(mesh, result.image, options, &result.stats);
    return result;
}

void RenderMeshReferenceTo(
    const Mesh& mesh,
    ImageOKLaba& target,
    const ReferenceRenderOptions& options,
    ReferenceRenderStats* out_stats) {

    if (!target.IsValid()) {
        throw std::runtime_error("RenderMeshReferenceTo: target image is invalid.");
    }

    std::string error;
    if (!ValidateMeshIndices(mesh, &error)) {
        throw std::runtime_error("RenderMeshReferenceTo: invalid mesh indices: " + error);
    }

    for (i32 y = 0; y < target.Height(); ++y) {
        for (i32 x = 0; x < target.Width(); ++x) {
            target.At(x, y) = options.clear_color;
        }
    }

    ReferenceRenderStats stats{};
    stats.triangles_total = static_cast<u32>(mesh.triangles.size());

    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        const Triangle& tri = mesh.triangles[ti];
        if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
            ++stats.triangles_skipped_degenerate;
            continue;
        }

        RasterizeTriangleFill(mesh, ti, tri, target, options, stats);
        ++stats.triangles_rasterized;
    }

    if (options.overlay_wireframe) {
        for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
            const Triangle& tri = mesh.triangles[ti];
            if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
                continue;
            }
            RasterizeTriangleWireframe(mesh, tri, target, options, stats);
        }
    }

    if (out_stats) {
        *out_stats = stats;
    }
}

} // namespace svec
