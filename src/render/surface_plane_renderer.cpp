#include "svec/render/surface_plane_renderer.h"

#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "svec/math/geometry.h"
#include "svec/math/scalar.h"

namespace svec {
    namespace {

        constexpr i32 kMsaaGridSize = 4;
        constexpr i32 kMsaaSampleCount = kMsaaGridSize * kMsaaGridSize;
        constexpr std::uint16_t kFullCoverageMask = static_cast<std::uint16_t>((1u << kMsaaSampleCount) - 1u);

        [[nodiscard]] constexpr std::array<Vec2, kMsaaSampleCount> BuildMsaaPattern() noexcept {
            return { {
                {0.125, 0.125}, {0.375, 0.125}, {0.625, 0.125}, {0.875, 0.125},
                {0.125, 0.375}, {0.375, 0.375}, {0.625, 0.375}, {0.875, 0.375},
                {0.125, 0.625}, {0.375, 0.625}, {0.625, 0.625}, {0.875, 0.625},
                {0.125, 0.875}, {0.375, 0.875}, {0.625, 0.875}, {0.875, 0.875}
            } };
        }

        constexpr std::array<Vec2, kMsaaSampleCount> kMsaaPattern = BuildMsaaPattern();

        [[nodiscard]] ColorOKLaba MakeZeroColor() noexcept {
            return { 0.0, 0.0, 0.0, 0.0 };
        }

        void ClearImage(ImageOKLaba& image, const ColorOKLaba& color) {
            for (i32 y = 0; y < image.Height(); ++y) {
                for (i32 x = 0; x < image.Width(); ++x) {
                    image.At(x, y) = color;
                }
            }
        }

        void AddScaledColor(ColorOKLaba& dst, const ColorOKLaba& src, f64 scale) noexcept {
            dst.L += src.L * scale;
            dst.a += src.a * scale;
            dst.b += src.b * scale;
            dst.alpha += src.alpha * scale;
        }

        void ScaleColorInPlace(ColorOKLaba& value, f64 scale) noexcept {
            value.L *= scale;
            value.a *= scale;
            value.b *= scale;
            value.alpha *= scale;
        }

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

        [[nodiscard]] bool PointInsideTriangle(
            const Vec2& p,
            const Vec2& p0,
            const Vec2& p1,
            const Vec2& p2) noexcept {

            const auto bc = ComputeBarycentric(p, p0, p1, p2, 1e-12);
            return bc.has_value() && bc->IsInside(1e-9);
        }

        [[nodiscard]] bool PixelSquareFullyInsideTriangle(
            i32 x,
            i32 y,
            const Vec2& p0,
            const Vec2& p1,
            const Vec2& p2) noexcept {

            const f64 fx = static_cast<f64>(x);
            const f64 fy = static_cast<f64>(y);

            return PointInsideTriangle({ fx, fy }, p0, p1, p2) &&
                PointInsideTriangle({ fx + 1.0, fy }, p0, p1, p2) &&
                PointInsideTriangle({ fx, fy + 1.0 }, p0, p1, p2) &&
                PointInsideTriangle({ fx + 1.0, fy + 1.0 }, p0, p1, p2);
        }

        [[nodiscard]] ColorOKLaba TintFitError(const TrianglePlane& plane) noexcept {
            const f64 t = Saturate(plane.fit_rmse * 4.0);
            return { 0.25 + 0.55 * t, 0.05 + 0.25 * t, 0.05 + 0.18 * (1.0 - t), 1.0 };
        }

        [[nodiscard]] ColorOKLaba ShadeSurfaceSample(
            const TrianglePlane& plane,
            SurfacePlaneRenderMode mode,
            const Vec2& surface_p) noexcept {

            return mode == SurfacePlaneRenderMode::PlaneShaded
                ? EvaluateTrianglePlane(plane, surface_p.x, surface_p.y, true)
                : TintFitError(plane);
        }

        void RasterizeTriangleWireframe(
            const Mesh& mesh,
            const Triangle& tri,
            ImageOKLaba& target,
            const SurfacePlaneRenderOptions& options,
            SurfacePlaneRenderStats& stats) {

            const Vec2 p0 = TransformSurfaceToTarget(TriangleP0(mesh, tri), options.transform);
            const Vec2 p1 = TransformSurfaceToTarget(TriangleP1(mesh, tri), options.transform);
            const Vec2 p2 = TransformSurfaceToTarget(TriangleP2(mesh, tri), options.transform);

            const f64 grow = Max(options.wire_half_width_px, 0.0) + 0.5;

            const f64 min_x_f = Min(p0.x, Min(p1.x, p2.x)) - grow;
            const f64 max_x_f = Max(p0.x, Max(p1.x, p2.x)) + grow;
            const f64 min_y_f = Min(p0.y, Min(p1.y, p2.y)) - grow;
            const f64 max_y_f = Max(p0.y, Max(p1.y, p2.y)) + grow;

            const i32 min_x = Clamp(static_cast<i32>(std::floor(min_x_f)), 0, target.Width() - 1);
            const i32 max_x = Clamp(static_cast<i32>(std::ceil(max_x_f) - 1), 0, target.Width() - 1);
            const i32 min_y = Clamp(static_cast<i32>(std::floor(min_y_f)), 0, target.Height() - 1);
            const i32 max_y = Clamp(static_cast<i32>(std::ceil(max_y_f) - 1), 0, target.Height() - 1);

            if (max_x < min_x || max_y < min_y) {
                return;
            }

            for (i32 y = min_y; y <= max_y; ++y) {
                for (i32 x = min_x; x <= max_x; ++x) {
                    const Vec2 p{ static_cast<f64>(x) + 0.5, static_cast<f64>(y) + 0.5 };
                    if (!PointInsideTriangle(p, p0, p1, p2)) {
                        continue;
                    }

                    const f64 d = Min(
                        DistancePointToSegment(p, p0, p1),
                        Min(DistancePointToSegment(p, p1, p2), DistancePointToSegment(p, p2, p0)));

                    if (d > options.wire_half_width_px) {
                        continue;
                    }

                    target.At(x, y) = options.wire_color;
                    ++stats.wire_pixels_shaded;
                }
            }
        }

        void RasterizeTriangleHighQuality(
            const Mesh& mesh,
            TriangleId triangle_id,
            const Triangle& tri,
            const std::vector<TrianglePlane>& planes,
            ImageOKLaba& target,
            std::vector<std::uint16_t>& sample_masks,
            const SurfacePlaneRenderOptions& options,
            SurfacePlaneRenderStats& stats) {

            const Vec2 p0 = TransformSurfaceToTarget(TriangleP0(mesh, tri), options.transform);
            const Vec2 p1 = TransformSurfaceToTarget(TriangleP1(mesh, tri), options.transform);
            const Vec2 p2 = TransformSurfaceToTarget(TriangleP2(mesh, tri), options.transform);

            const f64 min_x_f = Min(p0.x, Min(p1.x, p2.x));
            const f64 max_x_f = Max(p0.x, Max(p1.x, p2.x));
            const f64 min_y_f = Min(p0.y, Min(p1.y, p2.y));
            const f64 max_y_f = Max(p0.y, Max(p1.y, p2.y));

            const i32 min_x = Clamp(static_cast<i32>(std::floor(min_x_f)), 0, target.Width() - 1);
            const i32 max_x = Clamp(static_cast<i32>(std::ceil(max_x_f) - 1), 0, target.Width() - 1);
            const i32 min_y = Clamp(static_cast<i32>(std::floor(min_y_f)), 0, target.Height() - 1);
            const i32 max_y = Clamp(static_cast<i32>(std::ceil(max_y_f) - 1), 0, target.Height() - 1);

            if (max_x < min_x || max_y < min_y) {
                return;
            }

            bool rasterized = false;
            const std::size_t width = static_cast<std::size_t>(target.Width());
            const f64 sample_weight = 1.0 / static_cast<f64>(kMsaaSampleCount);

            for (i32 y = min_y; y <= max_y; ++y) {
                for (i32 x = min_x; x <= max_x; ++x) {
                    const std::size_t pixel_index = static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x);
                    std::uint16_t prev_mask = sample_masks[pixel_index];
                    if (prev_mask == kFullCoverageMask) {
                        continue;
                    }

                    if (PixelSquareFullyInsideTriangle(x, y, p0, p1, p2)) {
                        const Vec2 target_center{ static_cast<f64>(x) + 0.5, static_cast<f64>(y) + 0.5 };
                        const Vec2 surface_center = TransformTargetToSurface(target_center, options.transform);
                        if (prev_mask == 0u) {
                            ++stats.pixels_shaded;
                        }
                        target.At(x, y) = ShadeSurfaceSample(planes[triangle_id], options.mode, surface_center);
                        sample_masks[pixel_index] = kFullCoverageMask;
                        rasterized = true;
                        continue;
                    }

                    ColorOKLaba accum = MakeZeroColor();
                    std::uint16_t new_bits = 0u;

                    for (i32 sample_index = 0; sample_index < kMsaaSampleCount; ++sample_index) {
                        const std::uint16_t bit = static_cast<std::uint16_t>(1u << sample_index);
                        if ((prev_mask & bit) != 0u) {
                            continue;
                        }

                        const Vec2 target_p{
                            static_cast<f64>(x) + kMsaaPattern[static_cast<std::size_t>(sample_index)].x,
                            static_cast<f64>(y) + kMsaaPattern[static_cast<std::size_t>(sample_index)].y
                        };
                        if (!PointInsideTriangle(target_p, p0, p1, p2)) {
                            continue;
                        }

                        const Vec2 surface_p = TransformTargetToSurface(target_p, options.transform);
                        AddScaledColor(accum, ShadeSurfaceSample(planes[triangle_id], options.mode, surface_p), sample_weight);
                        new_bits = static_cast<std::uint16_t>(new_bits | bit);
                    }

                    if (new_bits == 0u) {
                        continue;
                    }

                    if (prev_mask == 0u) {
                        ++stats.pixels_shaded;
                    }

                    ColorOKLaba& dst = target.At(x, y);
                    dst.L += accum.L;
                    dst.a += accum.a;
                    dst.b += accum.b;
                    dst.alpha += accum.alpha;

                    sample_masks[pixel_index] = static_cast<std::uint16_t>(prev_mask | new_bits);
                    rasterized = true;
                }
            }

            if (rasterized) {
                ++stats.triangles_rasterized;
            }
        }

        void ResolveHighQualityImage(
            ImageOKLaba& target,
            const std::vector<std::uint16_t>& sample_masks,
            const ColorOKLaba& clear_color) {

            const std::size_t pixel_count = static_cast<std::size_t>(target.Width()) * static_cast<std::size_t>(target.Height());
            for (std::size_t i = 0; i < pixel_count; ++i) {
                const i32 x = static_cast<i32>(i % static_cast<std::size_t>(target.Width()));
                const i32 y = static_cast<i32>(i / static_cast<std::size_t>(target.Width()));

                const std::uint16_t mask = sample_masks[i];
                if (mask == 0u) {
                    target.At(x, y) = clear_color;
                    continue;
                }

                if (mask != kFullCoverageMask) {
                    const u32 covered = std::popcount(static_cast<unsigned>(mask));
                    if (covered > 0u) {
                        const f64 renorm = static_cast<f64>(kMsaaSampleCount) / static_cast<f64>(covered);
                        ScaleColorInPlace(target.At(x, y), renorm);
                    }
                }
            }
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

        ClearImage(out.image, MakeZeroColor());
        std::vector<std::uint16_t> sample_masks(
            static_cast<std::size_t>(output_size.width) * static_cast<std::size_t>(output_size.height),
            0u);

        for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
            const Triangle& tri = mesh.triangles[ti];
            if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
                ++out.stats.triangles_skipped_degenerate;
                continue;
            }

            RasterizeTriangleHighQuality(mesh, ti, tri, planes, out.image, sample_masks, options, out.stats);
        }

        ResolveHighQualityImage(out.image, sample_masks, options.clear_color);

        if (options.overlay_wireframe) {
            for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
                const Triangle& tri = mesh.triangles[ti];
                if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
                    continue;
                }
                RasterizeTriangleWireframe(mesh, tri, out.image, options, out.stats);
            }
        }

        return out;
    }

} // namespace svec
