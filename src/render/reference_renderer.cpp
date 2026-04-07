#include "svec/render/reference_renderer.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "svec/math/geometry.h"
#include "svec/math/scalar.h"
#include "svec/math/vec2.h"

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

        void RasterizeTriangleFillPointSampled(
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
            const i32 max_x = Clamp(static_cast<i32>(std::ceil(max_x_f) - 1), 0, target.Width() - 1);
            const i32 min_y = Clamp(static_cast<i32>(std::floor(min_y_f)), 0, target.Height() - 1);
            const i32 max_y = Clamp(static_cast<i32>(std::ceil(max_y_f) - 1), 0, target.Height() - 1);

            if (max_x < min_x || max_y < min_y) {
                return;
            }

            for (i32 y = min_y; y <= max_y; ++y) {
                for (i32 x = min_x; x <= max_x; ++x) {
                    const Vec2 p{ static_cast<f64>(x) + 0.5, static_cast<f64>(y) + 0.5 };
                    const auto bc = ComputeBarycentric(p, p0, p1, p2, 1e-12);
                    if (!bc.has_value() || !bc->IsInside(1e-9)) {
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

                    if (!IsPointNearTriangleEdge(p, p0, p1, p2, options.wire_half_width_px)) {
                        continue;
                    }

                    target.At(x, y) = options.wire_color;
                    ++stats.wire_pixels_shaded;
                }
            }
        }

        void RasterizeTriangleFillHighQuality(
            const Mesh& mesh,
            TriangleId triangle_id,
            const Triangle& tri,
            ImageOKLaba& target,
            std::vector<std::uint16_t>& sample_masks,
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
                        const Vec2 center{ static_cast<f64>(x) + 0.5, static_cast<f64>(y) + 0.5 };
                        const auto bc = ComputeBarycentric(center, p0, p1, p2, 1e-12);
                        if (!bc.has_value() || !bc->IsInside(1e-9)) {
                            continue;
                        }

                        if (prev_mask == 0u) {
                            ++stats.pixels_shaded;
                        }
                        target.At(x, y) = InterpolateTriangleColor(mesh, tri, *bc);
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

                        const Vec2 p{
                            static_cast<f64>(x) + kMsaaPattern[static_cast<std::size_t>(sample_index)].x,
                            static_cast<f64>(y) + kMsaaPattern[static_cast<std::size_t>(sample_index)].y
                        };
                        const auto bc = ComputeBarycentric(p, p0, p1, p2, 1e-12);
                        if (!bc.has_value() || !bc->IsInside(1e-9)) {
                            continue;
                        }

                        AddScaledColor(accum, InterpolateTriangleColor(mesh, tri, *bc), sample_weight);
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

        ReferenceRenderStats stats{};
        stats.triangles_total = static_cast<u32>(mesh.triangles.size());

        if (options.mode == ReferenceRenderMode::TriangleIdFlat) {
            ClearImage(target, options.clear_color);

            for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
                const Triangle& tri = mesh.triangles[ti];
                if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
                    ++stats.triangles_skipped_degenerate;
                    continue;
                }

                RasterizeTriangleFillPointSampled(mesh, ti, tri, target, options, stats);
                ++stats.triangles_rasterized;
            }
        }
        else {
            ClearImage(target, MakeZeroColor());
            std::vector<std::uint16_t> sample_masks(
                static_cast<std::size_t>(target.Width()) * static_cast<std::size_t>(target.Height()),
                0u);

            for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
                const Triangle& tri = mesh.triangles[ti];
                if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
                    ++stats.triangles_skipped_degenerate;
                    continue;
                }

                RasterizeTriangleFillHighQuality(mesh, ti, tri, target, sample_masks, options, stats);
            }

            ResolveHighQualityImage(target, sample_masks, options.clear_color);
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
