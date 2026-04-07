#include "svec/render/reference_renderer.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "svec/math/geometry.h"
#include "svec/math/scalar.h"
#include "svec/math/vec2.h"
#include "svec/render/triangle_raster_common.h"

namespace svec {
    namespace {

        struct ReferenceTriangleShadeSetup {
            detail::AffineColorField2D color_field{};
            ColorOKLaba flat_color{};
            bool use_affine_color = false;
        };

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

        [[nodiscard]] ReferenceTriangleShadeSetup BuildReferenceTriangleShadeSetup(
            const Mesh& mesh,
            TriangleId triangle_id,
            const Triangle& tri,
            const detail::TriangleRasterSetup& raster,
            ReferenceRenderMode mode) noexcept {

            ReferenceTriangleShadeSetup setup{};
            if (mode == ReferenceRenderMode::TriangleIdFlat) {
                setup.flat_color = TriangleIdDebugColor(triangle_id);
                return setup;
            }

            const ColorOKLaba& c0 = TriangleC0(mesh, tri);
            const ColorOKLaba& c1 = TriangleC1(mesh, tri);
            const ColorOKLaba& c2 = TriangleC2(mesh, tri);
            setup.color_field = detail::BuildAffineColorFieldThroughTriangle(
                raster.p0,
                c0,
                raster.p1,
                c1,
                raster.p2,
                c2);
            setup.use_affine_color = setup.color_field.valid;
            if (!setup.use_affine_color) {
                setup.flat_color = c0;
            }
            return setup;
        }

        [[nodiscard]] ColorOKLaba ShadeReferenceSample(
            const ReferenceTriangleShadeSetup& shade_setup,
            const Vec2& target_p) noexcept {

            if (shade_setup.use_affine_color) {
                return shade_setup.color_field.Evaluate(target_p);
            }
            return shade_setup.flat_color;
        }

        void RasterizeTriangleFillPointSampled(
            const detail::TriangleRasterSetup& raster,
            const ReferenceTriangleShadeSetup& shade_setup,
            ImageOKLaba& target,
            ReferenceRenderStats& stats) {

            const detail::TrianglePixelBounds bounds = detail::ComputeTrianglePixelBounds(
                raster,
                target.Width(),
                target.Height());

            if (bounds.IsEmpty()) {
                return;
            }

            for (i32 y = bounds.min_y; y <= bounds.max_y; ++y) {
                for (i32 x = bounds.min_x; x <= bounds.max_x; ++x) {
                    const Vec2 p{ static_cast<f64>(x) + 0.5, static_cast<f64>(y) + 0.5 };
                    if (!detail::PointInsideTriangle(p, raster)) {
                        continue;
                    }

                    target.At(x, y) = ShadeReferenceSample(shade_setup, p);
                    ++stats.pixels_shaded;
                }
            }
        }

        void RasterizeTriangleWireframe(
            const detail::TriangleRasterSetup& raster,
            ImageOKLaba& target,
            const ReferenceRenderOptions& options,
            ReferenceRenderStats& stats) {

            const f64 grow = Max(options.wire_half_width_px, 0.0) + 0.5;
            const detail::TrianglePixelBounds bounds = detail::ComputeTrianglePixelBounds(
                raster,
                target.Width(),
                target.Height(),
                grow);

            if (bounds.IsEmpty()) {
                return;
            }

            for (i32 y = bounds.min_y; y <= bounds.max_y; ++y) {
                for (i32 x = bounds.min_x; x <= bounds.max_x; ++x) {
                    const Vec2 p{ static_cast<f64>(x) + 0.5, static_cast<f64>(y) + 0.5 };
                    if (!detail::PointInsideTriangle(p, raster)) {
                        continue;
                    }

                    if (detail::DistancePointToTriangleEdges(p, raster) > options.wire_half_width_px) {
                        continue;
                    }

                    target.At(x, y) = options.wire_color;
                    ++stats.wire_pixels_shaded;
                }
            }
        }

        void RasterizeTriangleFillHighQuality(
            const detail::TriangleRasterSetup& raster,
            const ReferenceTriangleShadeSetup& shade_setup,
            ImageOKLaba& target,
            std::vector<std::uint16_t>& sample_masks,
            ReferenceRenderStats& stats) {

            const detail::TrianglePixelBounds bounds = detail::ComputeTrianglePixelBounds(
                raster,
                target.Width(),
                target.Height());

            if (bounds.IsEmpty()) {
                return;
            }

            bool rasterized = false;
            const std::size_t width = static_cast<std::size_t>(target.Width());
            const f64 sample_weight = 1.0 / static_cast<f64>(detail::kMsaaSampleCount);

            for (i32 y = bounds.min_y; y <= bounds.max_y; ++y) {
                for (i32 x = bounds.min_x; x <= bounds.max_x; ++x) {
                    const std::size_t pixel_index = static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x);
                    const std::uint16_t prev_mask = sample_masks[pixel_index];
                    if (prev_mask == detail::kFullCoverageMask) {
                        continue;
                    }

                    if (detail::PixelSquareFullyInsideTriangle(x, y, raster)) {
                        const Vec2 center{ static_cast<f64>(x) + 0.5, static_cast<f64>(y) + 0.5 };
                        if (prev_mask == 0u) {
                            ++stats.pixels_shaded;
                        }
                        target.At(x, y) = ShadeReferenceSample(shade_setup, center);
                        sample_masks[pixel_index] = detail::kFullCoverageMask;
                        rasterized = true;
                        continue;
                    }

                    ColorOKLaba accum = detail::MakeZeroColor();
                    std::uint16_t new_bits = 0u;

                    for (i32 sample_index = 0; sample_index < detail::kMsaaSampleCount; ++sample_index) {
                        const std::uint16_t bit = static_cast<std::uint16_t>(1u << sample_index);
                        if ((prev_mask & bit) != 0u) {
                            continue;
                        }

                        const Vec2 p{
                            static_cast<f64>(x) + detail::kMsaaPattern[static_cast<std::size_t>(sample_index)].x,
                            static_cast<f64>(y) + detail::kMsaaPattern[static_cast<std::size_t>(sample_index)].y
                        };
                        if (!detail::PointInsideTriangle(p, raster)) {
                            continue;
                        }

                        detail::AddScaledColor(accum, ShadeReferenceSample(shade_setup, p), sample_weight);
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

    } // namespace

    ReferenceRenderResult RenderMeshReference(
        const Mesh& mesh,
        const ImageSize& output_size,
        const ReferenceRenderOptions& options) {

        if (!output_size.IsValid()) {
            throw std::runtime_error("RenderMeshReference: invalid output_size.");
        }

        ReferenceRenderResult out{};
        out.image.Resize(output_size, options.clear_color);
        RenderMeshReferenceTo(mesh, out.image, options, &out.stats);
        return out;
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
            detail::ClearImage(target, options.clear_color);

            for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
                const Triangle& tri = mesh.triangles[ti];
                if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
                    ++stats.triangles_skipped_degenerate;
                    continue;
                }

                const detail::TriangleRasterSetup raster = detail::BuildTriangleRasterSetup(
                    TriangleP0(mesh, tri),
                    TriangleP1(mesh, tri),
                    TriangleP2(mesh, tri));
                if (!raster.valid) {
                    ++stats.triangles_skipped_degenerate;
                    continue;
                }

                const ReferenceTriangleShadeSetup shade_setup = BuildReferenceTriangleShadeSetup(
                    mesh,
                    ti,
                    tri,
                    raster,
                    options.mode);
                RasterizeTriangleFillPointSampled(raster, shade_setup, target, stats);
                ++stats.triangles_rasterized;
            }
        }
        else {
            detail::ClearImage(target, detail::MakeZeroColor());
            std::vector<std::uint16_t> sample_masks(
                static_cast<std::size_t>(target.Width()) * static_cast<std::size_t>(target.Height()),
                0u);

            for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
                const Triangle& tri = mesh.triangles[ti];
                if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
                    ++stats.triangles_skipped_degenerate;
                    continue;
                }

                const detail::TriangleRasterSetup raster = detail::BuildTriangleRasterSetup(
                    TriangleP0(mesh, tri),
                    TriangleP1(mesh, tri),
                    TriangleP2(mesh, tri));
                if (!raster.valid) {
                    ++stats.triangles_skipped_degenerate;
                    continue;
                }

                const ReferenceTriangleShadeSetup shade_setup = BuildReferenceTriangleShadeSetup(
                    mesh,
                    ti,
                    tri,
                    raster,
                    options.mode);
                RasterizeTriangleFillHighQuality(raster, shade_setup, target, sample_masks, stats);
            }

            detail::ResolveHighQualityImage(target, sample_masks, options.clear_color);
        }

        if (options.overlay_wireframe) {
            for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
                const Triangle& tri = mesh.triangles[ti];
                if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
                    continue;
                }

                const detail::TriangleRasterSetup raster = detail::BuildTriangleRasterSetup(
                    TriangleP0(mesh, tri),
                    TriangleP1(mesh, tri),
                    TriangleP2(mesh, tri));
                if (!raster.valid) {
                    continue;
                }
                RasterizeTriangleWireframe(raster, target, options, stats);
            }
        }

        if (out_stats) {
            *out_stats = stats;
        }
    }

} // namespace svec
