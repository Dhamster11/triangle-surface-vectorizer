#include "svec/render/surface_plane_renderer.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "svec/math/geometry.h"
#include "svec/math/scalar.h"
#include "svec/render/triangle_raster_common.h"

namespace svec {
    namespace {

        inline constexpr std::size_t kInvalidTriangleIndex = std::numeric_limits<std::size_t>::max();

        struct SurfaceTriangleShadeSetup {
            detail::AffineColorField2D color_field{};
            ColorOKLaba flat_color{};
            bool use_affine_color = false;
        };

        struct SurfaceEdgeAdjacency {
            std::size_t neighbor_triangle = kInvalidTriangleIndex;
            i32 neighbor_edge = -1;
            f64 blend_gate = 0.0;

            [[nodiscard]] bool HasNeighbor() const noexcept {
                return neighbor_triangle != kInvalidTriangleIndex && neighbor_edge >= 0;
            }

            [[nodiscard]] bool CanBlend() const noexcept {
                return HasNeighbor() && blend_gate > detail::kRasterEdgeEpsilon;
            }
        };

        struct SurfaceTriangleSetup {
            std::array<Vec2, 3> surface_points{};
            detail::TriangleRasterSetup raster{};
            SurfaceTriangleShadeSetup shade_setup{};
            std::array<SurfaceEdgeAdjacency, 3> adjacency{};
            bool active = false;
        };

        struct PointKey {
            std::uint64_t x_bits = 0;
            std::uint64_t y_bits = 0;

            [[nodiscard]] bool operator==(const PointKey& other) const noexcept = default;
        };

        struct EdgeKey {
            PointKey a{};
            PointKey b{};

            [[nodiscard]] bool operator==(const EdgeKey& other) const noexcept = default;
        };

        struct EdgeKeyHash {
            [[nodiscard]] std::size_t operator()(const EdgeKey& key) const noexcept {
                const auto mix = [](std::uint64_t v) noexcept {
                    v ^= v >> 33u;
                    v *= 0xff51afd7ed558ccdu;
                    v ^= v >> 33u;
                    v *= 0xc4ceb9fe1a85ec53u;
                    v ^= v >> 33u;
                    return v;
                };

                std::uint64_t h = mix(key.a.x_bits);
                h ^= mix(key.a.y_bits) + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
                h ^= mix(key.b.x_bits) + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
                h ^= mix(key.b.y_bits) + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
                return static_cast<std::size_t>(h);
            }
        };

        struct EdgeRecord {
            std::size_t triangle_index = kInvalidTriangleIndex;
            i32 edge_index = -1;

            [[nodiscard]] bool IsValid() const noexcept {
                return triangle_index != kInvalidTriangleIndex && edge_index >= 0;
            }
        };

        struct EdgeBucket {
            EdgeRecord first{};
            EdgeRecord second{};
            bool non_manifold = false;
        };

        struct EdgeBlendCandidate {
            std::size_t neighbor_triangle = kInvalidTriangleIndex;
            i32 edge_index = -1;
            f64 distance_px = std::numeric_limits<f64>::max();
            f64 blend_gate = 0.0;
            Vec2 projected_point{};

            [[nodiscard]] bool IsValid() const noexcept {
                return neighbor_triangle != kInvalidTriangleIndex &&
                    edge_index >= 0 &&
                    blend_gate > detail::kRasterEdgeEpsilon;
            }
        };

        [[nodiscard]] Vec2 TransformSurfaceToTarget(const Vec2& p, const SurfaceRenderTransform& transform) noexcept {
            return {
                p.x * transform.scale_x + transform.offset_x,
                p.y * transform.scale_y + transform.offset_y
            };
        }

        [[nodiscard]] ColorOKLaba TintFitError(const TrianglePlane& plane) noexcept {
            const f64 t = Saturate(plane.fit_rmse * 4.0);
            return {0.25 + 0.55 * t, 0.05 + 0.25 * t, 0.05 + 0.18 * (1.0 - t), 1.0};
        }

        [[nodiscard]] ColorOKLaba ShadeSurfaceSample(
            const TrianglePlane& plane,
            SurfacePlaneRenderMode mode,
            const Vec2& surface_p) noexcept {

            return mode == SurfacePlaneRenderMode::PlaneShaded
                ? EvaluateTrianglePlane(plane, surface_p.x, surface_p.y, true)
                : TintFitError(plane);
        }

        [[nodiscard]] SurfaceTriangleShadeSetup BuildSurfaceTriangleShadeSetup(
            const TrianglePlane& plane,
            SurfacePlaneRenderMode mode,
            const detail::TriangleRasterSetup& raster,
            const Vec2& surface_p0,
            const Vec2& surface_p1,
            const Vec2& surface_p2) noexcept {

            SurfaceTriangleShadeSetup setup{};
            if (mode != SurfacePlaneRenderMode::PlaneShaded) {
                setup.flat_color = TintFitError(plane);
                return setup;
            }

            const ColorOKLaba c0 = ShadeSurfaceSample(plane, mode, surface_p0);
            const ColorOKLaba c1 = ShadeSurfaceSample(plane, mode, surface_p1);
            const ColorOKLaba c2 = ShadeSurfaceSample(plane, mode, surface_p2);
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

        [[nodiscard]] ColorOKLaba ShadeSurfaceTargetSample(
            const SurfaceTriangleShadeSetup& shade_setup,
            const Vec2& target_p) noexcept {

            if (shade_setup.use_affine_color) {
                return shade_setup.color_field.Evaluate(target_p);
            }
            return shade_setup.flat_color;
        }

        [[nodiscard]] std::array<Vec2, 3> GetSurfaceTrianglePoints(const Mesh& mesh, const Triangle& tri) noexcept {
            return {TriangleP0(mesh, tri), TriangleP1(mesh, tri), TriangleP2(mesh, tri)};
        }

        [[nodiscard]] PointKey MakePointKey(const Vec2& p) noexcept {
            return {
                std::bit_cast<std::uint64_t>(p.x),
                std::bit_cast<std::uint64_t>(p.y)
            };
        }

        [[nodiscard]] bool PointKeyLess(const PointKey& lhs, const PointKey& rhs) noexcept {
            if (lhs.x_bits != rhs.x_bits) {
                return lhs.x_bits < rhs.x_bits;
            }
            return lhs.y_bits < rhs.y_bits;
        }

        [[nodiscard]] EdgeKey MakeEdgeKey(const Vec2& a, const Vec2& b) noexcept {
            const PointKey ka = MakePointKey(a);
            const PointKey kb = MakePointKey(b);
            return PointKeyLess(kb, ka) ? EdgeKey{kb, ka} : EdgeKey{ka, kb};
        }

        [[nodiscard]] f64 SmoothstepUnit(f64 t) noexcept {
            t = Saturate(t);
            return t * t * (3.0 - 2.0 * t);
        }

        [[nodiscard]] f64 ColorDiscontinuityMetric(const ColorOKLaba& lhs, const ColorOKLaba& rhs) noexcept {
            const f64 dL = lhs.L - rhs.L;
            const f64 da = lhs.a - rhs.a;
            const f64 db = lhs.b - rhs.b;
            const f64 dAlpha = lhs.alpha - rhs.alpha;
            return std::sqrt(dL * dL + da * da + db * db + 0.25 * dAlpha * dAlpha);
        }

        [[nodiscard]] f64 ComputeDiscontinuityBlendGate(
            f64 discontinuity,
            const SurfacePlaneRenderOptions& options) noexcept {

            if (!options.preserve_discontinuities) {
                return 1.0;
            }

            const f64 threshold = Max(options.discontinuity_threshold, 1e-6);
            if (discontinuity <= threshold) {
                return 1.0;
            }

            const f64 fade_end = threshold * 2.0;
            if (discontinuity >= fade_end) {
                return 0.0;
            }

            const f64 t = (discontinuity - threshold) / (fade_end - threshold);
            return 1.0 - SmoothstepUnit(t);
        }

        [[nodiscard]] const Vec2& RasterEdgePointA(const detail::TriangleRasterSetup& raster, i32 edge_index) noexcept {
            switch (edge_index) {
                case 0: return raster.p0;
                case 1: return raster.p1;
                default: return raster.p2;
            }
        }

        [[nodiscard]] const Vec2& RasterEdgePointB(const detail::TriangleRasterSetup& raster, i32 edge_index) noexcept {
            switch (edge_index) {
                case 0: return raster.p1;
                case 1: return raster.p2;
                default: return raster.p0;
            }
        }


        [[nodiscard]] Vec2 ProjectPointToSegment(const Vec2& p, const Vec2& a, const Vec2& b) noexcept {
            const Vec2 ab = b - a;
            const f64 ab_len2 = ab.LengthSquared();
            if (ab_len2 <= kEpsilon) {
                return a;
            }

            const f64 t = Clamp(Dot(p - a, ab) / ab_len2, 0.0, 1.0);
            return a + ab * t;
        }

        [[nodiscard]] ColorOKLaba AverageColor(const ColorOKLaba& lhs, const ColorOKLaba& rhs) noexcept {
            return {
                0.5 * (lhs.L + rhs.L),
                0.5 * (lhs.a + rhs.a),
                0.5 * (lhs.b + rhs.b),
                0.5 * (lhs.alpha + rhs.alpha)
            };
        }

        void BuildTriangleAdjacency(std::vector<SurfaceTriangleSetup>& triangle_setups) {
            std::unordered_map<EdgeKey, EdgeBucket, EdgeKeyHash> edge_buckets;
            edge_buckets.reserve(triangle_setups.size() * 3u);

            for (std::size_t ti = 0; ti < triangle_setups.size(); ++ti) {
                const SurfaceTriangleSetup& setup = triangle_setups[ti];
                if (!setup.active) {
                    continue;
                }

                for (i32 edge_index = 0; edge_index < 3; ++edge_index) {
                    const Vec2& a = setup.surface_points[static_cast<std::size_t>(edge_index)];
                    const Vec2& b = setup.surface_points[static_cast<std::size_t>((edge_index + 1) % 3)];
                    EdgeBucket& bucket = edge_buckets[MakeEdgeKey(a, b)];
                    const EdgeRecord record{ti, edge_index};

                    if (!bucket.first.IsValid()) {
                        bucket.first = record;
                    } else if (!bucket.second.IsValid()) {
                        bucket.second = record;
                    } else {
                        bucket.non_manifold = true;
                    }
                }
            }

            for (const auto& [key, bucket] : edge_buckets) {
                (void)key;
                if (bucket.non_manifold || !bucket.first.IsValid() || !bucket.second.IsValid()) {
                    continue;
                }
                if (bucket.first.triangle_index == bucket.second.triangle_index) {
                    continue;
                }

                triangle_setups[bucket.first.triangle_index]
                    .adjacency[static_cast<std::size_t>(bucket.first.edge_index)] = {
                        bucket.second.triangle_index,
                        bucket.second.edge_index,
                        0.0
                    };
                triangle_setups[bucket.second.triangle_index]
                    .adjacency[static_cast<std::size_t>(bucket.second.edge_index)] = {
                        bucket.first.triangle_index,
                        bucket.first.edge_index,
                        0.0
                    };
            }
        }

        void FinalizeAdjacencyBlendGates(
            std::vector<SurfaceTriangleSetup>& triangle_setups,
            const SurfacePlaneRenderOptions& options) {

            if (!options.smooth_internal_edges ||
                options.mode != SurfacePlaneRenderMode::PlaneShaded ||
                options.internal_edge_blend_radius_px <= detail::kRasterEdgeEpsilon) {
                return;
            }

            for (SurfaceTriangleSetup& setup : triangle_setups) {
                if (!setup.active) {
                    continue;
                }

                for (i32 edge_index = 0; edge_index < 3; ++edge_index) {
                    SurfaceEdgeAdjacency& adjacency = setup.adjacency[static_cast<std::size_t>(edge_index)];
                    if (!adjacency.HasNeighbor()) {
                        continue;
                    }

                    const SurfaceTriangleSetup& neighbor = triangle_setups[adjacency.neighbor_triangle];
                    if (!neighbor.active) {
                        continue;
                    }

                    const Vec2 edge_mid = (RasterEdgePointA(setup.raster, edge_index) +
                                           RasterEdgePointB(setup.raster, edge_index)) * 0.5;
                    const ColorOKLaba self_color = ShadeSurfaceTargetSample(setup.shade_setup, edge_mid);
                    const ColorOKLaba neighbor_color = ShadeSurfaceTargetSample(neighbor.shade_setup, edge_mid);
                    adjacency.blend_gate = ComputeDiscontinuityBlendGate(
                        ColorDiscontinuityMetric(self_color, neighbor_color),
                        options);
                }
            }
        }

        [[nodiscard]] EdgeBlendCandidate FindInternalEdgeBlendCandidate(
            const SurfaceTriangleSetup& triangle_setup,
            const Vec2& target_p,
            const SurfacePlaneRenderOptions& options) noexcept {

            EdgeBlendCandidate best{};
            const f64 radius_px = Max(options.internal_edge_blend_radius_px, 0.0);
            if (!options.smooth_internal_edges ||
                options.mode != SurfacePlaneRenderMode::PlaneShaded ||
                radius_px <= detail::kRasterEdgeEpsilon) {
                return best;
            }

            for (i32 edge_index = 0; edge_index < 3; ++edge_index) {
                const SurfaceEdgeAdjacency& adjacency = triangle_setup.adjacency[static_cast<std::size_t>(edge_index)];
                if (!adjacency.CanBlend()) {
                    continue;
                }

                const Vec2 edge_a = RasterEdgePointA(triangle_setup.raster, edge_index);
                const Vec2 edge_b = RasterEdgePointB(triangle_setup.raster, edge_index);
                const Vec2 projected_point = ProjectPointToSegment(target_p, edge_a, edge_b);
                const f64 distance_px = Distance(target_p, projected_point);
                if (distance_px >= radius_px || distance_px >= best.distance_px) {
                    continue;
                }

                best.neighbor_triangle = adjacency.neighbor_triangle;
                best.edge_index = edge_index;
                best.distance_px = distance_px;
                best.blend_gate = adjacency.blend_gate;
                best.projected_point = projected_point;
            }

            return best;
        }

        [[nodiscard]] ColorOKLaba BlendSurfaceTargetSampleIfNeeded(
            const SurfaceTriangleSetup& triangle_setup,
            const std::vector<SurfaceTriangleSetup>& triangle_setups,
            const SurfacePlaneRenderOptions& options,
            const Vec2& target_p,
            bool* out_blended = nullptr) noexcept {

            if (out_blended != nullptr) {
                *out_blended = false;
            }

            const ColorOKLaba self_color = ShadeSurfaceTargetSample(triangle_setup.shade_setup, target_p);
            const EdgeBlendCandidate candidate = FindInternalEdgeBlendCandidate(triangle_setup, target_p, options);
            if (!candidate.IsValid()) {
                return self_color;
            }

            const SurfaceTriangleSetup& neighbor_setup = triangle_setups[candidate.neighbor_triangle];
            if (!neighbor_setup.active) {
                return self_color;
            }

            const Vec2 edge_point = candidate.projected_point;
            const ColorOKLaba self_edge_color = ShadeSurfaceTargetSample(triangle_setup.shade_setup, edge_point);
            const ColorOKLaba neighbor_edge_color = ShadeSurfaceTargetSample(neighbor_setup.shade_setup, edge_point);

            const f64 local_blend_gate = ComputeDiscontinuityBlendGate(
                ColorDiscontinuityMetric(self_edge_color, neighbor_edge_color),
                options);
            const f64 combined_gate = candidate.blend_gate * local_blend_gate;
            if (combined_gate <= detail::kRasterEdgeEpsilon) {
                return self_color;
            }

            const f64 radius_px = Max(options.internal_edge_blend_radius_px, detail::kRasterEdgeEpsilon);
            const f64 proximity = 1.0 - candidate.distance_px / radius_px;
            const f64 edge_falloff = SmoothstepUnit(proximity);
            const f64 stitch_weight = Clamp(
                edge_falloff * combined_gate * Max(options.internal_edge_blend_strength, 0.0),
                0.0,
                1.0);
            if (stitch_weight <= detail::kRasterEdgeEpsilon) {
                return self_color;
            }

            const ColorOKLaba consensus_edge = AverageColor(self_edge_color, neighbor_edge_color);

            if (out_blended != nullptr) {
                *out_blended = true;
            }

            return {
                self_color.L + (consensus_edge.L - self_edge_color.L) * stitch_weight,
                self_color.a + (consensus_edge.a - self_edge_color.a) * stitch_weight,
                self_color.b + (consensus_edge.b - self_edge_color.b) * stitch_weight,
                self_color.alpha + (consensus_edge.alpha - self_edge_color.alpha) * stitch_weight
            };
        }

        void RasterizeTriangleWireframe(
            const detail::TriangleRasterSetup& raster,
            ImageOKLaba& target,
            const SurfacePlaneRenderOptions& options,
            SurfacePlaneRenderStats& stats) {

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
                    const Vec2 p{static_cast<f64>(x) + 0.5, static_cast<f64>(y) + 0.5};
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

        void RasterizeTriangleHighQuality(
            const SurfaceTriangleSetup& triangle_setup,
            const std::vector<SurfaceTriangleSetup>& triangle_setups,
            ImageOKLaba& target,
            std::vector<std::uint16_t>& sample_masks,
            const SurfacePlaneRenderOptions& options,
            SurfacePlaneRenderStats& stats) {

            const detail::TriangleRasterSetup& raster = triangle_setup.raster;
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
                        const Vec2 center{static_cast<f64>(x) + 0.5, static_cast<f64>(y) + 0.5};
                        bool blended = false;
                        const ColorOKLaba center_color = BlendSurfaceTargetSampleIfNeeded(
                            triangle_setup,
                            triangle_setups,
                            options,
                            center,
                            &blended);
                        if (prev_mask == 0u) {
                            ++stats.pixels_shaded;
                        }
                        target.At(x, y) = center_color;
                        sample_masks[pixel_index] = detail::kFullCoverageMask;
                        if (blended) {
                            stats.internal_edge_samples_blended += static_cast<u64>(detail::kMsaaSampleCount);
                        }
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

                        const Vec2 target_p{
                            static_cast<f64>(x) + detail::kMsaaPattern[static_cast<std::size_t>(sample_index)].x,
                            static_cast<f64>(y) + detail::kMsaaPattern[static_cast<std::size_t>(sample_index)].y
                        };
                        if (!detail::PointInsideTriangle(target_p, raster)) {
                            continue;
                        }

                        bool blended = false;
                        const ColorOKLaba sample_color = BlendSurfaceTargetSampleIfNeeded(
                            triangle_setup,
                            triangle_setups,
                            options,
                            target_p,
                            &blended);
                        detail::AddScaledColor(accum, sample_color, sample_weight);
                        if (blended) {
                            ++stats.internal_edge_samples_blended;
                        }
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

        detail::ClearImage(out.image, detail::MakeZeroColor());
        std::vector<std::uint16_t> sample_masks(
            static_cast<std::size_t>(output_size.width) * static_cast<std::size_t>(output_size.height),
            0u);

        std::vector<SurfaceTriangleSetup> triangle_setups(mesh.triangles.size());
        for (std::size_t ti = 0; ti < mesh.triangles.size(); ++ti) {
            const Triangle& tri = mesh.triangles[ti];
            if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
                ++out.stats.triangles_skipped_degenerate;
                continue;
            }

            SurfaceTriangleSetup setup{};
            setup.surface_points = GetSurfaceTrianglePoints(mesh, tri);
            setup.raster = detail::BuildTriangleRasterSetup(
                TransformSurfaceToTarget(setup.surface_points[0], options.transform),
                TransformSurfaceToTarget(setup.surface_points[1], options.transform),
                TransformSurfaceToTarget(setup.surface_points[2], options.transform));
            if (!setup.raster.valid) {
                ++out.stats.triangles_skipped_degenerate;
                continue;
            }

            setup.shade_setup = BuildSurfaceTriangleShadeSetup(
                planes[ti],
                options.mode,
                setup.raster,
                setup.surface_points[0],
                setup.surface_points[1],
                setup.surface_points[2]);
            setup.active = true;
            triangle_setups[ti] = setup;
        }

        BuildTriangleAdjacency(triangle_setups);
        FinalizeAdjacencyBlendGates(triangle_setups, options);

        for (const SurfaceTriangleSetup& setup : triangle_setups) {
            if (!setup.active) {
                continue;
            }
            RasterizeTriangleHighQuality(setup, triangle_setups, out.image, sample_masks, options, out.stats);
        }

        detail::ResolveHighQualityImage(out.image, sample_masks, options.clear_color);

        if (options.overlay_wireframe) {
            for (const SurfaceTriangleSetup& setup : triangle_setups) {
                if (!setup.active) {
                    continue;
                }
                RasterizeTriangleWireframe(setup.raster, out.image, options, out.stats);
            }
        }

        return out;
    }

} // namespace svec
