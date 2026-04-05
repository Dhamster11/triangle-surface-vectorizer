#include "svec/refine/adaptive_refinement.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

#include "svec/math/color.h"
#include "svec/math/geometry.h"
#include "svec/math/scalar.h"
#include "svec/surface/mesh_topology.h"

namespace svec {
    namespace {

        struct EdgeSelection {
            i32 local_opposite_index = -1;
            VertexId edge_a = kInvalidIndex;
            VertexId edge_b = kInvalidIndex;
            VertexId opposite = kInvalidIndex;
            f64 edge_length = 0.0;
        };

        struct SplitPointCandidate {
            Vec2 position{};
            f64 t = 0.5;
            f64 score = -1.0;
            bool valid = false;
        };

        [[nodiscard]] f64 ColorResidualProxy(const ColorOKLaba& ref, const ColorOKLaba& fit) noexcept {
            const f64 dL = ref.L - fit.L;
            const f64 da = ref.a - fit.a;
            const f64 db = ref.b - fit.b;
            return std::sqrt(dL * dL + 0.35 * (da * da + db * db));
        }

        [[nodiscard]] ColorOKLaba SampleImageOKLabaNearestImpl(const ImageOKLaba& image, const Vec2& p) {
            if (!image.IsValid()) {
                throw std::runtime_error("SampleImageOKLabaNearest: image is invalid.");
            }

            const i32 x = Clamp(static_cast<i32>(std::llround(p.x)), 0, image.Width() - 1);
            const i32 y = Clamp(static_cast<i32>(std::llround(p.y)), 0, image.Height() - 1);
            return image.At(x, y);
        }

        [[nodiscard]] Triangle MakeTrianglePreservingOrientation(
            const Mesh& mesh,
            const Triangle& reference_tri,
            VertexId a,
            VertexId b,
            VertexId c) {

            Triangle out{ {a, b, c} };
            const f64 ref_sign = TriangleAreaSigned2(
                TriangleP0(mesh, reference_tri),
                TriangleP1(mesh, reference_tri),
                TriangleP2(mesh, reference_tri));

            const Vec2& pa = mesh.vertices.at(a).position;
            const Vec2& pb = mesh.vertices.at(b).position;
            const Vec2& pc = mesh.vertices.at(c).position;
            const f64 new_sign = TriangleAreaSigned2(pa, pb, pc);

            if ((ref_sign > 0.0 && new_sign < 0.0) || (ref_sign < 0.0 && new_sign > 0.0)) {
                out.v = { a, c, b };
            }
            return out;
        }

        [[nodiscard]] f64 SafeNormalizedDotSquared(const Vec2& a, const Vec2& b) noexcept {
            const f64 la2 = a.LengthSquared();
            const f64 lb2 = b.LengthSquared();
            if (la2 <= kEpsilon || lb2 <= kEpsilon) {
                return 0.0;
            }
            const f64 dot = Dot(a, b) / std::sqrt(la2 * lb2);
            return dot * dot;
        }

        [[nodiscard]] f64 BaseStructureGate(
            const StructureTensorSample& sample,
            const SingleRefineStepOptions& options) noexcept {

            if (sample.coherence <= options.tensor_min_coherence ||
                sample.strength <= options.tensor_min_strength) {
                return 0.0;
            }

            const f64 coh_den = Max(1.0 - options.tensor_min_coherence, 1e-6);
            const f64 coherence_gate = Saturate((sample.coherence - options.tensor_min_coherence) / coh_den);
            const f64 strength_gate = sample.strength /
                (sample.strength + Max(options.tensor_strength_softness, 1e-6));
            return coherence_gate * strength_gate;
        }

        [[nodiscard]] f64 TensorScaleGate(
            f64 span_px,
            const SingleRefineStepOptions& options) noexcept {

            const f64 min_scale = Max(options.tensor_min_triangle_scale_px, 0.0);
            const f64 full_scale = Max(options.tensor_full_triangle_scale_px, min_scale + 1e-6);
            return Saturate((span_px - min_scale) / (full_scale - min_scale));
        }

        [[nodiscard]] f64 StructureGate(
            const StructureTensorSample& sample,
            f64 span_px,
            const SingleRefineStepOptions& options) noexcept {

            return BaseStructureGate(sample, options) * TensorScaleGate(span_px, options);
        }

        [[nodiscard]] const ErrorPyramidLevel* SelectGuidanceLevel(
            const ErrorPyramid* guidance_pyramid,
            f64 span,
            const SingleRefineStepOptions& options) noexcept {

            if (guidance_pyramid == nullptr || !guidance_pyramid->IsValid()) {
                return nullptr;
            }

            const f64 target_span = Max(options.tensor_target_span_px, 1.0);
            const f64 desired_scale = Clamp(
                span / target_span,
                1.0,
                Max(guidance_pyramid->levels.back().scale, 1.0));
            const f64 desired_log = std::log(desired_scale);

            const ErrorPyramidLevel* best = &guidance_pyramid->levels.front();
            f64 best_dist = std::abs(std::log(Max(best->scale, 1.0)) - desired_log);
            for (const ErrorPyramidLevel& level : guidance_pyramid->levels) {
                if (!level.structure_tensor.IsValid()) {
                    continue;
                }
                const f64 dist = std::abs(std::log(Max(level.scale, 1.0)) - desired_log);
                if (dist < best_dist) {
                    best = &level;
                    best_dist = dist;
                }
            }
            return best;
        }

        [[nodiscard]] StructureTensorSample SampleTensorGuidance(
            const ErrorPyramid* guidance_pyramid,
            const Vec2& image_space_point,
            f64 span,
            const SingleRefineStepOptions& options) {

            const ErrorPyramidLevel* level = SelectGuidanceLevel(guidance_pyramid, span, options);
            if (level == nullptr || !level->structure_tensor.IsValid()) {
                return {};
            }
            return SampleStructureTensorBilinear(*level, image_space_point);
        }

        [[nodiscard]] f64 AnisotropicDirectionPenalty(
            const Vec2& direction,
            const StructureTensorSample& sample,
            const SingleRefineStepOptions& options) noexcept {

            const f64 span_px = std::sqrt(Max(direction.LengthSquared(), 0.0));
            const f64 gate = StructureGate(sample, span_px, options);
            if (gate <= 0.0 || span_px <= kEpsilon) {
                return 0.0;
            }

            const f64 normal_component2 = SafeNormalizedDotSquared(direction, sample.normal);
            const f64 extra_anisotropy = Max(options.tensor_max_anisotropy - 1.0, 0.0) * gate;
            return extra_anisotropy * normal_component2;
        }

        [[nodiscard]] f64 ComputeSeamStructurePenalty(
            const ErrorPyramid* guidance_pyramid,
            const Vec2& seam_a,
            const Vec2& seam_b,
            f64 guidance_span,
            const SingleRefineStepOptions& options) {

            if (guidance_pyramid == nullptr || !guidance_pyramid->IsValid()) {
                return 0.0;
            }

            const Vec2 seam_dir = seam_b - seam_a;
            const f64 seam_len2 = seam_dir.LengthSquared();
            if (seam_len2 <= kEpsilon) {
                return 0.0;
            }

            const u32 sample_count = Max<u32>(options.tensor_seam_sample_count, 1u);
            const f64 seam_len = std::sqrt(seam_len2);
            const f64 small_scale_gate = 1.0 - TensorScaleGate(seam_len, options);
            if (small_scale_gate <= 0.0) {
                return 0.0;
            }

            f64 accum = 0.0;
            for (u32 i = 0; i < sample_count; ++i) {
                const f64 t = static_cast<f64>(i + 1u) / static_cast<f64>(sample_count + 1u);
                const Vec2 p = Lerp(seam_a, seam_b, t);
                const StructureTensorSample tensor = SampleTensorGuidance(guidance_pyramid, p, guidance_span, options);
                const f64 structure_gate = BaseStructureGate(tensor, options);
                if (structure_gate <= 0.0) {
                    continue;
                }

                const f64 tangent_alignment2 = SafeNormalizedDotSquared(seam_dir, tensor.tangent);
                const f64 tangent_bias = 0.35 + 0.65 * tangent_alignment2;
                accum += structure_gate * tangent_bias;
            }

            return small_scale_gate * (accum / static_cast<f64>(sample_count));
        }

        [[nodiscard]] EdgeSelection SelectSplitEdge(
            const Mesh& mesh,
            TriangleId triangle_id,
            const ErrorPyramid* guidance_pyramid,
            const SingleRefineStepOptions& options) {

            const Triangle& tri = mesh.triangles.at(triangle_id);
            const Vec2& p0 = TriangleP0(mesh, tri);
            const Vec2& p1 = TriangleP1(mesh, tri);
            const Vec2& p2 = TriangleP2(mesh, tri);

            EdgeSelection best{};
            f64 best_score = -1.0;
            const f64 tri_span = Max(Distance(p0, p1), Max(Distance(p1, p2), Distance(p2, p0)));

            for (i32 local_opp = 0; local_opp < 3; ++local_opp) {
                const VertexId opposite = tri.v[static_cast<std::size_t>(local_opp)];
                const VertexId ea = tri.v[static_cast<std::size_t>((local_opp + 1) % 3)];
                const VertexId eb = tri.v[static_cast<std::size_t>((local_opp + 2) % 3)];
                const Vec2& pa = mesh.vertices.at(ea).position;
                const Vec2& pb = mesh.vertices.at(eb).position;
                const Vec2& pop = mesh.vertices.at(opposite).position;
                const Vec2 edge_vec = pb - pa;
                const f64 edge_len = edge_vec.Length();
                if (edge_len <= 0.0) {
                    continue;
                }

                const Vec2 midpoint = Midpoint(pa, pb);
                const StructureTensorSample tensor = SampleTensorGuidance(
                    guidance_pyramid,
                    midpoint,
                    tri_span,
                    options);

                f64 penalty = AnisotropicDirectionPenalty(midpoint - pop, tensor, options);
                if (options.split_shared_neighbor && mesh.HasTopology()) {
                    const TriangleId neighbor_id = mesh.topology.at(triangle_id).neighbors.at(static_cast<std::size_t>(local_opp));
                    if (neighbor_id != kInvalidIndex && mesh.IsValidTriangleId(neighbor_id)) {
                        const Triangle& nbr = mesh.triangles.at(neighbor_id);
                        const i32 nbr_opp = FindLocalOppositeVertexIndex(nbr, ea, eb);
                        if (nbr_opp >= 0) {
                            const Vec2& neighbor_op = mesh.vertices.at(nbr.v[static_cast<std::size_t>(nbr_opp)]).position;
                            const f64 neighbor_penalty = AnisotropicDirectionPenalty(midpoint - neighbor_op, tensor, options);
                            const f64 neighbor_w = Max(options.tensor_neighbor_consistency_weight, 0.0);
                            penalty = (penalty + neighbor_w * neighbor_penalty) / (1.0 + neighbor_w);
                        }
                    }
                }

                const f64 score = edge_len / (1.0 + options.tensor_edge_metric_weight * penalty);
                if (score > best_score) {
                    best_score = score;
                    best.local_opposite_index = local_opp;
                    best.opposite = opposite;
                    best.edge_a = ea;
                    best.edge_b = eb;
                    best.edge_length = edge_len;
                }
            }

            if (best.local_opposite_index < 0) {
                const i32 longest = LongestEdgeIndex(p0, p1, p2);
                best.local_opposite_index = longest;
                best.opposite = tri.v[static_cast<std::size_t>(longest)];
                best.edge_a = tri.v[static_cast<std::size_t>((longest + 1) % 3)];
                best.edge_b = tri.v[static_cast<std::size_t>((longest + 2) % 3)];
                best.edge_length = Distance(mesh.vertices.at(best.edge_a).position, mesh.vertices.at(best.edge_b).position);
            }
            return best;
        }

        [[nodiscard]] ColorOKLaba ComputeInsertedVertexColor(
            const Mesh& mesh,
            VertexId a,
            VertexId b,
            const ImageOKLaba& reference,
            const Vec2& split_point,
            f64 edge_t,
            NewVertexColorMode mode) {

            switch (mode) {
            case NewVertexColorMode::MidpointOfEndpoints:
                return Lerp(mesh.vertices.at(a).color, mesh.vertices.at(b).color, edge_t);
            case NewVertexColorMode::SampleReferenceBilinear:
            default:
                return SampleImageOKLabaBilinear(reference, split_point);
            }
        }

        [[nodiscard]] bool TriangleBBoxExtentIsEnough(const Vec2& a, const Vec2& b, const Vec2& c, f64 min_extent) noexcept {
            if (min_extent <= 0.0) {
                return true;
            }
            const f64 dx = Max(a.x, Max(b.x, c.x)) - Min(a.x, Min(b.x, c.x));
            const f64 dy = Max(a.y, Max(b.y, c.y)) - Min(a.y, Min(b.y, c.y));
            return dx >= min_extent || dy >= min_extent;
        }

        [[nodiscard]] bool TriangleCandidateIsValid(const Vec2& a, const Vec2& b, const Vec2& c, const SingleRefineStepOptions& options) noexcept {
            if (IsDegenerateTriangle(a, b, c, options.min_triangle_area * 2.0)) {
                return false;
            }
            if (TriangleArea(a, b, c) <= options.min_triangle_area) {
                return false;
            }
            if (!TriangleBBoxExtentIsEnough(a, b, c, options.min_triangle_bbox_extent)) {
                return false;
            }
            return true;
        }

        [[nodiscard]] bool SplitPointIsSafe(const Vec2& split_point, const Vec2& a, const Vec2& b, const SingleRefineStepOptions& options) noexcept {
            const f64 min_sep = Max(options.min_midpoint_separation, options.min_edge_length * 1e-3);
            return Distance(split_point, a) > min_sep && Distance(split_point, b) > min_sep;
        }

        [[nodiscard]] f64 ComputeCrossEdgeContrast(
            const ImageOKLaba& reference,
            const Vec2& p,
            const Vec2& edge_dir,
            f64 probe_radius_px) {

            const f64 probe = Max(probe_radius_px, 0.5);
            const Vec2 normal{ -edge_dir.y, edge_dir.x };
            const ColorOKLaba c_plus = SampleImageOKLabaBilinear(reference, p + normal * probe);
            const ColorOKLaba c_minus = SampleImageOKLabaBilinear(reference, p - normal * probe);
            return ColorResidualProxy(c_plus, c_minus);
        }

        [[nodiscard]] f64 ComputeSplitPointProxyScore(
            const Mesh& mesh,
            VertexId edge_a,
            VertexId edge_b,
            const ImageOKLaba& reference,
            const ErrorPyramid* guidance_pyramid,
            const Vec2& split_point,
            f64 t,
            const Vec2& seed_opposite,
            const Vec2* neighbor_opposite,
            const SingleRefineStepOptions& options) {

            const ColorOKLaba ref = SampleImageOKLabaBilinear(reference, split_point);
            const ColorOKLaba fit = Lerp(mesh.vertices.at(edge_a).color, mesh.vertices.at(edge_b).color, t);
            const f64 residual = ColorResidualProxy(ref, fit);

            const Vec2 pa = mesh.vertices.at(edge_a).position;
            const Vec2 pb = mesh.vertices.at(edge_b).position;
            const Vec2 edge_vec = pb - pa;
            const f64 edge_len = edge_vec.Length();
            const Vec2 edge_dir = edge_len > 0.0 ? (edge_vec / edge_len) : Vec2{ 1.0, 0.0 };

            const f64 cross_edge = ComputeCrossEdgeContrast(reference, split_point, edge_dir, options.split_search_probe_radius_px);
            const StructureTensorSample tensor = SampleTensorGuidance(guidance_pyramid, split_point, edge_len, options);
            f64 metric_penalty = AnisotropicDirectionPenalty(split_point - seed_opposite, tensor, options);
            f64 seam_penalty = ComputeSeamStructurePenalty(
                guidance_pyramid,
                seed_opposite,
                split_point,
                Max(edge_len, Distance(split_point, seed_opposite)),
                options);
            if (neighbor_opposite != nullptr) {
                const f64 neighbor_penalty = AnisotropicDirectionPenalty(split_point - *neighbor_opposite, tensor, options);
                const f64 neighbor_seam_penalty = ComputeSeamStructurePenalty(
                    guidance_pyramid,
                    *neighbor_opposite,
                    split_point,
                    Max(edge_len, Distance(split_point, *neighbor_opposite)),
                    options);
                const f64 neighbor_w = Max(options.tensor_neighbor_consistency_weight, 0.0);
                metric_penalty = (metric_penalty + neighbor_w * neighbor_penalty) / (1.0 + neighbor_w);
                seam_penalty = (seam_penalty + neighbor_w * neighbor_seam_penalty) / (1.0 + neighbor_w);
            }
            const f64 center_penalty = options.split_search_center_penalty * std::abs(t - 0.5);

            return
                options.split_search_residual_weight * residual +
                options.split_search_cross_edge_weight * cross_edge -
                options.tensor_split_metric_weight * metric_penalty -
                options.tensor_seam_penalty_weight * seam_penalty -
                center_penalty;
        }

        [[nodiscard]] SplitPointCandidate FindBestSplitPointOnEdge(
            const Mesh& mesh,
            TriangleId neighbor_id,
            const EdgeSelection& edge,
            const ImageOKLaba& reference,
            const ErrorPyramid* guidance_pyramid,
            const SingleRefineStepOptions& options) {

            const Vec2& pa = mesh.vertices.at(edge.edge_a).position;
            const Vec2& pb = mesh.vertices.at(edge.edge_b).position;
            const Vec2& seed_op = mesh.vertices.at(edge.opposite).position;

            const f64 min_t = Clamp(options.split_search_min_t, 0.05, 0.49);
            const f64 max_t = Clamp(options.split_search_max_t, 0.51, 0.95);
            const u32 candidate_count = Max<u32>(3u, options.split_search_candidate_count);

            std::vector<f64> ts;
            ts.reserve(candidate_count);
            if (candidate_count == 1u) {
                ts.push_back(0.5);
            } else {
                const f64 denom = static_cast<f64>(candidate_count - 1u);
                for (u32 i = 0; i < candidate_count; ++i) {
                    const f64 alpha = static_cast<f64>(i) / denom;
                    ts.push_back(Lerp(min_t, max_t, alpha));
                }
            }

            std::vector<SplitPointCandidate> candidates(ts.size());

            const Vec2* neighbor_op_ptr = nullptr;
            Vec2 neighbor_op{};
            if (neighbor_id != kInvalidIndex) {
                const Triangle& nbr = mesh.triangles.at(neighbor_id);
                const i32 nbr_opp = FindLocalOppositeVertexIndex(nbr, edge.edge_a, edge.edge_b);
                if (nbr_opp < 0) {
                    return {};
                }
                neighbor_op = mesh.vertices.at(nbr.v[static_cast<std::size_t>(nbr_opp)]).position;
                neighbor_op_ptr = &neighbor_op;
            }

            for (std::size_t i = 0; i < ts.size(); ++i) {
                const f64 t = ts[i];
                const Vec2 split_point = Lerp(pa, pb, t);
                if (!SplitPointIsSafe(split_point, pa, pb, options)) {
                    continue;
                }

                candidates[i].valid = true;
                candidates[i].position = split_point;
                candidates[i].t = t;
                candidates[i].score = ComputeSplitPointProxyScore(
                    mesh,
                    edge.edge_a,
                    edge.edge_b,
                    reference,
                    guidance_pyramid,
                    split_point,
                    t,
                    seed_op,
                    neighbor_op_ptr,
                    options);
            }

            std::sort(candidates.begin(), candidates.end(), [&](const SplitPointCandidate& a, const SplitPointCandidate& b) {
                return a.score > b.score;
            });

            for (const SplitPointCandidate& candidate : candidates) {
                if (!candidate.valid) {
                    continue;
                }
                return candidate;
            }

            if (SplitPointIsSafe(Midpoint(pa, pb), pa, pb, options)) {
                SplitPointCandidate fallback{};
                fallback.valid = true;
                fallback.position = Midpoint(pa, pb);
                fallback.t = 0.5;
                fallback.score = ComputeSplitPointProxyScore(
                    mesh,
                    edge.edge_a,
                    edge.edge_b,
                    reference,
                    guidance_pyramid,
                    fallback.position,
                    fallback.t,
                    seed_op,
                    neighbor_op_ptr,
                    options);
                return fallback;
            }

            return {};
        }

        void PushUniqueTriangleId(std::vector<TriangleId>& ids, TriangleId id) {
            if (id == kInvalidIndex) {
                return;
            }
            if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
                ids.push_back(id);
            }
        }

        void SplitTriangleAcrossEdge(
            Mesh& mesh,
            TriangleId triangle_id,
            VertexId edge_a,
            VertexId edge_b,
            VertexId midpoint_vertex,
            const SingleRefineStepOptions& options,
            u32& inout_triangles_added,
            std::vector<TriangleId>& touched_ids) {

            Triangle& tri = mesh.triangles.at(triangle_id);
            const i32 local_opp = FindLocalOppositeVertexIndex(tri, edge_a, edge_b);
            if (local_opp < 0) {
                throw std::runtime_error("SplitTriangleAcrossEdge: edge is not part of triangle.");
            }

            const VertexId opposite = tri.v[static_cast<std::size_t>(local_opp)];
            const Triangle reference_tri = tri;

            const Triangle tri0 = MakeTrianglePreservingOrientation(mesh, reference_tri, opposite, edge_a, midpoint_vertex);
            const Triangle tri1 = MakeTrianglePreservingOrientation(mesh, reference_tri, opposite, midpoint_vertex, edge_b);

            const Vec2& p0a = mesh.vertices.at(tri0.v[0]).position;
            const Vec2& p0b = mesh.vertices.at(tri0.v[1]).position;
            const Vec2& p0c = mesh.vertices.at(tri0.v[2]).position;
            const Vec2& p1a = mesh.vertices.at(tri1.v[0]).position;
            const Vec2& p1b = mesh.vertices.at(tri1.v[1]).position;
            const Vec2& p1c = mesh.vertices.at(tri1.v[2]).position;

            if (!TriangleCandidateIsValid(p0a, p0b, p0c, options) || !TriangleCandidateIsValid(p1a, p1b, p1c, options)) {
                throw std::runtime_error("SplitTriangleAcrossEdge: split would create degenerate child triangle.");
            }

            tri = tri0;
            PushUniqueTriangleId(touched_ids, triangle_id);

            mesh.triangles.push_back(tri1);
            PushUniqueTriangleId(touched_ids, static_cast<TriangleId>(mesh.triangles.size() - 1));
            ++inout_triangles_added;
        }

        [[nodiscard]] bool HasRoomForSplit(const Mesh& mesh, const AdaptiveRefinementOptions& options, bool shared_neighbor) noexcept {
            const u32 vertices_needed = 1;
            const u32 triangles_needed = shared_neighbor ? 2 : 1;

            if (options.max_vertex_count > 0 && mesh.vertices.size() + vertices_needed > options.max_vertex_count) {
                return false;
            }
            if (options.max_triangle_count > 0 && mesh.triangles.size() + triangles_needed > options.max_triangle_count) {
                return false;
            }
            return true;
        }

    } // namespace

    const char* ToString(RefineStepFailureReason reason) noexcept {
        switch (reason) {
        case RefineStepFailureReason::None: return "none";
        case RefineStepFailureReason::SeedTriangleTooSmall: return "seed_triangle_too_small";
        case RefineStepFailureReason::SeedTriangleBBoxTooSmall: return "seed_triangle_bbox_too_small";
        case RefineStepFailureReason::SplitEdgeTooShort: return "split_edge_too_short";
        case RefineStepFailureReason::SplitPointUnsafe: return "split_point_unsafe";
        case RefineStepFailureReason::NeighborChildInvalid: return "neighbor_child_invalid";
        case RefineStepFailureReason::SplitExecutionFailed: return "split_execution_failed";
        default: return "unknown";
        }
    }

    ColorOKLaba SampleImageOKLabaNearest(const ImageOKLaba& image, const Vec2& p) {
        return SampleImageOKLabaNearestImpl(image, p);
    }

    ColorOKLaba SampleImageOKLabaBilinear(const ImageOKLaba& image, const Vec2& p) {
        if (!image.IsValid()) {
            throw std::runtime_error("SampleImageOKLabaBilinear: image is invalid.");
        }

        const f64 x = Clamp(p.x, 0.0, static_cast<f64>(image.Width() - 1));
        const f64 y = Clamp(p.y, 0.0, static_cast<f64>(image.Height() - 1));

        const i32 x0 = Clamp(static_cast<i32>(std::floor(x)), 0, image.Width() - 1);
        const i32 y0 = Clamp(static_cast<i32>(std::floor(y)), 0, image.Height() - 1);
        const i32 x1 = Clamp(x0 + 1, 0, image.Width() - 1);
        const i32 y1 = Clamp(y0 + 1, 0, image.Height() - 1);

        const f64 tx = x - static_cast<f64>(x0);
        const f64 ty = y - static_cast<f64>(y0);

        const ColorOKLaba c00 = image.At(x0, y0);
        const ColorOKLaba c10 = image.At(x1, y0);
        const ColorOKLaba c01 = image.At(x0, y1);
        const ColorOKLaba c11 = image.At(x1, y1);

        const ColorOKLaba cx0 = Lerp(c00, c10, tx);
        const ColorOKLaba cx1 = Lerp(c01, c11, tx);
        return Lerp(cx0, cx1, ty);
    }

    RefineStepResult RefineTriangleGeometryOnly(
        Mesh& mesh,
        TriangleId seed,
        const ImageOKLaba& reference,
        const ErrorPyramid& guidance_pyramid,
        const SingleRefineStepOptions& options) {

        if (!reference.IsValid()) {
            throw std::runtime_error("RefineTriangleGeometryOnly: reference image is invalid.");
        }

        if (!mesh.IsValidTriangleId(seed)) {
            throw std::runtime_error("RefineTriangleGeometryOnly: triangle id out of range.");
        }

        if (!mesh.HasTopology()) {
            const BuildTopologyResult topo = BuildTriangleTopology(mesh);
            if (!topo.ok) {
                throw std::runtime_error("RefineTriangleGeometryOnly: topology build failed: " + topo.error);
            }
        }

        RefineStepResult out{};

        const Triangle& seed_tri = mesh.triangles.at(seed);
        if (IsDegenerate(mesh, seed_tri) || ComputeTriangleArea(mesh, seed_tri) <= options.min_triangle_area) {
            out.failure_reason = RefineStepFailureReason::SeedTriangleTooSmall;
            return out;
        }

        const Vec2& seed_p0 = TriangleP0(mesh, seed_tri);
        const Vec2& seed_p1 = TriangleP1(mesh, seed_tri);
        const Vec2& seed_p2 = TriangleP2(mesh, seed_tri);
        if (!TriangleBBoxExtentIsEnough(seed_p0, seed_p1, seed_p2, options.min_triangle_bbox_extent)) {
            out.failure_reason = RefineStepFailureReason::SeedTriangleBBoxTooSmall;
            return out;
        }

        const EdgeSelection edge = SelectSplitEdge(mesh, seed, &guidance_pyramid, options);
        if (edge.edge_length <= options.min_edge_length) {
            out.failure_reason = RefineStepFailureReason::SplitEdgeTooShort;
            return out;
        }

        TriangleId neighbor_id = kInvalidIndex;
        if (options.split_shared_neighbor && mesh.HasTopology() && edge.local_opposite_index >= 0) {
            neighbor_id = mesh.topology.at(seed).neighbors.at(static_cast<std::size_t>(edge.local_opposite_index));
        }

        const Vec2& pa = mesh.vertices.at(edge.edge_a).position;
        const Vec2& pb = mesh.vertices.at(edge.edge_b).position;

        SplitPointCandidate split_candidate{};
        if (options.use_optimal_split_point) {
            split_candidate = FindBestSplitPointOnEdge(mesh, neighbor_id, edge, reference, &guidance_pyramid, options);
        }
        else {
            const Vec2 midpoint = Midpoint(pa, pb);
            if (SplitPointIsSafe(midpoint, pa, pb, options)) {
                split_candidate.valid = true;
                split_candidate.position = midpoint;
                split_candidate.t = 0.5;
            }
        }

        if (!split_candidate.valid) {
            out.failure_reason = RefineStepFailureReason::SplitPointUnsafe;
            return out;
        }

        const Vec2 split_point = split_candidate.position;

        if (neighbor_id != kInvalidIndex) {
            const Triangle& nbr = mesh.triangles.at(neighbor_id);
            const i32 nbr_opp = FindLocalOppositeVertexIndex(nbr, edge.edge_a, edge.edge_b);
            if (nbr_opp < 0) {
                throw std::runtime_error("RefineTriangleGeometryOnly: topology/neighbor mismatch.");
            }
            const VertexId nbr_opposite = nbr.v[static_cast<std::size_t>(nbr_opp)];
            const Vec2& np = mesh.vertices.at(nbr_opposite).position;
            if (!TriangleCandidateIsValid(np, pa, split_point, options) ||
                !TriangleCandidateIsValid(np, split_point, pb, options)) {
                out.failure_reason = RefineStepFailureReason::NeighborChildInvalid;
                return out;
            }
        }

        const Triangle seed_before = mesh.triangles.at(seed);
        const TriangleNeighbors seed_neighbors_before = mesh.topology.at(seed);

        Triangle neighbor_before{};
        TriangleNeighbors neighbor_neighbors_before{};
        if (neighbor_id != kInvalidIndex) {
            neighbor_before = mesh.triangles.at(neighbor_id);
            neighbor_neighbors_before = mesh.topology.at(neighbor_id);
        }

        const ColorOKLaba midpoint_color = ComputeInsertedVertexColor(
            mesh,
            edge.edge_a,
            edge.edge_b,
            reference,
            split_point,
            split_candidate.t,
            options.new_vertex_color_mode);

        const VertexId midpoint_vertex_id = static_cast<VertexId>(mesh.vertices.size());
        mesh.vertices.push_back(Vertex{ split_point, midpoint_color });

        out.split_performed = true;
        out.seed_triangle_id = seed;
        out.seed_neighbor_triangle_id = neighbor_id;
        out.new_vertex_id = midpoint_vertex_id;
        out.vertices_added = 1;
        out.split_edge_length = edge.edge_length;

        TriangleId seed_new_triangle_id = kInvalidIndex;
        TriangleId neighbor_new_triangle_id = kInvalidIndex;

        try {
            SplitTriangleAcrossEdge(mesh, seed, edge.edge_a, edge.edge_b, midpoint_vertex_id, options, out.triangles_added, out.touched_triangle_ids);
            seed_new_triangle_id = static_cast<TriangleId>(mesh.triangles.size() - 1);

            if (neighbor_id != kInvalidIndex) {
                out.split_shared_neighbor = true;
                SplitTriangleAcrossEdge(mesh, neighbor_id, edge.edge_a, edge.edge_b, midpoint_vertex_id, options, out.triangles_added, out.touched_triangle_ids);
                neighbor_new_triangle_id = static_cast<TriangleId>(mesh.triangles.size() - 1);
            }
        }
        catch (const std::runtime_error&) {
            mesh.vertices.pop_back();
            out = RefineStepResult{};
            out.failure_reason = RefineStepFailureReason::SplitExecutionFailed;
            return out;
        }

        if (options.rebuild_topology_after_split) {
            LocalEdgeSplitTopologyUpdate update{};
            update.seed_triangle_id = seed;
            update.seed_new_triangle_id = seed_new_triangle_id;
            update.neighbor_triangle_id = neighbor_id;
            update.neighbor_new_triangle_id = neighbor_new_triangle_id;
            update.seed_triangle_before = seed_before;
            update.seed_neighbors_before = seed_neighbors_before;
            update.neighbor_triangle_before = neighbor_before;
            update.neighbor_neighbors_before = neighbor_neighbors_before;
            update.split_edge_a = edge.edge_a;
            update.split_edge_b = edge.edge_b;
            update.split_vertex = midpoint_vertex_id;

            const auto topo_t0 = std::chrono::high_resolution_clock::now();
            const BuildTopologyResult topo = UpdateTopologyAfterLocalEdgeSplit(mesh, update);
            const auto topo_t1 = std::chrono::high_resolution_clock::now();
            out.topology_rebuild_performed = true;
            out.topology_rebuild_ms = std::chrono::duration<f64, std::milli>(topo_t1 - topo_t0).count();
            if (!topo.ok) {
                throw std::runtime_error("RefineTriangleGeometryOnly: local topology update failed: " + topo.error);
            }
        }
        else {
            mesh.ClearTopology();
        }

        return out;
    }

    RefineStepResult RefineTriangleOnce(
        Mesh& mesh,
        TriangleId seed,
        const ImageOKLaba& reference,
        const ErrorPyramid& guidance_pyramid,
        const SingleRefineStepOptions& options) {

        if (!reference.IsValid()) {
            throw std::runtime_error("RefineTriangleOnce: reference image is invalid.");
        }

        std::string error;
        if (!ValidateMeshGeometry(mesh, &error)) {
            throw std::runtime_error("RefineTriangleOnce: invalid mesh: " + error);
        }
        if (!mesh.IsValidTriangleId(seed)) {
            throw std::runtime_error("RefineTriangleOnce: triangle id out of range.");
        }

        RefineStepResult out{};
        const MeshErrorReport before = ComputeMeshError(mesh, reference, options.error_options);
        out.error_before = before.summary;

        out = RefineTriangleGeometryOnly(mesh, seed, reference, guidance_pyramid, options);
        if (!out.split_performed) {
            out.error_before = before.summary;
            out.error_after = before.summary;
            return out;
        }

        out.error_before = before.summary;
        out.error_after = ComputeMeshError(mesh, reference, options.error_options).summary;
        return out;
    }

    RefineStepResult RefineWorstTriangleOnce(
        Mesh& mesh,
        const ImageOKLaba& reference,
        const ErrorPyramid& guidance_pyramid,
        const SingleRefineStepOptions& options) {

        if (!reference.IsValid()) {
            throw std::runtime_error("RefineWorstTriangleOnce: reference image is invalid.");
        }

        std::string error;
        if (!ValidateMeshGeometry(mesh, &error)) {
            throw std::runtime_error("RefineWorstTriangleOnce: invalid mesh: " + error);
        }

        const MeshErrorReport before = ComputeMeshError(mesh, reference, options.error_options);
        const TriangleId seed = before.summary.worst_triangle_id;
        if (!mesh.IsValidTriangleId(seed)) {
            RefineStepResult out{};
            out.error_before = before.summary;
            out.error_after = before.summary;
            return out;
        }

        return RefineTriangleOnce(mesh, seed, reference, guidance_pyramid, options);
    }

    RefineStepResult RefineTriangleGeometryOnly(
        Mesh& mesh,
        TriangleId seed,
        const ImageOKLaba& reference,
        const SingleRefineStepOptions& options) {

        const ErrorPyramid guidance_pyramid = BuildErrorPyramid(reference);
        return RefineTriangleGeometryOnly(mesh, seed, reference, guidance_pyramid, options);
    }

    RefineStepResult RefineTriangleOnce(
        Mesh& mesh,
        TriangleId seed,
        const ImageOKLaba& reference,
        const SingleRefineStepOptions& options) {

        const ErrorPyramid guidance_pyramid = BuildErrorPyramid(reference);
        return RefineTriangleOnce(mesh, seed, reference, guidance_pyramid, options);
    }

    RefineStepResult RefineWorstTriangleOnce(
        Mesh& mesh,
        const ImageOKLaba& reference,
        const SingleRefineStepOptions& options) {

        const ErrorPyramid guidance_pyramid = BuildErrorPyramid(reference);
        return RefineWorstTriangleOnce(mesh, reference, guidance_pyramid, options);
    }

    AdaptiveRefinementReport AdaptiveRefineMesh(
        Mesh& mesh,
        const ImageOKLaba& reference,
        const AdaptiveRefinementOptions& options) {

        if (!reference.IsValid()) {
            throw std::runtime_error("AdaptiveRefineMesh: reference image is invalid.");
        }

        std::string error;
        if (!ValidateMeshGeometry(mesh, &error)) {
            throw std::runtime_error("AdaptiveRefineMesh: invalid mesh: " + error);
        }

        AdaptiveRefinementReport out{};
        out.initial_error = ComputeMeshError(mesh, reference, options.step.error_options).summary;
        out.final_error = out.initial_error;
        const ErrorPyramid guidance_pyramid = BuildErrorPyramid(reference);

        for (u32 iter = 0; iter < options.max_iterations; ++iter) {
            if (options.target_weighted_rmse > 0.0 && out.final_error.weighted_rmse <= options.target_weighted_rmse) {
                break;
            }

            const TriangleId seed = out.final_error.worst_triangle_id;
            if (!mesh.IsValidTriangleId(seed)) {
                break;
            }

            bool shared_neighbor = false;
            if (options.step.split_shared_neighbor) {
                if (!mesh.HasTopology()) {
                    const BuildTopologyResult topo = BuildTriangleTopology(mesh);
                    if (!topo.ok) {
                        throw std::runtime_error("AdaptiveRefineMesh: topology build failed: " + topo.error);
                    }
                }
                const EdgeSelection edge = SelectSplitEdge(mesh, seed, &guidance_pyramid, options.step);
                if (edge.local_opposite_index >= 0) {
                    shared_neighbor = mesh.topology.at(seed).neighbors.at(static_cast<std::size_t>(edge.local_opposite_index)) != kInvalidIndex;
                }
            }

            if (!HasRoomForSplit(mesh, options, shared_neighbor)) {
                break;
            }

            const f64 prev_rmse = out.final_error.weighted_rmse;
            RefineStepResult step = RefineTriangleOnce(mesh, seed, reference, guidance_pyramid, options.step);
            if (!step.split_performed) {
                break;
            }

            out.steps.push_back(step);
            out.final_error = step.error_after;

            if (options.stop_when_no_improvement && out.final_error.weighted_rmse >= prev_rmse) {
                break;
            }
        }

        return out;
    }

} // namespace svec
