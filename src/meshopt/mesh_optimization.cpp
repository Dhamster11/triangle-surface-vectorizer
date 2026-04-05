#include "svec/meshopt/mesh_optimization.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "svec/math/geometry.h"
#include "svec/math/scalar.h"
#include "svec/error/hierarchical_surface_error.h"
#include "svec/pyramid/error_pyramid.h"
#include "svec/refine/adaptive_refinement.h"
#include "svec/surface/mesh_topology.h"
#include "svec/surface/triangle_plane.h"

namespace svec {
namespace {

struct EdgeKey {
    VertexId a = kInvalidIndex;
    VertexId b = kInvalidIndex;

    EdgeKey() = default;
    EdgeKey(VertexId x, VertexId y) {
        if (x < y) {
            a = x;
            b = y;
        } else {
            a = y;
            b = x;
        }
    }

    [[nodiscard]] bool operator==(const EdgeKey& rhs) const noexcept {
        return a == rhs.a && b == rhs.b;
    }
};

struct EdgeKeyHasher {
    [[nodiscard]] std::size_t operator()(const EdgeKey& key) const noexcept {
        return (static_cast<std::size_t>(key.a) << 32u) ^ static_cast<std::size_t>(key.b);
    }
};

struct FlipCandidate {
    TriangleId tri_a = kInvalidIndex;
    TriangleId tri_b = kInvalidIndex;
    VertexId shared_a = kInvalidIndex;
    VertexId shared_b = kInvalidIndex;
    VertexId opposite_a = kInvalidIndex;
    VertexId opposite_b = kInvalidIndex;
};

struct LocalProxyErrorAccum {
    f64 weighted_error_area_sum = 0.0;
    f64 total_area = 0.0;
};

[[nodiscard]] f64 ClampToImageX(const ImageOKLaba& reference, f64 x) noexcept {
    return Clamp(x, 0.0, static_cast<f64>(reference.Width() - 1));
}

[[nodiscard]] f64 ClampToImageY(const ImageOKLaba& reference, f64 y) noexcept {
    return Clamp(y, 0.0, static_cast<f64>(reference.Height() - 1));
}

[[nodiscard]] Triangle MakeTrianglePreservingOrientation(
    const Mesh& mesh,
    const Triangle& reference_tri,
    VertexId a,
    VertexId b,
    VertexId c) {

    Triangle out{{a, b, c}};
    const f64 ref_sign = TriangleAreaSigned2(
        TriangleP0(mesh, reference_tri),
        TriangleP1(mesh, reference_tri),
        TriangleP2(mesh, reference_tri));

    const Vec2& pa = mesh.vertices.at(a).position;
    const Vec2& pb = mesh.vertices.at(b).position;
    const Vec2& pc = mesh.vertices.at(c).position;
    const f64 new_sign = TriangleAreaSigned2(pa, pb, pc);

    if ((ref_sign > 0.0 && new_sign < 0.0) || (ref_sign < 0.0 && new_sign > 0.0)) {
        out.v = {a, c, b};
    }
    return out;
}

[[nodiscard]] f64 AngleDegAt(const Vec2& center, const Vec2& a, const Vec2& b) noexcept {
    const Vec2 va = a - center;
    const Vec2 vb = b - center;
    const f64 la = va.Length();
    const f64 lb = vb.Length();
    if (la <= kEpsilon || lb <= kEpsilon) {
        return 0.0;
    }
    const f64 c = Clamp(Dot(va, vb) / (la * lb), -1.0, 1.0);
    return std::acos(c) * 180.0 / kPi;
}

[[nodiscard]] f64 TriangleMinAngleDegrees(const Vec2& p0, const Vec2& p1, const Vec2& p2) noexcept {
    const f64 a0 = AngleDegAt(p0, p1, p2);
    const f64 a1 = AngleDegAt(p1, p0, p2);
    const f64 a2 = AngleDegAt(p2, p0, p1);
    return Min(a0, Min(a1, a2));
}

[[nodiscard]] f64 EdgeStrengthAlong(const EdgeMap* edge_map, const Vec2& a, const Vec2& b) {
    if (edge_map == nullptr) {
        return 0.0;
    }
    const Vec2 m = Midpoint(a, b);
    const f64 ea = SampleEdgeMapBilinear(*edge_map, a);
    const f64 em = SampleEdgeMapBilinear(*edge_map, m);
    const f64 eb = SampleEdgeMapBilinear(*edge_map, b);
    return (ea + em + eb) / 3.0;
}

[[nodiscard]] bool ArePointsOnOppositeSidesOfLine(
    const Vec2& line_a,
    const Vec2& line_b,
    const Vec2& p,
    const Vec2& q,
    f64 eps = 1e-12) noexcept {

    const f64 sp = Cross(line_b - line_a, p - line_a);
    const f64 sq = Cross(line_b - line_a, q - line_a);
    return (sp > eps && sq < -eps) || (sp < -eps && sq > eps);
}

[[nodiscard]] bool IsFlipGeometricallyValid(
    const Mesh& mesh,
    VertexId shared_a,
    VertexId shared_b,
    VertexId opposite_a,
    VertexId opposite_b) noexcept {

    const Vec2& a = mesh.vertices.at(shared_a).position;
    const Vec2& b = mesh.vertices.at(shared_b).position;
    const Vec2& c = mesh.vertices.at(opposite_a).position;
    const Vec2& d = mesh.vertices.at(opposite_b).position;

    if (!ArePointsOnOppositeSidesOfLine(a, b, c, d)) {
        return false;
    }
    if (!ArePointsOnOppositeSidesOfLine(c, d, a, b)) {
        return false;
    }
    return true;
}

[[nodiscard]] FlipCandidate MakeFlipCandidate(const Mesh& mesh, TriangleId tri_a_id, TriangleId tri_b_id) {
    const Triangle& tri_a = mesh.triangles.at(tri_a_id);
    const Triangle& tri_b = mesh.triangles.at(tri_b_id);

    for (i32 opp = 0; opp < 3; ++opp) {
        if (mesh.topology.at(tri_a_id).neighbors[static_cast<std::size_t>(opp)] != tri_b_id) {
            continue;
        }

        const VertexId shared_a = tri_a.v[static_cast<std::size_t>((opp + 1) % 3)];
        const VertexId shared_b = tri_a.v[static_cast<std::size_t>((opp + 2) % 3)];
        const VertexId opposite_a = tri_a.v[static_cast<std::size_t>(opp)];

        const i32 opp_b = FindLocalOppositeVertexIndex(tri_b, shared_a, shared_b);
        if (opp_b < 0) {
            break;
        }
        const VertexId opposite_b = tri_b.v[static_cast<std::size_t>(opp_b)];

        return {tri_a_id, tri_b_id, shared_a, shared_b, opposite_a, opposite_b};
    }

    return {};
}

[[nodiscard]] Plane1 InterpolationPlaneFromTriangleChannel(
    const Vec2& p0,
    const Vec2& p1,
    const Vec2& p2,
    f64 z0,
    f64 z1,
    f64 z2) noexcept {

    Plane1 out{};
    const f64 dx10 = p1.x - p0.x;
    const f64 dy10 = p1.y - p0.y;
    const f64 dx20 = p2.x - p0.x;
    const f64 dy20 = p2.y - p0.y;
    const f64 det = dx10 * dy20 - dy10 * dx20;
    if (std::abs(det) <= 1e-12) {
        out.c = (z0 + z1 + z2) / 3.0;
        return out;
    }

    const f64 dz10 = z1 - z0;
    const f64 dz20 = z2 - z0;
    out.cx = (dz10 * dy20 - dy10 * dz20) / det;
    out.cy = (dx10 * dz20 - dz10 * dx20) / det;
    out.c = z0 - out.cx * p0.x - out.cy * p0.y;
    return out;
}

[[nodiscard]] TrianglePlane BuildInterpolationTrianglePlane(const Mesh& mesh, TriangleId triangle_id) {
    const Triangle& tri = mesh.triangles.at(triangle_id);
    const Vec2& p0 = TriangleP0(mesh, tri);
    const Vec2& p1 = TriangleP1(mesh, tri);
    const Vec2& p2 = TriangleP2(mesh, tri);
    const ColorOKLaba& c0 = TriangleC0(mesh, tri);
    const ColorOKLaba& c1 = TriangleC1(mesh, tri);
    const ColorOKLaba& c2 = TriangleC2(mesh, tri);

    TrianglePlane out{};
    out.triangle_id = triangle_id;
    out.L = InterpolationPlaneFromTriangleChannel(p0, p1, p2, c0.L, c1.L, c2.L);
    out.a = InterpolationPlaneFromTriangleChannel(p0, p1, p2, c0.a, c1.a, c2.a);
    out.b = InterpolationPlaneFromTriangleChannel(p0, p1, p2, c0.b, c1.b, c2.b);
    out.alpha = InterpolationPlaneFromTriangleChannel(p0, p1, p2, c0.alpha, c1.alpha, c2.alpha);
    return out;
}

[[nodiscard]] HierarchicalSurfaceErrorOptions MakeOptimizationHierarchicalErrorOptions(f64 alpha_weight) noexcept {
    HierarchicalSurfaceErrorOptions out{};
    out.color_weight = 1.0;
    out.gradient_weight = 0.40;
    out.detail_weight = 0.22;
    out.structure_weight = 0.85;
    out.peak_weight = 0.65;
    out.alpha_weight = alpha_weight;
    out.per_level_samples = 7;
    out.max_levels_used = 3;
    out.clamp_alpha = true;
    return out;
}

[[nodiscard]] LocalProxyErrorAccum ComputeLocalProxyErrorAccum(
    const Mesh& mesh,
    const std::vector<TriangleId>& triangle_ids,
    const ErrorPyramid& pyramid,
    const HierarchicalSurfaceErrorOptions& error_options) {

    LocalProxyErrorAccum out{};
    for (TriangleId ti : triangle_ids) {
        if (!mesh.IsValidTriangleId(ti)) {
            continue;
        }
        const Triangle& tri = mesh.triangles[ti];
        if (IsDegenerate(mesh, tri)) {
            continue;
        }
        const TrianglePlane plane = BuildInterpolationTrianglePlane(mesh, ti);
        const TriangleHierarchicalError e = ComputeTriangleHierarchicalSurfaceError(mesh, ti, pyramid, plane, error_options);
        if (e.triangle_area <= 0.0) {
            continue;
        }
        out.weighted_error_area_sum += e.composite_error * e.triangle_area;
        out.total_area += e.triangle_area;
    }
    return out;
}

[[nodiscard]] f64 LocalProxyError(const LocalProxyErrorAccum& acc) noexcept {
    if (acc.total_area <= 1e-12) {
        return 0.0;
    }
    return acc.weighted_error_area_sum / acc.total_area;
}

[[nodiscard]] std::vector<TriangleId> UniqueTriangleIds(std::initializer_list<TriangleId> ids) {
    std::vector<TriangleId> out;
    out.reserve(ids.size());
    for (TriangleId id : ids) {
        if (id == kInvalidIndex) {
            continue;
        }
        if (std::find(out.begin(), out.end(), id) == out.end()) {
            out.push_back(id);
        }
    }
    return out;
}


[[nodiscard]] f64 LocalMinAngleDegrees(const Mesh& mesh, const std::vector<TriangleId>& triangle_ids) {
    f64 out = 180.0;
    bool any = false;
    for (TriangleId ti : triangle_ids) {
        if (!mesh.IsValidTriangleId(ti)) {
            continue;
        }
        if (IsDegenerate(mesh, mesh.triangles[ti])) {
            return 0.0;
        }
        out = Min(out, ComputeTriangleMinAngleDegrees(mesh, ti));
        any = true;
    }
    return any ? out : 0.0;
}

[[nodiscard]] bool AnyIncidentTriangleDegenerate(const Mesh& mesh, const std::vector<TriangleId>& tris) noexcept {
    for (TriangleId ti : tris) {
        if (!mesh.IsValidTriangleId(ti)) {
            return true;
        }
        if (IsDegenerate(mesh, mesh.triangles[ti])) {
            return true;
        }
    }
    return false;
}

struct SmoothingTopologyCache {
    std::vector<std::vector<TriangleId>> incident;
    std::vector<std::vector<VertexId>> one_ring;
    std::vector<bool> boundary;
};

[[nodiscard]] bool ShouldRunSlowOptimizationAudit() noexcept {
    const char* value = std::getenv("SVEC_OPTIMIZE_SLOW_AUDIT");
    if (value == nullptr) {
        return false;
    }
    return value[0] != '\0' && value[0] != '0';
}

} // namespace

std::vector<std::vector<TriangleId>> BuildVertexIncidentTriangles(const Mesh& mesh) {
    std::vector<std::vector<TriangleId>> out(mesh.vertices.size());
    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        const Triangle& tri = mesh.triangles[ti];
        for (VertexId v : tri.v) {
            if (v == kInvalidIndex || v >= out.size()) {
                throw std::runtime_error("BuildVertexIncidentTriangles: invalid vertex index in triangle.");
            }
            out[v].push_back(ti);
        }
    }
    return out;
}

std::vector<std::vector<VertexId>> BuildVertexOneRingAdjacency(const Mesh& mesh) {
    std::vector<std::unordered_set<VertexId>> sets(mesh.vertices.size());
    for (const Triangle& tri : mesh.triangles) {
        const VertexId v0 = tri.v[0];
        const VertexId v1 = tri.v[1];
        const VertexId v2 = tri.v[2];
        sets[v0].insert(v1); sets[v0].insert(v2);
        sets[v1].insert(v0); sets[v1].insert(v2);
        sets[v2].insert(v0); sets[v2].insert(v1);
    }

    std::vector<std::vector<VertexId>> out(mesh.vertices.size());
    for (std::size_t i = 0; i < sets.size(); ++i) {
        out[i].assign(sets[i].begin(), sets[i].end());
    }
    return out;
}

std::vector<bool> BuildBoundaryVertexMask(const Mesh& mesh) {
    if (!mesh.HasTopology()) {
        throw std::runtime_error("BuildBoundaryVertexMask: mesh topology is required.");
    }

    std::vector<bool> out(mesh.vertices.size(), false);
    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        const Triangle& tri = mesh.triangles[ti];
        const TriangleNeighbors& nbr = mesh.topology[ti];
        for (i32 opp = 0; opp < 3; ++opp) {
            if (nbr.neighbors[static_cast<std::size_t>(opp)] != kInvalidIndex) {
                continue;
            }
            const VertexId a = tri.v[static_cast<std::size_t>((opp + 1) % 3)];
            const VertexId b = tri.v[static_cast<std::size_t>((opp + 2) % 3)];
            out[a] = true;
            out[b] = true;
        }
    }
    return out;
}

[[nodiscard]] SmoothingTopologyCache BuildSmoothingTopologyCache(const Mesh& mesh) {
    SmoothingTopologyCache out{};
    out.incident = svec::BuildVertexIncidentTriangles(mesh);
    out.one_ring = svec::BuildVertexOneRingAdjacency(mesh);
    out.boundary = svec::BuildBoundaryVertexMask(mesh);
    return out;
}

[[nodiscard]] MeshErrorSummary BuildProxyMeshErrorSummary(
    const Mesh& mesh,
    const ErrorPyramid& pyramid,
    const HierarchicalSurfaceErrorOptions& error_options) {

    MeshErrorSummary out{};
    out.triangles_total = static_cast<u32>(mesh.triangles.size());

    f64 weighted_sum = 0.0;
    f64 area_sum = 0.0;
    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        const Triangle& tri = mesh.triangles[ti];
        if (IsDegenerate(mesh, tri)) {
            ++out.triangles_skipped_degenerate;
            continue;
        }

        const TrianglePlane plane = BuildInterpolationTrianglePlane(mesh, ti);
        const TriangleHierarchicalError e = ComputeTriangleHierarchicalSurfaceError(mesh, ti, pyramid, plane, error_options);
        ++out.triangles_evaluated;
        out.sample_count_total += e.sample_count;

        if (e.triangle_area > 1e-12) {
            weighted_sum += e.composite_error * e.triangle_area;
            area_sum += e.triangle_area;
        }
        if (e.composite_error > out.worst_triangle_weighted_rmse) {
            out.worst_triangle_weighted_rmse = e.composite_error;
            out.worst_triangle_id = ti;
        }
    }

    if (area_sum > 1e-12) {
        out.weighted_rmse = weighted_sum / area_sum;
        out.rmse_lab = out.weighted_rmse;
        out.mse_L = out.weighted_rmse * out.weighted_rmse;
    }
    return out;
}

f64 ComputeTriangleMinAngleDegrees(const Mesh& mesh, TriangleId triangle_id) {
    if (!mesh.IsValidTriangleId(triangle_id)) {
        throw std::runtime_error("ComputeTriangleMinAngleDegrees: triangle id out of range.");
    }
    const Triangle& tri = mesh.triangles[triangle_id];
    return TriangleMinAngleDegrees(TriangleP0(mesh, tri), TriangleP1(mesh, tri), TriangleP2(mesh, tri));
}

f64 ComputeMeshMinTriangleAngleDegrees(const Mesh& mesh, bool skip_degenerate) {
    f64 out = 180.0;
    bool any = false;
    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        const Triangle& tri = mesh.triangles[ti];
        if (skip_degenerate && IsDegenerate(mesh, tri)) {
            continue;
        }
        out = Min(out, ComputeTriangleMinAngleDegrees(mesh, ti));
        any = true;
    }
    return any ? out : 0.0;
}

f64 ComputeMeshMeanTriangleMinAngleDegrees(const Mesh& mesh, bool skip_degenerate) {
    f64 sum = 0.0;
    u64 n = 0;
    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        const Triangle& tri = mesh.triangles[ti];
        if (skip_degenerate && IsDegenerate(mesh, tri)) {
            continue;
        }
        sum += ComputeTriangleMinAngleDegrees(mesh, ti);
        ++n;
    }
    return n > 0 ? sum / static_cast<f64>(n) : 0.0;
}

EdgeFlipPassReport FlipImprovingInteriorEdges(
    Mesh& mesh,
    const ImageOKLaba& reference,
    const ErrorEstimatorOptions& error_options,
    const EdgeFlipOptions& options,
    const EdgeMap* edge_map,
    const ErrorPyramid* error_pyramid,
    const HierarchicalSurfaceErrorOptions* hier_error_options) {

    EdgeFlipPassReport report{};
    if (!options.enabled || mesh.triangles.empty()) {
        return report;
    }
    if (!reference.IsValid()) {
        throw std::runtime_error("FlipImprovingInteriorEdges: reference image is invalid.");
    }
    if (!mesh.HasTopology()) {
        const BuildTopologyResult topo = BuildTriangleTopology(mesh);
        if (!topo.ok) {
            throw std::runtime_error("FlipImprovingInteriorEdges: failed to build topology: " + topo.error);
        }
    }

    std::unordered_set<std::size_t> visited;
    visited.reserve(mesh.triangles.size() * 2);

    for (TriangleId ta = 0; ta < mesh.triangles.size(); ++ta) {
        for (i32 opp = 0; opp < 3; ++opp) {
            const TriangleId tb = mesh.topology[ta].neighbors[static_cast<std::size_t>(opp)];
            if (tb == kInvalidIndex) {
                continue;
            }
            ++report.interior_edges;
            const TriangleId lo = Min(ta, tb);
            const TriangleId hi = Max(ta, tb);
            const std::size_t pair_key = (static_cast<std::size_t>(lo) << 32u) ^ static_cast<std::size_t>(hi);
            if (!visited.insert(pair_key).second) {
                continue;
            }
            ++report.candidate_edges;

            const FlipCandidate c = MakeFlipCandidate(mesh, ta, tb);
            if (c.tri_a == kInvalidIndex) {
                ++report.skipped_invalid;
                continue;
            }

            if (!IsFlipGeometricallyValid(mesh, c.shared_a, c.shared_b, c.opposite_a, c.opposite_b)) {
                ++report.skipped_invalid;
                continue;
            }

            const Vec2& pa = mesh.vertices[c.shared_a].position;
            const Vec2& pb = mesh.vertices[c.shared_b].position;
            if (options.preserve_strong_edges && edge_map != nullptr) {
                if (EdgeStrengthAlong(edge_map, pa, pb) >= options.strong_edge_threshold) {
                    ++report.skipped_strong_edge;
                    continue;
                }
            }

            const f64 quality_before = Min(ComputeTriangleMinAngleDegrees(mesh, c.tri_a),
                                           ComputeTriangleMinAngleDegrees(mesh, c.tri_b));

            const Triangle old_a = mesh.triangles[c.tri_a];
            const Triangle old_b = mesh.triangles[c.tri_b];
            const auto local_tris = UniqueTriangleIds({c.tri_a, c.tri_b});
            const f64 err_before =
                (error_pyramid != nullptr && hier_error_options != nullptr)
                    ? LocalProxyError(ComputeLocalProxyErrorAccum(mesh, local_tris, *error_pyramid, *hier_error_options))
                    : 0.0;

            mesh.triangles[c.tri_a] = MakeTrianglePreservingOrientation(mesh, old_a, c.opposite_a, c.shared_a, c.opposite_b);
            mesh.triangles[c.tri_b] = MakeTrianglePreservingOrientation(mesh, old_b, c.opposite_b, c.shared_b, c.opposite_a);

            if (IsDegenerate(mesh, mesh.triangles[c.tri_a]) || IsDegenerate(mesh, mesh.triangles[c.tri_b])) {
                mesh.triangles[c.tri_a] = old_a;
                mesh.triangles[c.tri_b] = old_b;
                ++report.skipped_invalid;
                continue;
            }

            const f64 quality_after = Min(ComputeTriangleMinAngleDegrees(mesh, c.tri_a),
                                          ComputeTriangleMinAngleDegrees(mesh, c.tri_b));
            if (quality_after < quality_before + options.min_quality_improvement_degrees) {
                mesh.triangles[c.tri_a] = old_a;
                mesh.triangles[c.tri_b] = old_b;
                ++report.skipped_quality;
                continue;
            }

            const f64 err_after =
                (error_pyramid != nullptr && hier_error_options != nullptr)
                    ? LocalProxyError(ComputeLocalProxyErrorAccum(mesh, local_tris, *error_pyramid, *hier_error_options))
                    : 0.0;
            const f64 max_allowed = err_before * (1.0 + options.max_local_error_increase_ratio) + 1e-12;
            if (error_pyramid != nullptr && hier_error_options != nullptr && err_after > max_allowed) {
                mesh.triangles[c.tri_a] = old_a;
                mesh.triangles[c.tri_b] = old_b;
                ++report.skipped_error;
                continue;
            }

            ++report.flips_applied;
        }
    }

    if (options.rebuild_topology_after_pass && report.flips_applied > 0) {
        const BuildTopologyResult topo = BuildTriangleTopology(mesh);
        if (!topo.ok) {
            throw std::runtime_error("FlipImprovingInteriorEdges: failed to rebuild topology: " + topo.error);
        }
    }

    return report;
}

VertexSmoothingPassReport SmoothMeshVerticesOnceWithCacheImpl(
    Mesh& mesh,
    const ImageOKLaba& reference,
    const ErrorEstimatorOptions& error_options,
    const VertexSmoothingOptions& options,
    const EdgeMap* edge_map,
    const ErrorPyramid* error_pyramid,
    const HierarchicalSurfaceErrorOptions* hier_error_options,
    const SmoothingTopologyCache& cache) {

    (void)error_options;
    VertexSmoothingPassReport report{};
    if (!options.enabled || mesh.vertices.empty()) {
        return report;
    }
    if (!reference.IsValid()) {
        throw std::runtime_error("SmoothMeshVerticesOnce: reference image is invalid.");
    }
    if (!mesh.HasTopology()) {
        throw std::runtime_error("SmoothMeshVerticesOnce: mesh topology is required.");
    }

    const auto& incident = cache.incident;
    const auto& one_ring = cache.one_ring;
    const auto& boundary = cache.boundary;

    for (VertexId vi = 0; vi < mesh.vertices.size(); ++vi) {
        ++report.vertices_considered;

        if (options.skip_boundary_vertices && boundary[vi]) {
            ++report.skipped_boundary;
            continue;
        }
        if (incident[vi].empty() || one_ring[vi].empty()) {
            ++report.skipped_geometry;
            continue;
        }

        const Vec2 old_pos = mesh.vertices[vi].position;
        if (options.preserve_strong_edges && edge_map != nullptr) {
            if (SampleEdgeMapBilinear(*edge_map, old_pos) >= options.strong_edge_threshold) {
                ++report.skipped_strong_edge;
                continue;
            }
        }

        Vec2 avg{0.0, 0.0};
        for (VertexId nbr : one_ring[vi]) {
            avg += mesh.vertices[nbr].position;
        }
        avg /= static_cast<f64>(one_ring[vi].size());

        Vec2 proposed = Lerp(old_pos, avg, options.lambda);
        Vec2 delta = proposed - old_pos;
        const f64 move_len = delta.Length();
        if (move_len <= 1e-12) {
            ++report.skipped_geometry;
            continue;
        }
        if (move_len > options.max_move_distance && options.max_move_distance > 0.0) {
            delta *= (options.max_move_distance / move_len);
            proposed = old_pos + delta;
        }

        proposed.x = ClampToImageX(reference, proposed.x);
        proposed.y = ClampToImageY(reference, proposed.y);

        const Vertex old_vertex = mesh.vertices[vi];
        const f64 err_before =
            (error_pyramid != nullptr && hier_error_options != nullptr)
                ? LocalProxyError(ComputeLocalProxyErrorAccum(mesh, incident[vi], *error_pyramid, *hier_error_options))
                : 0.0;
        const f64 angle_before = LocalMinAngleDegrees(mesh, incident[vi]);

        mesh.vertices[vi].position = proposed;
        if (options.update_vertex_color_from_reference) {
            mesh.vertices[vi].color = SampleImageOKLabaBilinear(reference, proposed);
        }

        if (AnyIncidentTriangleDegenerate(mesh, incident[vi])) {
            mesh.vertices[vi] = old_vertex;
            ++report.skipped_geometry;
            continue;
        }

        const f64 err_after =
            (error_pyramid != nullptr && hier_error_options != nullptr)
                ? LocalProxyError(ComputeLocalProxyErrorAccum(mesh, incident[vi], *error_pyramid, *hier_error_options))
                : 0.0;
        const f64 max_allowed = err_before * (1.0 + options.max_local_error_increase_ratio) + 1e-12;
        if (error_pyramid != nullptr && hier_error_options != nullptr && err_after > max_allowed) {
            mesh.vertices[vi] = old_vertex;
            ++report.skipped_error;
            continue;
        }

        const f64 angle_after = LocalMinAngleDegrees(mesh, incident[vi]);
        if (angle_after + options.max_local_min_angle_drop_degrees < angle_before) {
            mesh.vertices[vi] = old_vertex;
            ++report.skipped_geometry;
            continue;
        }

        ++report.vertices_moved;
    }

    return report;
}

VertexSmoothingPassReport SmoothMeshVerticesOnce(
    Mesh& mesh,
    const ImageOKLaba& reference,
    const ErrorEstimatorOptions& error_options,
    const VertexSmoothingOptions& options,
    const EdgeMap* edge_map,
    const ErrorPyramid* error_pyramid,
    const HierarchicalSurfaceErrorOptions* hier_error_options) {

    if (!mesh.HasTopology()) {
        const BuildTopologyResult topo = BuildTriangleTopology(mesh);
        if (!topo.ok) {
            throw std::runtime_error("SmoothMeshVerticesOnce: failed to build topology: " + topo.error);
        }
    }

    const SmoothingTopologyCache cache = BuildSmoothingTopologyCache(mesh);
    return SmoothMeshVerticesOnceWithCacheImpl(
        mesh,
        reference,
        error_options,
        options,
        edge_map,
        error_pyramid,
        hier_error_options,
        cache);
}

MeshOptimizationReport OptimizeMesh(
    Mesh& mesh,
    const ImageOKLaba& reference,
    const MeshOptimizationOptions& options,
    const EdgeMap* edge_map) {

    if (!reference.IsValid()) {
        throw std::runtime_error("OptimizeMesh: reference image is invalid.");
    }

    if (!mesh.HasTopology()) {
        const BuildTopologyResult topo = BuildTriangleTopology(mesh);
        if (!topo.ok) {
            throw std::runtime_error("OptimizeMesh: failed to build topology: " + topo.error);
        }
    }

    MeshOptimizationReport report{};
    const ErrorPyramid optimization_error_pyramid = BuildErrorPyramid(reference);
    const HierarchicalSurfaceErrorOptions optimization_hier_error =
        MakeOptimizationHierarchicalErrorOptions(options.error_options.alpha_weight);

    const bool slow_audit = ShouldRunSlowOptimizationAudit();
    report.initial_error = slow_audit
        ? ComputeMeshError(mesh, reference, options.error_options).summary
        : BuildProxyMeshErrorSummary(mesh, optimization_error_pyramid, optimization_hier_error);
    report.initial_min_triangle_angle_deg = ComputeMeshMinTriangleAngleDegrees(mesh, true);
    report.initial_mean_min_angle_deg = ComputeMeshMeanTriangleMinAngleDegrees(mesh, true);

    for (u32 iter = 0; iter < Max(options.outer_iterations, 1u); ++iter) {
        MeshOptimizationIterationReport it{};
        bool any_change = false;

        it.flip = FlipImprovingInteriorEdges(mesh, reference, options.error_options, options.flip, edge_map, &optimization_error_pyramid, &optimization_hier_error);
        any_change = any_change || (it.flip.flips_applied > 0);

        if (options.smooth.enabled) {
            const u32 smooth_iters = Max(options.smooth.iterations, 1u);
            it.smoothing_passes.reserve(smooth_iters);
            const SmoothingTopologyCache smoothing_cache = BuildSmoothingTopologyCache(mesh);
            for (u32 s = 0; s < smooth_iters; ++s) {
                const VertexSmoothingPassReport pass = SmoothMeshVerticesOnceWithCacheImpl(
                    mesh,
                    reference,
                    options.error_options,
                    options.smooth,
                    edge_map,
                    &optimization_error_pyramid,
                    &optimization_hier_error,
                    smoothing_cache);
                any_change = any_change || (pass.vertices_moved > 0);
                it.smoothing_passes.push_back(pass);
            }
        }

        report.iterations.push_back(it);

        if (options.stop_when_no_changes && !any_change) {
            break;
        }
    }

    report.final_error = slow_audit
        ? ComputeMeshError(mesh, reference, options.error_options).summary
        : BuildProxyMeshErrorSummary(mesh, optimization_error_pyramid, optimization_hier_error);
    report.final_min_triangle_angle_deg = ComputeMeshMinTriangleAngleDegrees(mesh, true);
    report.final_mean_min_angle_deg = ComputeMeshMeanTriangleMinAngleDegrees(mesh, true);
    return report;
}

} // namespace svec
