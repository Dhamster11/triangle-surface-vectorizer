#include "svec/refine/edge_aware_refinement.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <queue>
#include <stdexcept>

#include "svec/error/hierarchical_surface_error.h"
#include "svec/math/geometry.h"
#include "svec/math/scalar.h"
#include "svec/pyramid/error_pyramid.h"
#include "svec/surface/mesh_topology.h"
#include "svec/surface/triangle_plane.h"

namespace svec {
namespace {

[[nodiscard]] f64 CombinedTriangleScore(
    const TriangleErrorMetrics& error,
    const EdgeTriangleMetrics& edge,
    const EdgeAwareSelectionOptions& options) noexcept {

    const f64 edge_term = 1.0 + options.edge_weight * std::pow(Saturate(edge.max_edge_strength), options.edge_power);
    const f64 strong_bonus = edge.max_edge_strength >= options.strong_edge_threshold
        ? options.strong_edge_bonus * edge.max_edge_strength
        : 0.0;
    const f64 mean_bonus = options.mean_edge_bonus * edge.mean_edge_strength;

    return error.weighted_rmse * edge_term + strong_bonus + mean_bonus;
}

[[nodiscard]] bool TriangleBBoxExtentIsEnough(const Mesh& mesh, const Triangle& tri, f64 min_extent) noexcept {
    if (min_extent <= 0.0) {
        return true;
    }
    const Vec2& p0 = TriangleP0(mesh, tri);
    const Vec2& p1 = TriangleP1(mesh, tri);
    const Vec2& p2 = TriangleP2(mesh, tri);
    const f64 dx = Max(p0.x, Max(p1.x, p2.x)) - Min(p0.x, Min(p1.x, p2.x));
    const f64 dy = Max(p0.y, Max(p1.y, p2.y)) - Min(p0.y, Min(p1.y, p2.y));
    return dx >= min_extent || dy >= min_extent;
}

[[nodiscard]] bool CanTriangleStillSplitQuick(const Mesh& mesh, TriangleId ti, const SingleRefineStepOptions& options) noexcept {
    if (!mesh.IsValidTriangleId(ti)) {
        return false;
    }
    const Triangle& tri = mesh.triangles[ti];
    if (tri.HasDuplicateVertices() || IsDegenerate(mesh, tri, options.min_triangle_area * 2.0)) {
        return false;
    }
    if (ComputeTriangleArea(mesh, tri) <= options.min_triangle_area) {
        return false;
    }
    if (!TriangleBBoxExtentIsEnough(mesh, tri, options.min_triangle_bbox_extent)) {
        return false;
    }
    const auto lengths = EdgeLengths(TriangleP0(mesh, tri), TriangleP1(mesh, tri), TriangleP2(mesh, tri));
    return Max(lengths[0], Max(lengths[1], lengths[2])) > options.min_edge_length;
}

[[nodiscard]] bool CanPerformConservativeSplit(const Mesh& mesh, const EdgeAwareRefinementOptions& options) noexcept {
    if (options.max_vertex_count > 0 && mesh.vertices.size() + 1 > options.max_vertex_count) {
        return false;
    }
    if (options.max_triangle_count > 0 && mesh.triangles.size() + 2 > options.max_triangle_count) {
        return false;
    }
    return true;
}

struct ErrorAccum {
    u32 triangles_evaluated = 0;
    u32 triangles_skipped_degenerate = 0;
    u64 sample_count_total = 0;
    f64 weighted_error_area_sum = 0.0;
    f64 total_area = 0.0;
};

struct TriangleCacheEntry {
    TriangleErrorMetrics error{};
    EdgeTriangleMetrics edge{};
    f64 score = -1.0;
    u32 version = 1;
    bool valid = false;
    bool blocked = false;
    f64 proxy_error = 0.0;
    f64 proxy_area = 0.0;
};

struct HeapItem {
    f64 score = -1.0;
    TriangleId triangle_id = kInvalidIndex;
    u32 version = 0;

    [[nodiscard]] bool operator<(const HeapItem& rhs) const noexcept {
        return score < rhs.score;
    }
};

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

[[nodiscard]] HierarchicalSurfaceErrorOptions MakeEdgeAwareHierarchicalErrorOptions(f64 alpha_weight) noexcept {
    HierarchicalSurfaceErrorOptions out{};
    out.color_weight = 1.0;
    out.gradient_weight = 0.45;
    out.detail_weight = 0.28;
    out.structure_weight = 0.90;
    out.peak_weight = 0.70;
    out.alpha_weight = alpha_weight;
    out.per_level_samples = 7;
    out.max_levels_used = 3;
    out.clamp_alpha = true;
    return out;
}

[[nodiscard]] bool ShouldRunSlowEdgeAwareAudit() noexcept {
    const char* value = std::getenv("SVEC_EDGE_AWARE_SLOW_AUDIT");
    if (value == nullptr) {
        return false;
    }
    return value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] TriangleErrorMetrics MakeTriangleProxyErrorMetrics(
    TriangleId triangle_id,
    const TriangleHierarchicalError& error) noexcept {

    TriangleErrorMetrics out{};
    out.triangle_id = triangle_id;
    out.sample_count = error.sample_count;
    out.mse_L = error.composite_error * error.composite_error;
    out.rmse_lab = error.color_rmse;
    out.rmse_alpha = 0.0;
    out.weighted_rmse = error.composite_error;
    out.max_weighted_error = error.peak_residual;
    return out;
}

[[nodiscard]] std::array<Vec2, 7> BuildEdgeProxySamples(const Mesh& mesh, const Triangle& tri) {
    const Vec2& p0 = TriangleP0(mesh, tri);
    const Vec2& p1 = TriangleP1(mesh, tri);
    const Vec2& p2 = TriangleP2(mesh, tri);
    const Vec2 c = TriangleCentroid(p0, p1, p2);
    return {
        c,
        Midpoint(p0, p1),
        Midpoint(p1, p2),
        Midpoint(p2, p0),
        Midpoint(p0, c),
        Midpoint(p1, c),
        Midpoint(p2, c)
    };
}

void AddTriangleToAccum(ErrorAccum& acc, const TriangleCacheEntry& entry) noexcept {
    if (!entry.valid || !entry.error.HasSamples() || entry.proxy_area <= 1e-12) {
        ++acc.triangles_skipped_degenerate;
        return;
    }
    ++acc.triangles_evaluated;
    acc.sample_count_total += entry.error.sample_count;
    acc.weighted_error_area_sum += entry.proxy_error * entry.proxy_area;
    acc.total_area += entry.proxy_area;
}

void RemoveTriangleFromAccum(ErrorAccum& acc, const TriangleCacheEntry& entry) noexcept {
    if (!entry.valid || !entry.error.HasSamples() || entry.proxy_area <= 1e-12) {
        if (acc.triangles_skipped_degenerate > 0) {
            --acc.triangles_skipped_degenerate;
        }
        return;
    }
    if (acc.triangles_evaluated > 0) {
        --acc.triangles_evaluated;
    }
    acc.sample_count_total -= entry.error.sample_count;
    acc.weighted_error_area_sum -= entry.proxy_error * entry.proxy_area;
    acc.total_area -= entry.proxy_area;
}

[[nodiscard]] MeshErrorSummary BuildSummaryFromCache(
    const std::vector<TriangleCacheEntry>& cache,
    const ErrorAccum& acc,
    u32 triangle_count) noexcept {

    MeshErrorSummary out{};
    out.triangles_total = triangle_count;
    out.triangles_evaluated = acc.triangles_evaluated;
    out.triangles_skipped_degenerate = acc.triangles_skipped_degenerate;
    out.sample_count_total = acc.sample_count_total;
    if (acc.total_area > 1e-12) {
        out.weighted_rmse = acc.weighted_error_area_sum / acc.total_area;
        out.rmse_lab = out.weighted_rmse;
        out.mse_L = out.weighted_rmse * out.weighted_rmse;
    }

    for (TriangleId ti = 0; ti < cache.size(); ++ti) {
        if (!cache[ti].valid || !cache[ti].error.HasSamples()) {
            continue;
        }
        if (cache[ti].proxy_error > out.worst_triangle_weighted_rmse) {
            out.worst_triangle_weighted_rmse = cache[ti].proxy_error;
            out.worst_triangle_id = ti;
        }
    }

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

void RecomputeTriangleCacheEntry(
    TriangleCacheEntry& entry,
    const Mesh& mesh,
    TriangleId ti,
    const EdgeMap& edge_map,
    const EdgeAwareSelectionOptions& options,
    const SingleRefineStepOptions& step_options,
    const ErrorPyramid& pyramid,
    const HierarchicalSurfaceErrorOptions& hier_error_options) {

    entry.valid = false;
    entry.error = TriangleErrorMetrics{};
    entry.error.triangle_id = ti;
    entry.edge = EdgeTriangleMetrics{};
    entry.edge.triangle_id = ti;
    entry.score = -1.0;
    entry.proxy_error = 0.0;
    entry.proxy_area = 0.0;

    if (!mesh.IsValidTriangleId(ti)) {
        return;
    }

    const Triangle& tri = mesh.triangles[ti];
    if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
        entry.valid = true;
        entry.blocked = true;
        return;
    }

    const TrianglePlane plane = BuildInterpolationTrianglePlane(mesh, ti);
    const TriangleHierarchicalError proxy_error = ComputeTriangleHierarchicalSurfaceError(mesh, ti, pyramid, plane, hier_error_options);
    entry.error = MakeTriangleProxyErrorMetrics(ti, proxy_error);
    entry.proxy_error = proxy_error.composite_error;
    entry.proxy_area = proxy_error.triangle_area;
    entry.edge = ComputeTriangleEdgeMetrics(mesh, ti, edge_map, options);
    entry.valid = true;
    entry.blocked = !CanTriangleStillSplitQuick(mesh, ti, step_options);
    entry.score = entry.blocked ? -1.0 : CombinedTriangleScore(entry.error, entry.edge, options);
}

EdgeAwareSelectionReport MakeMinimalSelectionReport(
    const TriangleCacheEntry& entry,
    TriangleId triangle_id,
    f64 score,
    const MeshErrorSummary& summary,
    bool keep_inline_score) {

    EdgeAwareSelectionReport report{};
    report.mesh_error = summary;
    report.selected_triangle_id = triangle_id;
    report.selected_score = score;
    if (keep_inline_score && entry.valid) {
        report.edge_metrics.push_back(entry.edge);
        report.per_triangle_score.push_back(EdgeAwareTriangleScore{
            triangle_id,
            entry.error.weighted_rmse,
            entry.edge.mean_edge_strength,
            entry.edge.max_edge_strength,
            score});
    }
    return report;
}

[[nodiscard]] EdgeAwareSelectionReport SelectTriangleForEdgeAwareRefinementImpl(
    const Mesh& mesh,
    const EdgeMap& edge_map,
    const EdgeAwareSelectionOptions& options,
    const SingleRefineStepOptions* step_options,
    const ErrorPyramid& pyramid,
    const HierarchicalSurfaceErrorOptions& hier_error_options) {

    EdgeAwareSelectionReport out{};
    out.mesh_error = BuildProxyMeshErrorSummary(mesh, pyramid, hier_error_options);
    out.edge_metrics.reserve(mesh.triangles.size());
    out.per_triangle_score.reserve(mesh.triangles.size());

    f64 best_score = -1.0;
    TriangleId best_triangle = kInvalidIndex;

    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        if (!mesh.IsValidTriangleId(ti)) {
            continue;
        }
        const Triangle& tri = mesh.triangles[ti];
        if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
            out.edge_metrics.push_back(EdgeTriangleMetrics{ti});
            out.per_triangle_score.push_back(EdgeAwareTriangleScore{ti});
            continue;
        }

        const TrianglePlane plane = BuildInterpolationTrianglePlane(mesh, ti);
        const TriangleHierarchicalError proxy_error = ComputeTriangleHierarchicalSurfaceError(mesh, ti, pyramid, plane, hier_error_options);
        const TriangleErrorMetrics te = MakeTriangleProxyErrorMetrics(ti, proxy_error);
        EdgeTriangleMetrics em = ComputeTriangleEdgeMetrics(mesh, ti, edge_map, options);
        out.edge_metrics.push_back(em);

        EdgeAwareTriangleScore score{};
        score.triangle_id = ti;
        score.weighted_rmse = te.weighted_rmse;
        score.mean_edge_strength = em.mean_edge_strength;
        score.max_edge_strength = em.max_edge_strength;
        score.score = (step_options == nullptr || CanTriangleStillSplitQuick(mesh, ti, *step_options))
            ? CombinedTriangleScore(te, em, options)
            : -1.0;
        out.per_triangle_score.push_back(score);

        if (score.score > best_score) {
            best_score = score.score;
            best_triangle = ti;
        }
    }

    out.selected_triangle_id = best_triangle;
    out.selected_score = Max(best_score, 0.0);
    return out;
}

} // namespace

EdgeTriangleMetrics ComputeTriangleEdgeMetrics(
    const Mesh& mesh,
    TriangleId triangle_id,
    const EdgeMap& edge_map,
    const EdgeAwareSelectionOptions& options) {

    if (!edge_map.IsValid()) {
        throw std::runtime_error("ComputeTriangleEdgeMetrics: edge_map is invalid.");
    }
    if (!mesh.IsValidTriangleId(triangle_id)) {
        throw std::runtime_error("ComputeTriangleEdgeMetrics: triangle id out of range.");
    }

    const Triangle& tri = mesh.triangles[triangle_id];
    EdgeTriangleMetrics out{};
    out.triangle_id = triangle_id;

    if (options.skip_degenerate_triangles && IsDegenerate(mesh, tri)) {
        return out;
    }

    const auto samples = BuildEdgeProxySamples(mesh, tri);
    f64 sum = 0.0;
    for (const Vec2& p : samples) {
        const f64 e = SampleEdgeMapBilinear(edge_map, p);
        sum += e;
        out.max_edge_strength = Max(out.max_edge_strength, e);
        ++out.sample_count;
    }

    if (out.sample_count > 0) {
        out.mean_edge_strength = sum / static_cast<f64>(out.sample_count);
    }
    return out;
}

EdgeAwareSelectionReport SelectTriangleForEdgeAwareRefinement(
    const Mesh& mesh,
    const ImageOKLaba& reference,
    const EdgeMap& edge_map,
    const EdgeAwareSelectionOptions& options) {

    if (!reference.IsValid()) {
        throw std::runtime_error("SelectTriangleForEdgeAwareRefinement: reference is invalid.");
    }
    if (!edge_map.IsValid()) {
        throw std::runtime_error("SelectTriangleForEdgeAwareRefinement: edge_map is invalid.");
    }

    const ErrorPyramid pyramid = BuildErrorPyramid(reference);
    const HierarchicalSurfaceErrorOptions hier_error_options =
        MakeEdgeAwareHierarchicalErrorOptions(options.error_options.alpha_weight);
    return SelectTriangleForEdgeAwareRefinementImpl(mesh, edge_map, options, nullptr, pyramid, hier_error_options);
}

EdgeAwareRefinementReport EdgeAwareRefineMesh(
    Mesh& mesh,
    const ImageOKLaba& reference,
    const EdgeAwareRefinementOptions& options) {

    if (!reference.IsValid()) {
        throw std::runtime_error("EdgeAwareRefineMesh: reference is invalid.");
    }

    std::string error;
    if (!ValidateMeshGeometry(mesh, &error)) {
        throw std::runtime_error("EdgeAwareRefineMesh: invalid mesh: " + error);
    }

    EdgeAwareRefinementReport out{};
    out.edge_map = ComputeEdgeMapSobel(reference, options.edge_map);
    const ErrorPyramid selection_pyramid = BuildErrorPyramid(reference);
    const HierarchicalSurfaceErrorOptions hier_error_options =
        MakeEdgeAwareHierarchicalErrorOptions(options.step.error_options.alpha_weight);
    const bool slow_audit = ShouldRunSlowEdgeAwareAudit();

    auto ensure_edge_map = [&]() {
        if (!options.precompute_edge_map_once || !out.edge_map.IsValid()) {
            out.edge_map = ComputeEdgeMapSobel(reference, options.edge_map);
        }
    };

    auto reached_limits = [&]() -> bool {
        return !CanPerformConservativeSplit(mesh, options);
    };

    auto should_stop_for_error = [&](const MeshErrorSummary& summary) -> bool {
        return options.target_weighted_rmse > 0.0 && summary.weighted_rmse <= options.target_weighted_rmse;
    };

    const u32 iter_cap = options.max_iterations > 0 ? options.max_iterations : 0xFFFFFFFFu;
    out.initial_error = slow_audit
        ? ComputeMeshError(mesh, reference, options.step.error_options).summary
        : BuildProxyMeshErrorSummary(mesh, selection_pyramid, hier_error_options);
    out.final_error = out.initial_error;

    if (!mesh.HasTopology()) {
        const BuildTopologyResult topo = BuildTriangleTopology(mesh);
        if (!topo.ok) {
            throw std::runtime_error("EdgeAwareRefineMesh: topology build failed: " + topo.error);
        }
    }

    const u32 bootstrap_cap = Min(options.bootstrap_iterations, iter_cap);
    for (u32 iter = 0; iter < bootstrap_cap; ++iter) {
        if (should_stop_for_error(out.final_error) || reached_limits()) {
            return out;
        }

        ensure_edge_map();
        EdgeAwareSelectionOptions select_options = options.select;
        select_options.mode = EdgeAwareSelectionMode::FullRescan;
        select_options.keep_full_debug_snapshots = false;

        EdgeAwareSelectionReport selection = SelectTriangleForEdgeAwareRefinementImpl(
            mesh,
            out.edge_map,
            select_options,
            &options.step,
            selection_pyramid,
            hier_error_options);
        out.selections.push_back(selection);
        if (!mesh.IsValidTriangleId(selection.selected_triangle_id)) {
            return out;
        }

        const f64 prev_rmse = out.final_error.weighted_rmse;
        RefineStepResult step = RefineTriangleGeometryOnly(mesh, selection.selected_triangle_id, reference, selection_pyramid, options.step);
        step.error_before = out.final_error;
        if (!step.split_performed) {
            step.error_after = out.final_error;
            ++out.blocked_split_attempts;
            out.steps.push_back(step);
            return out;
        }

        out.final_error = slow_audit
            ? ComputeMeshError(mesh, reference, options.step.error_options).summary
            : BuildProxyMeshErrorSummary(mesh, selection_pyramid, hier_error_options);
        step.error_after = out.final_error;
        out.steps.push_back(step);
        if (options.stop_when_no_improvement && out.final_error.weighted_rmse >= prev_rmse) {
            return out;
        }
    }

    if (options.select.mode == EdgeAwareSelectionMode::FullRescan) {
        for (u32 iter = bootstrap_cap; iter < iter_cap; ++iter) {
            if (should_stop_for_error(out.final_error) || reached_limits()) {
                break;
            }

            ensure_edge_map();
            EdgeAwareSelectionReport selection = SelectTriangleForEdgeAwareRefinementImpl(
                mesh,
                out.edge_map,
                options.select,
                &options.step,
                selection_pyramid,
                hier_error_options);
            out.selections.push_back(selection);
            if (!mesh.IsValidTriangleId(selection.selected_triangle_id)) {
                break;
            }

            const f64 prev_rmse = out.final_error.weighted_rmse;
            RefineStepResult step = RefineTriangleGeometryOnly(mesh, selection.selected_triangle_id, reference, selection_pyramid, options.step);
            step.error_before = out.final_error;
            if (!step.split_performed) {
                step.error_after = out.final_error;
                ++out.blocked_split_attempts;
                out.steps.push_back(step);
                break;
            }

            out.final_error = slow_audit
                ? ComputeMeshError(mesh, reference, options.step.error_options).summary
                : BuildProxyMeshErrorSummary(mesh, selection_pyramid, hier_error_options);
            step.error_after = out.final_error;
            out.steps.push_back(step);
            if (options.stop_when_no_improvement && out.final_error.weighted_rmse >= prev_rmse) {
                break;
            }
        }
        return out;
    }

    std::vector<TriangleCacheEntry> cache(mesh.triangles.size());
    std::priority_queue<HeapItem> heap;
    ErrorAccum accum{};

    auto rebuild_heap_state = [&]() {
        cache.assign(mesh.triangles.size(), TriangleCacheEntry{});
        while (!heap.empty()) {
            heap.pop();
        }
        accum = ErrorAccum{};
        for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
            RecomputeTriangleCacheEntry(cache[ti], mesh, ti, out.edge_map, options.select, options.step, selection_pyramid, hier_error_options);
            AddTriangleToAccum(accum, cache[ti]);
            if (cache[ti].score >= 0.0) {
                heap.push(HeapItem{cache[ti].score, ti, cache[ti].version});
            }
        }
        out.final_error = BuildSummaryFromCache(cache, accum, static_cast<u32>(mesh.triangles.size()));
    };

    rebuild_heap_state();
    if (!slow_audit) {
        out.initial_error = out.final_error;
    }

    for (u32 iter = bootstrap_cap; iter < iter_cap; ++iter) {
        if (should_stop_for_error(out.final_error) || reached_limits()) {
            break;
        }
        ensure_edge_map();

        HeapItem candidate{};
        bool have_candidate = false;
        while (!heap.empty()) {
            candidate = heap.top();
            heap.pop();
            if (candidate.triangle_id >= cache.size()) {
                ++out.stale_heap_pops;
                continue;
            }
            const TriangleCacheEntry& entry = cache[candidate.triangle_id];
            if (!entry.valid || entry.version != candidate.version || entry.blocked || entry.score < 0.0) {
                ++out.stale_heap_pops;
                continue;
            }
            have_candidate = true;
            break;
        }

        if (!have_candidate) {
            EdgeAwareSelectionOptions select_options = options.select;
            select_options.mode = EdgeAwareSelectionMode::FullRescan;
            select_options.keep_full_debug_snapshots = false;
            EdgeAwareSelectionReport selection = SelectTriangleForEdgeAwareRefinementImpl(
                mesh,
                out.edge_map,
                select_options,
                &options.step,
                selection_pyramid,
                hier_error_options);
            if (!mesh.IsValidTriangleId(selection.selected_triangle_id)) {
                break;
            }
            candidate = HeapItem{selection.selected_score, selection.selected_triangle_id, 0};
            have_candidate = true;
        }

        const f64 prev_rmse = out.final_error.weighted_rmse;
        const TriangleId selected_triangle = candidate.triangle_id;
        if (selected_triangle >= cache.size()) {
            break;
        }
        const TriangleCacheEntry selected_entry = cache[selected_triangle];
        out.selections.push_back(MakeMinimalSelectionReport(
            selected_entry,
            selected_triangle,
            candidate.score,
            out.final_error,
            options.select.keep_full_debug_snapshots));

        RefineStepResult step = RefineTriangleGeometryOnly(mesh, selected_triangle, reference, selection_pyramid, options.step);
        step.error_before = out.final_error;
        if (!step.split_performed) {
            if (selected_triangle < cache.size()) {
                cache[selected_triangle].blocked = true;
                cache[selected_triangle].score = -1.0;
                ++cache[selected_triangle].version;
            }
            step.error_after = out.final_error;
            ++out.blocked_split_attempts;
            out.steps.push_back(step);
            continue;
        }

        if (mesh.triangles.size() > cache.size()) {
            cache.resize(mesh.triangles.size());
        }

        for (TriangleId tid : step.touched_triangle_ids) {
            if (tid >= cache.size()) {
                continue;
            }
            if (cache[tid].valid) {
                RemoveTriangleFromAccum(accum, cache[tid]);
            }
        }

        for (TriangleId tid : step.touched_triangle_ids) {
            if (tid >= cache.size()) {
                continue;
            }
            ++cache[tid].version;
            RecomputeTriangleCacheEntry(cache[tid], mesh, tid, out.edge_map, options.select, options.step, selection_pyramid, hier_error_options);
            AddTriangleToAccum(accum, cache[tid]);
            if (cache[tid].score >= 0.0) {
                heap.push(HeapItem{cache[tid].score, tid, cache[tid].version});
            }
        }

        if ((iter & 255u) == 255u) {
            rebuild_heap_state();
        } else {
            out.final_error = BuildSummaryFromCache(cache, accum, static_cast<u32>(mesh.triangles.size()));
        }

        step.error_after = out.final_error;
        out.steps.push_back(step);

        if (options.stop_when_no_improvement && out.final_error.weighted_rmse >= prev_rmse) {
            break;
        }
    }

    if (slow_audit) {
        out.final_error = ComputeMeshError(mesh, reference, options.step.error_options).summary;
        if (!out.steps.empty()) {
            out.steps.back().error_after = out.final_error;
        }
    }

    return out;
}

} // namespace svec
