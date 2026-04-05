#include "svec/refine/pyramid_refinement.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>
#include <stdexcept>
#include <vector>

#include "svec/image/scanline_integral_stats.h"
#include "svec/math/geometry.h"
#include "svec/math/scalar.h"
#include "svec/math/vec2.h"
#include "svec/surface/mesh_topology.h"

namespace svec {
    namespace {

        struct HeapNode {
            f64 score = 0.0;
            TriangleId triangle_id = kInvalidIndex;
        };

        class IndexedMaxHeap {
        public:
            void EnsureCapacity(std::size_t triangle_count) {
                if (m_positions.size() < triangle_count) {
                    m_positions.resize(triangle_count, -1);
                }
            }

            [[nodiscard]] bool Empty() const noexcept {
                return m_nodes.empty();
            }

            [[nodiscard]] std::size_t Size() const noexcept {
                return m_nodes.size();
            }

            void Clear() {
                for (const HeapNode& node : m_nodes) {
                    if (node.triangle_id < m_positions.size()) {
                        m_positions[node.triangle_id] = -1;
                    }
                }
                m_nodes.clear();
            }

            void Upsert(TriangleId triangle_id, f64 score, PyramidRefinementTelemetry* telemetry) {
                EnsureCapacity(static_cast<std::size_t>(triangle_id) + 1u);
                const i32 pos = m_positions[triangle_id];
                if (pos < 0) {
                    m_nodes.push_back(HeapNode{score, triangle_id});
                    const i32 new_pos = static_cast<i32>(m_nodes.size() - 1u);
                    m_positions[triangle_id] = new_pos;
                    SiftUp(new_pos);
                    if (telemetry != nullptr) {
                        ++telemetry->heap_pushes_total;
                        telemetry->heap_max_size = Max<u64>(telemetry->heap_max_size, static_cast<u64>(m_nodes.size()));
                    }
                    return;
                }

                HeapNode& node = m_nodes[static_cast<std::size_t>(pos)];
                const f64 old_score = node.score;
                node.score = score;
                if (score > old_score) {
                    SiftUp(pos);
                } else if (score < old_score) {
                    SiftDown(pos);
                }
            }

            [[nodiscard]] HeapNode PopMax() {
                if (m_nodes.empty()) {
                    return {};
                }

                HeapNode out = m_nodes.front();
                const HeapNode last = m_nodes.back();
                m_nodes.pop_back();
                m_positions[out.triangle_id] = -1;
                if (!m_nodes.empty()) {
                    m_nodes.front() = last;
                    m_positions[last.triangle_id] = 0;
                    SiftDown(0);
                }
                return out;
            }

            void Remove(TriangleId triangle_id) {
                if (triangle_id >= m_positions.size()) {
                    return;
                }
                const i32 pos = m_positions[triangle_id];
                if (pos < 0) {
                    return;
                }

                const i32 last_pos = static_cast<i32>(m_nodes.size() - 1);
                m_positions[triangle_id] = -1;
                if (pos == last_pos) {
                    m_nodes.pop_back();
                    return;
                }

                const HeapNode last = m_nodes.back();
                m_nodes.pop_back();
                m_nodes[static_cast<std::size_t>(pos)] = last;
                m_positions[last.triangle_id] = pos;

                const i32 parent = pos > 0 ? (pos - 1) / 2 : 0;
                if (pos > 0 && HigherPriority(m_nodes[static_cast<std::size_t>(pos)], m_nodes[static_cast<std::size_t>(parent)])) {
                    SiftUp(pos);
                } else {
                    SiftDown(pos);
                }
            }

        private:
            [[nodiscard]] bool HigherPriority(const HeapNode& lhs, const HeapNode& rhs) const noexcept {
                if (lhs.score != rhs.score) {
                    return lhs.score > rhs.score;
                }
                return lhs.triangle_id < rhs.triangle_id;
            }

            void SwapNodes(i32 a, i32 b) {
                if (a == b) {
                    return;
                }
                std::swap(m_nodes[static_cast<std::size_t>(a)], m_nodes[static_cast<std::size_t>(b)]);
                m_positions[m_nodes[static_cast<std::size_t>(a)].triangle_id] = a;
                m_positions[m_nodes[static_cast<std::size_t>(b)].triangle_id] = b;
            }

            void SiftUp(i32 pos) {
                while (pos > 0) {
                    const i32 parent = (pos - 1) / 2;
                    if (!HigherPriority(m_nodes[static_cast<std::size_t>(pos)], m_nodes[static_cast<std::size_t>(parent)])) {
                        break;
                    }
                    SwapNodes(pos, parent);
                    pos = parent;
                }
            }

            void SiftDown(i32 pos) {
                const i32 n = static_cast<i32>(m_nodes.size());
                while (true) {
                    const i32 left = pos * 2 + 1;
                    const i32 right = left + 1;
                    i32 best = pos;
                    if (left < n && HigherPriority(m_nodes[static_cast<std::size_t>(left)], m_nodes[static_cast<std::size_t>(best)])) {
                        best = left;
                    }
                    if (right < n && HigherPriority(m_nodes[static_cast<std::size_t>(right)], m_nodes[static_cast<std::size_t>(best)])) {
                        best = right;
                    }
                    if (best == pos) {
                        break;
                    }
                    SwapNodes(pos, best);
                    pos = best;
                }
            }

            std::vector<HeapNode> m_nodes;
            std::vector<i32> m_positions;
        };

        struct TriangleEdgeProxy {
            f64 mean_strength = 0.0;
            f64 peak_strength = 0.0;
        };

        struct TriangleSafetyState {
            u32 depth = 0;
            u32 failed_attempts = 0;
            u32 low_gain_events = 0;
            u32 cooldown_until_split = 0;
            u64 region_key = 0;
        };

        struct RegionSafetyState {
            u32 cooldown_until_split = 0;
            u32 consecutive_hits = 0;
            u32 low_gain_events = 0;
            u32 last_selected_split = 0;
        };

        struct LocalErrorAccum {
            f64 weighted_error_area_sum = 0.0;
            f64 area_sum = 0.0;
        };

        void EnsureCacheSize(std::vector<PyramidTriangleCacheEntry>& cache, std::size_t n) {
            if (cache.size() < n) cache.resize(n);
        }
        void EnsureSeenSize(std::vector<u8>& seen, std::size_t n) {
            if (seen.size() < n) seen.resize(n, 0u);
        }

        void RecolorAllVerticesFromReference(Mesh& mesh, const ImageOKLaba& reference) {
            for (auto& v : mesh.vertices) {
                v.color = SampleImageOKLabaBilinear(reference, v.position);
            }
        }

        [[nodiscard]] TriangleEdgeProxy ComputeTriangleEdgeProxy(
            const Mesh& mesh,
            TriangleId ti,
            const EdgeMap& edge_map) {

            TriangleEdgeProxy out{};
            if (!edge_map.IsValid() || !mesh.IsValidTriangleId(ti)) {
                return out;
            }

            const Triangle& tri = mesh.triangles[ti];
            const Vec2& p0 = TriangleP0(mesh, tri);
            const Vec2& p1 = TriangleP1(mesh, tri);
            const Vec2& p2 = TriangleP2(mesh, tri);
            const Vec2 c = TriangleCentroid(p0, p1, p2);
            const Vec2 m01 = Midpoint(p0, p1);
            const Vec2 m12 = Midpoint(p1, p2);
            const Vec2 m20 = Midpoint(p2, p0);
            const Vec2 q0 = Midpoint(p0, c);
            const Vec2 q1 = Midpoint(p1, c);
            const Vec2 q2 = Midpoint(p2, c);

            const std::array<Vec2, 7> samples = { c, m01, m12, m20, q0, q1, q2 };
            f64 sum = 0.0;
            f64 peak = 0.0;
            for (const Vec2& s : samples) {
                const f64 v = SampleEdgeMapBilinear(edge_map, s);
                sum += v;
                peak = Max(peak, v);
            }

            out.mean_strength = sum / static_cast<f64>(samples.size());
            out.peak_strength = peak;
            return out;
        }

        [[nodiscard]] f64 ComputeEdgeBiasedSelectionScore(
            const TriangleHierarchicalError& error,
            const TriangleEdgeProxy& edge,
            const PyramidEdgeBiasOptions& options) noexcept {

            f64 score = Max(error.weighted_score, 0.0);
            if (!options.enabled || score <= 0.0) {
                return score;
            }

            const f64 mean_term = std::pow(Saturate(edge.mean_strength), Max(options.power, 1e-6));
            const f64 peak_term = std::pow(Saturate(edge.peak_strength), Max(options.power, 1e-6));

            f64 multiplier = 1.0 + options.mean_weight * mean_term + options.peak_weight * peak_term;
            if (edge.peak_strength >= options.strong_edge_threshold) {
                multiplier += options.strong_edge_bonus;
            }
            multiplier = Clamp(multiplier, 1.0, Max(options.max_multiplier, 1.0));
            return score * multiplier;
        }

        void EnsureTriangleSafetySize(std::vector<TriangleSafetyState>& states, std::size_t n) {
            if (states.size() < n) states.resize(n);
        }

        [[nodiscard]] u64 MakeRegionKey(i32 cell_x, i32 cell_y) noexcept {
            return (static_cast<u64>(static_cast<u32>(cell_y)) << 32u)
                 | static_cast<u64>(static_cast<u32>(cell_x));
        }

        [[nodiscard]] u64 RegionKeyForTriangle(const Mesh& mesh, TriangleId ti, u32 cell_px) {
            if (!mesh.IsValidTriangleId(ti)) {
                return 0;
            }
            const Triangle& tri = mesh.triangles[ti];
            const Vec2 c = TriangleCentroid(TriangleP0(mesh, tri), TriangleP1(mesh, tri), TriangleP2(mesh, tri));
            const f64 step = static_cast<f64>(Max<u32>(1u, cell_px));
            const i32 cell_x = static_cast<i32>(std::floor(Max(c.x, 0.0) / step));
            const i32 cell_y = static_cast<i32>(std::floor(Max(c.y, 0.0) / step));
            return MakeRegionKey(cell_x, cell_y);
        }

        void RefreshTriangleSafetyRegion(
            const Mesh& mesh,
            TriangleId ti,
            std::vector<TriangleSafetyState>& states,
            const PyramidRefinementOptions& options) {

            if (!options.safety.enabled || !mesh.IsValidTriangleId(ti) || ti >= states.size()) {
                return;
            }
            states[ti].region_key = RegionKeyForTriangle(mesh, ti, options.safety.hotspot_cell_px);
        }

        [[nodiscard]] f64 ComputeEffectiveSelectionScore(
            const Mesh& mesh,
            TriangleId ti,
            const PyramidTriangleCacheEntry& entry,
            std::vector<TriangleSafetyState>& states,
            std::unordered_map<u64, RegionSafetyState>& regions,
            const PyramidRefinementOptions& options,
            u32 splits_done,
            PyramidRefinementTelemetry* telemetry) {

            f64 score = entry.raw_selection_score;
            if (!options.safety.enabled || score <= 0.0) {
                return score;
            }
            if (!mesh.IsValidTriangleId(ti) || ti >= states.size()) {
                return score;
            }

            TriangleSafetyState& state = states[ti];
            state.region_key = RegionKeyForTriangle(mesh, ti, options.safety.hotspot_cell_px);
            RegionSafetyState& region = regions[state.region_key];

            if (state.depth >= options.safety.max_triangle_depth) {
                if (telemetry != nullptr) {
                    ++telemetry->safety_suppressed_by_depth;
                }
                return 0.0;
            }
            if (state.failed_attempts >= options.safety.max_failed_attempts_per_triangle) {
                if (telemetry != nullptr) {
                    ++telemetry->safety_suppressed_by_failed_attempts;
                }
                return 0.0;
            }
            if (splits_done < state.cooldown_until_split || splits_done < region.cooldown_until_split) {
                if (telemetry != nullptr) {
                    ++telemetry->safety_suppressed_by_cooldown;
                }
                return 0.0;
            }

            score /= (1.0 + options.safety.depth_penalty * static_cast<f64>(state.depth));
            score /= (1.0 + options.safety.low_gain_penalty * static_cast<f64>(state.low_gain_events));
            if (region.consecutive_hits > 1) {
                score /= (1.0 + options.safety.region_repeat_penalty * static_cast<f64>(region.consecutive_hits - 1));
            }
            return score >= options.safety.min_effective_score ? score : 0.0;
        }

        void ApplyCachedSelectionScore(
            const Mesh& mesh,
            TriangleId ti,
            std::vector<PyramidTriangleCacheEntry>& cache,
            IndexedMaxHeap& heap,
            std::vector<TriangleSafetyState>& states,
            std::unordered_map<u64, RegionSafetyState>& regions,
            const PyramidRefinementOptions& options,
            u32 splits_done,
            PyramidRefinementTelemetry* telemetry) {

            if (!mesh.IsValidTriangleId(ti) || ti >= cache.size()) {
                return;
            }
            PyramidTriangleCacheEntry& entry = cache[ti];
            if (!entry.valid) {
                heap.Remove(ti);
                return;
            }
            entry.selection_score = ComputeEffectiveSelectionScore(mesh, ti, entry, states, regions, options, splits_done, telemetry);
            if (entry.selection_score <= 0.0) {
                heap.Remove(ti);
            } else {
                heap.Upsert(ti, entry.selection_score, telemetry);
            }
        }

        [[nodiscard]] bool IsSelectableTriangle(
            const Mesh& mesh,
            TriangleId triangle_id,
            const std::vector<PyramidTriangleCacheEntry>& cache) noexcept {
            return mesh.IsValidTriangleId(triangle_id)
                && triangle_id < cache.size()
                && cache[triangle_id].valid
                && cache[triangle_id].selection_score > 0.0;
        }

        void PropagateSafetyAfterSplit(
            const Mesh& mesh,
            TriangleId selected_triangle_id,
            std::size_t old_triangle_count,
            const RefineStepResult& step,
            std::vector<TriangleSafetyState>& states,
            const PyramidRefinementOptions& options) {

            if (!options.safety.enabled) {
                return;
            }
            EnsureTriangleSafetySize(states, mesh.triangles.size());
            TriangleSafetyState parent{};
            if (selected_triangle_id < old_triangle_count && selected_triangle_id < states.size()) {
                parent = states[selected_triangle_id];
            }
            parent.depth += 1;
            parent.failed_attempts = 0;

            if (mesh.IsValidTriangleId(selected_triangle_id)) {
                states[selected_triangle_id] = parent;
                RefreshTriangleSafetyRegion(mesh, selected_triangle_id, states, options);
            }
            for (TriangleId ti = static_cast<TriangleId>(old_triangle_count); ti < mesh.triangles.size(); ++ti) {
                states[ti] = parent;
                RefreshTriangleSafetyRegion(mesh, ti, states, options);
            }
            for (TriangleId tid : step.touched_triangle_ids) {
                if (mesh.IsValidTriangleId(tid) && tid < states.size()) {
                    RefreshTriangleSafetyRegion(mesh, tid, states, options);
                }
            }
        }

        [[nodiscard]] LocalErrorAccum ComputeLocalErrorAccum(
            const std::vector<TriangleId>& triangle_ids,
            const Mesh& mesh,
            const std::vector<PyramidTriangleCacheEntry>& cache) {

            LocalErrorAccum out{};
            for (TriangleId tid : triangle_ids) {
                if (!mesh.IsValidTriangleId(tid) || tid >= cache.size()) {
                    continue;
                }
                const PyramidTriangleCacheEntry& entry = cache[tid];
                if (!entry.valid || entry.error.triangle_area <= 1e-12) {
                    continue;
                }
                out.weighted_error_area_sum += entry.error.composite_error * entry.error.triangle_area;
                out.area_sum += entry.error.triangle_area;
            }
            return out;
        }

        [[nodiscard]] f64 LocalErrorMean(const LocalErrorAccum& acc) noexcept {
            if (acc.area_sum <= 1e-12) {
                return 0.0;
            }
            return acc.weighted_error_area_sum / acc.area_sum;
        }

        void ReapplyRegionCooldown(
            const Mesh& mesh,
            u64 region_key,
            std::vector<PyramidTriangleCacheEntry>& cache,
            IndexedMaxHeap& heap,
            std::vector<TriangleSafetyState>& states,
            std::unordered_map<u64, RegionSafetyState>& regions,
            const PyramidRefinementOptions& options,
            u32 splits_done,
            PyramidRefinementTelemetry* telemetry) {

            for (TriangleId tid = 0; tid < mesh.triangles.size() && tid < cache.size() && tid < states.size(); ++tid) {
                if (!cache[tid].valid) {
                    continue;
                }
                if (states[tid].region_key != region_key) {
                    continue;
                }
                ApplyCachedSelectionScore(mesh, tid, cache, heap, states, regions, options, splits_done, telemetry);
            }
        }

        void HandleFailedSplitSafety(
            const Mesh& mesh,
            TriangleId ti,
            std::vector<PyramidTriangleCacheEntry>& cache,
            IndexedMaxHeap& heap,
            std::vector<TriangleSafetyState>& states,
            std::unordered_map<u64, RegionSafetyState>& regions,
            const PyramidRefinementOptions& options,
            u32 splits_done,
            PyramidRefinementTelemetry* telemetry) {

            if (!options.safety.enabled || !mesh.IsValidTriangleId(ti)) {
                return;
            }
            EnsureTriangleSafetySize(states, mesh.triangles.size());
            TriangleSafetyState& state = states[ti];
            ++state.failed_attempts;
            if (state.failed_attempts >= options.safety.max_failed_attempts_per_triangle) {
                state.cooldown_until_split = splits_done + options.safety.region_cooldown_splits;
                RegionSafetyState& region = regions[state.region_key];
                region.cooldown_until_split = Max(region.cooldown_until_split, state.cooldown_until_split);
                ++telemetry->safety_region_cooldowns;
                ReapplyRegionCooldown(mesh, state.region_key, cache, heap, states, regions, options, splits_done, telemetry);
            }
        }

        [[nodiscard]] bool ApplyHotspotGovernorOnSelection(
            const Mesh& mesh,
            TriangleId ti,
            std::vector<PyramidTriangleCacheEntry>& cache,
            IndexedMaxHeap& heap,
            std::vector<TriangleSafetyState>& states,
            std::unordered_map<u64, RegionSafetyState>& regions,
            const PyramidRefinementOptions& options,
            u32 splits_done,
            u64& last_region_key,
            PyramidRefinementTelemetry* telemetry) {

            if (!options.safety.enabled || !mesh.IsValidTriangleId(ti) || ti >= states.size()) {
                return false;
            }
            TriangleSafetyState& state = states[ti];
            RegionSafetyState& region = regions[state.region_key];
            if (last_region_key == state.region_key) {
                ++region.consecutive_hits;
            } else {
                region.consecutive_hits = 1;
                last_region_key = state.region_key;
            }
            region.last_selected_split = splits_done;
            if (region.consecutive_hits <= options.safety.max_consecutive_region_hits) {
                return false;
            }

            region.cooldown_until_split = splits_done + options.safety.region_cooldown_splits;
            ++telemetry->safety_region_cooldowns;
            ReapplyRegionCooldown(mesh, state.region_key, cache, heap, states, regions, options, splits_done, telemetry);
            return true;
        }

        void HandleLocalProgressAfterRefresh(
            const Mesh& mesh,
            TriangleId selected_triangle_id,
            const std::vector<TriangleId>& refresh_set,
            std::vector<PyramidTriangleCacheEntry>& cache,
            IndexedMaxHeap& heap,
            std::vector<TriangleSafetyState>& states,
            std::unordered_map<u64, RegionSafetyState>& regions,
            const PyramidRefinementOptions& options,
            u32 splits_done,
            f64 local_before,
            PyramidRefinementTelemetry* telemetry) {

            if (!options.safety.enabled || !mesh.IsValidTriangleId(selected_triangle_id) || selected_triangle_id >= states.size()) {
                return;
            }
            const f64 local_after = LocalErrorMean(ComputeLocalErrorAccum(refresh_set, mesh, cache));
            if (local_before <= 1e-12) {
                return;
            }
            const f64 improvement_ratio = (local_before - local_after) / Max(local_before, 1e-12);
            TriangleSafetyState& state = states[selected_triangle_id];
            RegionSafetyState& region = regions[state.region_key];
            if (improvement_ratio >= options.safety.min_local_error_drop_ratio) {
                state.low_gain_events = 0;
                region.low_gain_events = 0;
                return;
            }

            ++telemetry->safety_low_gain_events;
            ++state.low_gain_events;
            ++region.low_gain_events;
            if (state.low_gain_events >= options.safety.max_low_gain_events_per_triangle ||
                region.low_gain_events >= options.safety.max_low_gain_events_per_triangle) {
                const u32 cooldown = options.safety.region_cooldown_splits * (1u + region.low_gain_events);
                state.cooldown_until_split = splits_done + cooldown;
                region.cooldown_until_split = Max(region.cooldown_until_split, splits_done + cooldown);
                ++telemetry->safety_region_cooldowns;
                ReapplyRegionCooldown(mesh, state.region_key, cache, heap, states, regions, options, splits_done, telemetry);
                return;
            }

            ApplyCachedSelectionScore(mesh, selected_triangle_id, cache, heap, states, regions, options, splits_done, telemetry);
        }

        void RefreshAllScoresWithSafety(
            const Mesh& mesh,
            const std::vector<TriangleId>& refresh_set,
            std::vector<PyramidTriangleCacheEntry>& cache,
            IndexedMaxHeap& heap,
            std::vector<TriangleSafetyState>& states,
            std::unordered_map<u64, RegionSafetyState>& regions,
            const PyramidRefinementOptions& options,
            u32 splits_done,
            PyramidRefinementTelemetry* telemetry) {

            for (TriangleId tid : refresh_set) {
                ApplyCachedSelectionScore(mesh, tid, cache, heap, states, regions, options, splits_done, telemetry);
            }
        }

        void RefreshTriangleCache(
            const Mesh& mesh,
            TriangleId ti,
            const ImageOKLaba& reference,
            const ErrorPyramid& pyramid,
            const EdgeMap* edge_map,
            const PyramidRefinementOptions& options,
            const ScanlineIntegralStats* stats,
            std::vector<PyramidTriangleCacheEntry>& cache,
            IndexedMaxHeap& heap,
            std::vector<TriangleSafetyState>& states,
            std::unordered_map<u64, RegionSafetyState>& regions,
            u32 splits_done,
            f64& total_composite_error,
            PyramidRefinementTelemetry* telemetry,
            std::vector<u8>* refresh_seen) {

            if (!mesh.IsValidTriangleId(ti)) return;
            EnsureCacheSize(cache, mesh.triangles.size());
            heap.EnsureCapacity(mesh.triangles.size());
            if (telemetry) {
                ++telemetry->refresh_calls_total;
            }
            if (refresh_seen) {
                EnsureSeenSize(*refresh_seen, mesh.triangles.size());
                if ((*refresh_seen)[ti] == 0u) {
                    (*refresh_seen)[ti] = 1u;
                    if (telemetry) {
                        ++telemetry->refresh_unique_triangles;
                    }
                }
            }

            PyramidTriangleCacheEntry& entry = cache[ti];
            if (entry.valid) {
                total_composite_error -= entry.error.composite_error;
            }
            ++entry.version;
            entry.valid = true;

            const auto plane_t0 = std::chrono::high_resolution_clock::now();
            entry.plane = FitTrianglePlane(mesh, ti, reference, stats, options.plane_fit);
            const auto plane_t1 = std::chrono::high_resolution_clock::now();

            entry.error = ComputeTriangleHierarchicalSurfaceError(mesh, ti, pyramid, entry.plane, options.error);
            const auto error_t1 = std::chrono::high_resolution_clock::now();

            if (telemetry) {
                telemetry->time_plane_fit_ms += std::chrono::duration<f64, std::milli>(plane_t1 - plane_t0).count();
                telemetry->time_hier_error_ms += std::chrono::duration<f64, std::milli>(error_t1 - plane_t1).count();
            }

            entry.edge_mean_strength = 0.0;
            entry.edge_peak_strength = 0.0;
            entry.raw_selection_score = Max(entry.error.weighted_score, 0.0);

            if (edge_map != nullptr && edge_map->IsValid() && options.edge_bias.enabled) {
                const auto edge_t0 = std::chrono::high_resolution_clock::now();
                const TriangleEdgeProxy edge = ComputeTriangleEdgeProxy(mesh, ti, *edge_map);
                entry.edge_mean_strength = edge.mean_strength;
                entry.edge_peak_strength = edge.peak_strength;
                entry.raw_selection_score = ComputeEdgeBiasedSelectionScore(entry.error, edge, options.edge_bias);
                const auto edge_t1 = std::chrono::high_resolution_clock::now();
                if (telemetry) {
                    telemetry->time_edge_bias_ms += std::chrono::duration<f64, std::milli>(edge_t1 - edge_t0).count();
                    ++telemetry->edge_proxy_evaluations;
                }
            }

            total_composite_error += entry.error.composite_error;
            RefreshTriangleSafetyRegion(mesh, ti, states, options);
            ApplyCachedSelectionScore(mesh, ti, cache, heap, states, regions, options, splits_done, telemetry);
        }

        void EnsureRefreshMarkSize(std::vector<u32>& marks, std::size_t n) {
            if (marks.size() < n) marks.resize(n, 0u);
        }

        void BeginRefreshCollection(std::vector<u32>& marks, u32& stamp, std::size_t n) {
            EnsureRefreshMarkSize(marks, n);
            ++stamp;
            if (stamp == 0u) {
                std::fill(marks.begin(), marks.end(), 0u);
                stamp = 1u;
            }
        }

        void PushRefreshTriangleId(
            const Mesh& mesh,
            TriangleId ti,
            std::vector<TriangleId>& refresh_set,
            std::vector<u32>& refresh_marks,
            u32 refresh_stamp) {

            if (!mesh.IsValidTriangleId(ti) || ti >= refresh_marks.size()) {
                return;
            }
            if (refresh_marks[ti] == refresh_stamp) {
                return;
            }
            refresh_marks[ti] = refresh_stamp;
            refresh_set.push_back(ti);
        }

        void BuildRefreshSet(
            const Mesh& mesh,
            const std::vector<TriangleId>& touched_ids,
            std::vector<TriangleId>& refresh_set,
            std::vector<u32>& refresh_marks,
            u32& refresh_stamp,
            PyramidRefinementTelemetry* telemetry) {

            refresh_set.clear();
            BeginRefreshCollection(refresh_marks, refresh_stamp, mesh.triangles.size());

            for (TriangleId tid : touched_ids) {
                PushRefreshTriangleId(mesh, tid, refresh_set, refresh_marks, refresh_stamp);
            }

            if (!mesh.HasTopology()) {
                return;
            }

            const std::size_t touched_count = refresh_set.size();
            for (std::size_t i = 0; i < touched_count; ++i) {
                const TriangleId tid = refresh_set[i];
                if (!mesh.IsValidTriangleId(tid) || tid >= mesh.topology.size()) {
                    continue;
                }
                for (TriangleId n : mesh.topology[tid].neighbors) {
                    if (n == kInvalidIndex) {
                        continue;
                    }
                    const std::size_t before = refresh_set.size();
                    PushRefreshTriangleId(mesh, n, refresh_set, refresh_marks, refresh_stamp);
                    if (telemetry != nullptr && refresh_set.size() > before) {
                        ++telemetry->refresh_neighbor_calls;
                    }
                }
            }
        }

        [[nodiscard]] f64 ComputeMeanCompositeError(const Mesh& mesh, f64 total_composite_error) noexcept {
            if (mesh.triangles.empty()) {
                return 0.0;
            }
            return total_composite_error / static_cast<f64>(mesh.triangles.size());
        }

        MeshHierarchicalErrorSummary BuildSummaryFromCache(
            const Mesh& mesh,
            const std::vector<PyramidTriangleCacheEntry>& cache,
            f64 total_composite_error) {

            MeshHierarchicalErrorSummary out{};
            if (mesh.triangles.empty()) return out;
            out.mean_composite_error = ComputeMeanCompositeError(mesh, total_composite_error);
            for (TriangleId ti = 0; ti < mesh.triangles.size() && ti < cache.size(); ++ti) {
                const auto& entry = cache[ti];
                if (!entry.valid) continue;
                out.sample_count += entry.error.sample_count;
                if (entry.error.composite_error > out.max_composite_error) {
                    out.max_composite_error = entry.error.composite_error;
                    out.worst_triangle_id = ti;
                }
            }
            return out;
        }

        PyramidRefinementReport PyramidRefineMeshImpl(
            Mesh& mesh,
            const ImageOKLaba& reference,
            const EdgeMap* edge_map,
            const PyramidRefinementOptions& options) {

            if (!reference.IsValid()) {
                throw std::runtime_error("PyramidRefineMesh: reference image is invalid.");
            }

            PyramidRefinementReport report{};
            report.splits_requested = options.max_splits;
            const auto build_pyramid_t0 = std::chrono::high_resolution_clock::now();
            report.pyramid = BuildErrorPyramid(reference, options.pyramid);
            const auto build_pyramid_t1 = std::chrono::high_resolution_clock::now();
            report.telemetry.time_build_pyramid_ms = std::chrono::duration<f64, std::milli>(build_pyramid_t1 - build_pyramid_t0).count();
            const ScanlineIntegralStats scanline_stats(reference);

            if (!mesh.HasTopology()) {
                const auto topo_t0 = std::chrono::high_resolution_clock::now();
                const BuildTopologyResult topo = BuildTriangleTopology(mesh);
                const auto topo_t1 = std::chrono::high_resolution_clock::now();
                report.telemetry.time_initial_topology_ms += std::chrono::duration<f64, std::milli>(topo_t1 - topo_t0).count();
                report.telemetry.topology_rebuild_count += 1;
                if (!topo.ok) {
                    throw std::runtime_error("PyramidRefineMesh: topology build failed: " + topo.error);
                }
            }
            const auto recolor_t0 = std::chrono::high_resolution_clock::now();
            RecolorAllVerticesFromReference(mesh, reference);
            const auto recolor_t1 = std::chrono::high_resolution_clock::now();
            report.telemetry.time_recolor_vertices_ms = std::chrono::duration<f64, std::milli>(recolor_t1 - recolor_t0).count();

            std::vector<PyramidTriangleCacheEntry> cache(mesh.triangles.size());
            IndexedMaxHeap heap;
            heap.EnsureCapacity(mesh.triangles.size());
            std::vector<TriangleSafetyState> safety_states(mesh.triangles.size());
            std::unordered_map<u64, RegionSafetyState> region_states;
            region_states.reserve(mesh.triangles.size());
            std::vector<u8> refresh_seen(mesh.triangles.size(), 0u);
            std::vector<TriangleId> refresh_set;
            std::vector<u32> refresh_marks(mesh.triangles.size(), 0u);
            u32 refresh_stamp = 0u;
            f64 total_composite_error = 0.0;
            f64 best_mean_error = 0.0;
            u32 stagnant_batches = 0;
            u64 last_selected_region_key = std::numeric_limits<u64>::max();

            const auto initial_cache_t0 = std::chrono::high_resolution_clock::now();
            for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
                RefreshTriangleCache(mesh, ti, reference, report.pyramid, edge_map, options, &scanline_stats, cache, heap, safety_states, region_states, 0u, total_composite_error, &report.telemetry, &refresh_seen);
            }
            const auto initial_cache_t1 = std::chrono::high_resolution_clock::now();
            report.telemetry.initial_cache_triangle_count = mesh.triangles.size();
            report.telemetry.time_initial_cache_ms = std::chrono::duration<f64, std::milli>(initial_cache_t1 - initial_cache_t0).count();
            report.initial_error = BuildSummaryFromCache(mesh, cache, total_composite_error);
            report.final_error = report.initial_error;
            best_mean_error = report.initial_error.mean_composite_error;

            u32 splits_done = 0;
            while (splits_done < options.max_splits) {
                if (options.max_triangles > 0 && mesh.triangles.size() >= options.max_triangles) {
                    report.stop_reason = PyramidRefinementStopReason::MaxTrianglesReached;
                    break;
                }
                const f64 current_mean_error = ComputeMeanCompositeError(mesh, total_composite_error);
                report.final_error.mean_composite_error = current_mean_error;
                if (splits_done >= options.bootstrap_splits && current_mean_error <= options.target_mean_error) {
                    report.stop_reason = PyramidRefinementStopReason::TargetMeanErrorReached;
                    break;
                }

                u32 batch_splits = 0;
                bool stopped_on_min_error = false;
                const u32 batch_goal = Max<u32>(1, options.batch_size);
                while (batch_splits < batch_goal && splits_done < options.max_splits && !heap.Empty()) {
                    const HeapNode node = heap.PopMax();
                    ++report.telemetry.heap_pops_total;
                    if (!IsSelectableTriangle(mesh, node.triangle_id, cache)) {
                        ++report.stale_heap_pops;
                        continue;
                    }
                    ++report.telemetry.heap_valid_pops;
                    if (ApplyHotspotGovernorOnSelection(mesh, node.triangle_id, cache, heap, safety_states, region_states, options, splits_done, last_selected_region_key, &report.telemetry)) {
                        continue;
                    }
                    const PyramidTriangleCacheEntry& entry = cache[node.triangle_id];
                    if (entry.error.composite_error < options.min_error_to_split) {
                        report.stop_reason = PyramidRefinementStopReason::MinErrorToSplitReached;
                        stopped_on_min_error = true;
                        if (options.stop_when_heap_exhausted) {
                            while (!heap.Empty()) { (void)heap.PopMax(); }
                        }
                        break;
                    }

                    const auto split_t0 = std::chrono::high_resolution_clock::now();
                    RefineStepResult step = RefineTriangleGeometryOnly(mesh, node.triangle_id, reference, report.pyramid, options.split);
                    const auto split_t1 = std::chrono::high_resolution_clock::now();
                    report.telemetry.time_split_geometry_ms += std::chrono::duration<f64, std::milli>(split_t1 - split_t0).count();
                    if (step.topology_rebuild_performed) {
                        report.telemetry.topology_rebuild_count += 1;
                        report.telemetry.time_topology_rebuild_ms += step.topology_rebuild_ms;
                    }
                    if (!step.split_performed) {
                        ++report.blocked_split_attempts;
                        switch (step.failure_reason) {
                        case RefineStepFailureReason::SeedTriangleTooSmall: ++report.telemetry.split_rejected_seed_too_small; break;
                        case RefineStepFailureReason::SeedTriangleBBoxTooSmall: ++report.telemetry.split_rejected_bbox_too_small; break;
                        case RefineStepFailureReason::SplitEdgeTooShort: ++report.telemetry.split_rejected_edge_too_short; break;
                        case RefineStepFailureReason::SplitPointUnsafe: ++report.telemetry.split_rejected_split_point_unsafe; break;
                        case RefineStepFailureReason::NeighborChildInvalid: ++report.telemetry.split_rejected_neighbor_child_invalid; break;
                        case RefineStepFailureReason::SplitExecutionFailed: ++report.telemetry.split_rejected_split_execution_failed; break;
                        case RefineStepFailureReason::None: default: break;
                        }
                        HandleFailedSplitSafety(mesh, node.triangle_id, cache, heap, safety_states, region_states, options, splits_done, &report.telemetry);
                        const auto refresh_t0 = std::chrono::high_resolution_clock::now();
                        RefreshTriangleCache(mesh, node.triangle_id, reference, report.pyramid, edge_map, options, &scanline_stats, cache, heap, safety_states, region_states, splits_done, total_composite_error, &report.telemetry, &refresh_seen);
                        const auto refresh_t1 = std::chrono::high_resolution_clock::now();
                        report.telemetry.time_refresh_ms += std::chrono::duration<f64, std::milli>(refresh_t1 - refresh_t0).count();
                        continue;
                    }

                    ++batch_splits;
                    ++splits_done;
                    const std::size_t old_triangle_count = cache.size();
                    EnsureCacheSize(cache, mesh.triangles.size());
                    heap.EnsureCapacity(mesh.triangles.size());
                    EnsureTriangleSafetySize(safety_states, mesh.triangles.size());
                    PropagateSafetyAfterSplit(mesh, node.triangle_id, old_triangle_count, step, safety_states, options);
                    if (options.split.rebuild_topology_after_split) {
                        if (!mesh.HasTopology()) {
                            throw std::runtime_error("PyramidRefineMesh: topology unexpectedly missing after local split update.");
                        }
                    }
                    else if (!mesh.HasTopology()) {
                        const auto topo_t0 = std::chrono::high_resolution_clock::now();
                        const BuildTopologyResult topo = BuildTriangleTopology(mesh);
                        const auto topo_t1 = std::chrono::high_resolution_clock::now();
                        report.telemetry.time_topology_rebuild_ms += std::chrono::duration<f64, std::milli>(topo_t1 - topo_t0).count();
                        report.telemetry.topology_rebuild_count += 1;
                        if (!topo.ok) {
                            throw std::runtime_error("PyramidRefineMesh: topology rebuild failed: " + topo.error);
                        }
                    }
                    const auto refresh_t0 = std::chrono::high_resolution_clock::now();
                    BuildRefreshSet(mesh, step.touched_triangle_ids, refresh_set, refresh_marks, refresh_stamp, &report.telemetry);
                    const f64 local_before = LocalErrorMean(ComputeLocalErrorAccum(refresh_set, mesh, cache));
                    for (TriangleId refresh_id : refresh_set) {
                        RefreshTriangleCache(mesh, refresh_id, reference, report.pyramid, edge_map, options, &scanline_stats, cache, heap, safety_states, region_states, splits_done, total_composite_error, &report.telemetry, &refresh_seen);
                    }
                    HandleLocalProgressAfterRefresh(mesh, node.triangle_id, refresh_set, cache, heap, safety_states, region_states, options, splits_done, local_before, &report.telemetry);
                    RefreshAllScoresWithSafety(mesh, refresh_set, cache, heap, safety_states, region_states, options, splits_done, &report.telemetry);
                    const auto refresh_t1 = std::chrono::high_resolution_clock::now();
                    report.telemetry.time_refresh_ms += std::chrono::duration<f64, std::milli>(refresh_t1 - refresh_t0).count();
                    if (options.max_triangles > 0 && mesh.triangles.size() >= options.max_triangles) {
                        report.stop_reason = PyramidRefinementStopReason::MaxTrianglesReached;
                        break;
                    }
                }

                if (batch_splits == 0) {
                    if (report.stop_reason == PyramidRefinementStopReason::None) {
                        if (stopped_on_min_error) {
                            report.stop_reason = PyramidRefinementStopReason::MinErrorToSplitReached;
                        }
                        else if (heap.Empty()) {
                            report.stop_reason = PyramidRefinementStopReason::HeapExhausted;
                        }
                        else {
                            report.stop_reason = PyramidRefinementStopReason::NoSplitsPerformedInBatch;
                        }
                    }
                    break;
                }
                ++report.batches_performed;
                const f64 batch_mean_error = ComputeMeanCompositeError(mesh, total_composite_error);
                report.final_error.mean_composite_error = batch_mean_error;
                if (options.safety.enabled && options.safety.stop_on_progress_stall && splits_done >= options.bootstrap_splits) {
                    if (best_mean_error - batch_mean_error >= options.safety.min_batch_mean_error_drop) {
                        best_mean_error = batch_mean_error;
                        stagnant_batches = 0;
                    } else {
                        ++stagnant_batches;
                        ++report.telemetry.safety_progress_stall_batches;
                        if (stagnant_batches >= options.safety.stall_batch_window) {
                            report.stop_reason = PyramidRefinementStopReason::ProgressStalled;
                            break;
                        }
                    }
                }
                if (report.stop_reason == PyramidRefinementStopReason::MaxTrianglesReached) {
                    break;
                }
            }

            if (report.stop_reason == PyramidRefinementStopReason::None && splits_done >= options.max_splits) {
                report.stop_reason = PyramidRefinementStopReason::MaxSplitsReached;
            }

            report.splits_performed = splits_done;
            const auto final_planes_t0 = std::chrono::high_resolution_clock::now();
            report.final_planes = FitAllTrianglePlanes(mesh, reference, &scanline_stats, options.plane_fit);
            const auto final_planes_t1 = std::chrono::high_resolution_clock::now();
            report.telemetry.time_final_planes_ms = std::chrono::duration<f64, std::milli>(final_planes_t1 - final_planes_t0).count();
            report.final_error = ComputeMeshHierarchicalSurfaceError(mesh, report.pyramid, report.final_planes, options.error);
            const auto final_error_t1 = std::chrono::high_resolution_clock::now();
            report.telemetry.time_final_error_ms = std::chrono::duration<f64, std::milli>(final_error_t1 - final_planes_t1).count();
            return report;
        }

    } // namespace

    const char* ToString(PyramidRefinementStopReason reason) noexcept {
        switch (reason) {
        case PyramidRefinementStopReason::None: return "none";
        case PyramidRefinementStopReason::MaxSplitsReached: return "max_splits_reached";
        case PyramidRefinementStopReason::MaxTrianglesReached: return "max_triangles_reached";
        case PyramidRefinementStopReason::TargetMeanErrorReached: return "target_mean_error_reached";
        case PyramidRefinementStopReason::MinErrorToSplitReached: return "min_error_to_split_reached";
        case PyramidRefinementStopReason::HeapExhausted: return "heap_exhausted";
        case PyramidRefinementStopReason::NoSplitsPerformedInBatch: return "no_splits_performed_in_batch";
        case PyramidRefinementStopReason::ProgressStalled: return "progress_stalled";
        default: return "unknown";
        }
    }

    PyramidRefinementReport PyramidRefineMesh(
        Mesh& mesh,
        const ImageOKLaba& reference,
        const PyramidRefinementOptions& options) {
        return PyramidRefineMeshImpl(mesh, reference, nullptr, options);
    }

    PyramidRefinementReport PyramidRefineMesh(
        Mesh& mesh,
        const ImageOKLaba& reference,
        const EdgeMap& edge_map,
        const PyramidRefinementOptions& options) {
        return PyramidRefineMeshImpl(mesh, reference, &edge_map, options);
    }

} // namespace svec
