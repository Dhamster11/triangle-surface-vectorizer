#include "svec/surface/mesh_topology.h"

#include <array>
#include <unordered_map>
#include <utility>
#include <vector>

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

struct EdgeRef {
    TriangleId triangle = kInvalidIndex;
    i32 opposite_vertex_index = -1;
};

[[nodiscard]] std::array<EdgeKey, 3> MakeTriangleEdgeKeys(const Triangle& tri) noexcept {
    return {
        EdgeKey{tri.v[1], tri.v[2]},
        EdgeKey{tri.v[0], tri.v[2]},
        EdgeKey{tri.v[0], tri.v[1]}
    };
}

[[nodiscard]] bool ContainsTriangleId(const std::vector<TriangleId>& ids, TriangleId id) noexcept {
    for (TriangleId value : ids) {
        if (value == id) {
            return true;
        }
    }
    return false;
}

void PushUniqueTriangleId(std::vector<TriangleId>& ids, TriangleId id) {
    if (id == kInvalidIndex) {
        return;
    }
    if (!ContainsTriangleId(ids, id)) {
        ids.push_back(id);
    }
}

[[nodiscard]] bool FindSharedEdge(const Triangle& a, const Triangle& b, VertexId& out_u, VertexId& out_v) noexcept {
    VertexId shared[2]{kInvalidIndex, kInvalidIndex};
    i32 count = 0;
    for (VertexId va : a.v) {
        for (VertexId vb : b.v) {
            if (va == vb) {
                if (count < 2) {
                    shared[count] = va;
                }
                ++count;
                break;
            }
        }
    }
    if (count != 2) {
        return false;
    }
    out_u = shared[0];
    out_v = shared[1];
    return true;
}

[[nodiscard]] TriangleId FindTriangleContainingEdge(
    const Mesh& mesh,
    const std::vector<TriangleId>& candidates,
    VertexId a,
    VertexId b) noexcept {

    for (TriangleId tid : candidates) {
        if (!mesh.IsValidTriangleId(tid)) {
            continue;
        }
        if (FindLocalOppositeVertexIndex(mesh.triangles[tid], a, b) >= 0) {
            return tid;
        }
    }
    return kInvalidIndex;
}

void SetBidirectionalNeighbor(
    Mesh& mesh,
    TriangleId a,
    TriangleId b,
    VertexId edge_u,
    VertexId edge_v,
    BuildTopologyResult& result) {

    const i32 a_local = FindLocalOppositeVertexIndex(mesh.triangles[a], edge_u, edge_v);
    const i32 b_local = FindLocalOppositeVertexIndex(mesh.triangles[b], edge_u, edge_v);
    if (a_local < 0 || b_local < 0) {
        result.ok = false;
        result.error = "UpdateTopologyAfterLocalEdgeSplit: failed to locate shared edge in local pair.";
        return;
    }

    mesh.topology[a].neighbors[static_cast<std::size_t>(a_local)] = b;
    mesh.topology[b].neighbors[static_cast<std::size_t>(b_local)] = a;
}

void AttachExternalNeighbor(
    Mesh& mesh,
    const std::vector<TriangleId>& affected,
    TriangleId child_id,
    VertexId edge_u,
    VertexId edge_v,
    TriangleId external_neighbor,
    BuildTopologyResult& result) {

    if (!mesh.IsValidTriangleId(child_id)) {
        result.ok = false;
        result.error = "UpdateTopologyAfterLocalEdgeSplit: child triangle id is invalid.";
        return;
    }

    const i32 child_local = FindLocalOppositeVertexIndex(mesh.triangles[child_id], edge_u, edge_v);
    if (child_local < 0) {
        result.ok = false;
        result.error = "UpdateTopologyAfterLocalEdgeSplit: child triangle does not contain perimeter edge.";
        return;
    }

    mesh.topology[child_id].neighbors[static_cast<std::size_t>(child_local)] = external_neighbor;

    if (external_neighbor == kInvalidIndex || ContainsTriangleId(affected, external_neighbor)) {
        return;
    }
    if (!mesh.IsValidTriangleId(external_neighbor)) {
        result.ok = false;
        result.error = "UpdateTopologyAfterLocalEdgeSplit: external neighbor triangle id is invalid.";
        return;
    }

    const i32 ext_local = FindLocalOppositeVertexIndex(mesh.triangles[external_neighbor], edge_u, edge_v);
    if (ext_local < 0) {
        result.ok = false;
        result.error = "UpdateTopologyAfterLocalEdgeSplit: external neighbor no longer contains perimeter edge.";
        return;
    }

    if (external_neighbor >= mesh.topology.size()) {
        mesh.topology.resize(mesh.triangles.size());
    }
    mesh.topology[external_neighbor].neighbors[static_cast<std::size_t>(ext_local)] = child_id;
}

void TransferPerimeterNeighbors(
    Mesh& mesh,
    const std::vector<TriangleId>& affected,
    const Triangle& before,
    const TriangleNeighbors& before_neighbors,
    VertexId split_edge_a,
    VertexId split_edge_b,
    BuildTopologyResult& result) {

    if (!result.ok) {
        return;
    }

    const i32 split_local = FindLocalOppositeVertexIndex(before, split_edge_a, split_edge_b);
    if (split_local < 0) {
        result.ok = false;
        result.error = "UpdateTopologyAfterLocalEdgeSplit: split edge is not present in pre-split triangle.";
        return;
    }

    const VertexId opposite = before.v[static_cast<std::size_t>(split_local)];

    const std::array<std::pair<VertexId, VertexId>, 2> perimeter_edges = {
        std::pair<VertexId, VertexId>{opposite, split_edge_a},
        std::pair<VertexId, VertexId>{opposite, split_edge_b}
    };

    for (const auto& [u, v] : perimeter_edges) {
        const i32 before_local = FindLocalOppositeVertexIndex(before, u, v);
        if (before_local < 0) {
            result.ok = false;
            result.error = "UpdateTopologyAfterLocalEdgeSplit: failed to locate pre-split perimeter edge.";
            return;
        }

        const TriangleId child = FindTriangleContainingEdge(mesh, affected, u, v);
        if (child == kInvalidIndex) {
            result.ok = false;
            result.error = "UpdateTopologyAfterLocalEdgeSplit: failed to map perimeter edge onto child triangle.";
            return;
        }

        const TriangleId external = before_neighbors.neighbors[static_cast<std::size_t>(before_local)];
        AttachExternalNeighbor(mesh, affected, child, u, v, external, result);
        if (!result.ok) {
            return;
        }
    }
}

} // namespace

i32 FindLocalOppositeVertexIndex(const Triangle& tri, VertexId a, VertexId b) noexcept {
    for (i32 i = 0; i < 3; ++i) {
        const VertexId v0 = tri.v[(i + 1) % 3];
        const VertexId v1 = tri.v[(i + 2) % 3];
        if ((v0 == a && v1 == b) || (v0 == b && v1 == a)) {
            return i;
        }
    }
    return -1;
}

BuildTopologyResult BuildTriangleTopology(Mesh& mesh) {
    BuildTopologyResult result{};

    std::string error;
    if (!ValidateMeshGeometry(mesh, &error)) {
        result.error = error;
        return result;
    }

    mesh.topology.assign(mesh.triangles.size(), {});

    std::unordered_map<EdgeKey, std::vector<EdgeRef>, EdgeKeyHasher> edge_map;
    edge_map.reserve(mesh.triangles.size() * 3);

    for (TriangleId ti = 0; ti < mesh.triangles.size(); ++ti) {
        const Triangle& tri = mesh.triangles[ti];
        const auto keys = MakeTriangleEdgeKeys(tri);
        for (i32 local_opposite = 0; local_opposite < 3; ++local_opposite) {
            edge_map[keys[static_cast<std::size_t>(local_opposite)]].push_back(
                EdgeRef{ti, local_opposite});
        }
    }

    for (const auto& [key, refs] : edge_map) {
        if (refs.size() == 1) {
            continue; // boundary edge
        }

        if (refs.size() > 2) {
            result.has_non_manifold_edges = true;
            continue;
        }

        const EdgeRef a = refs[0];
        const EdgeRef b = refs[1];

        mesh.topology[a.triangle].neighbors[static_cast<std::size_t>(a.opposite_vertex_index)] = b.triangle;
        mesh.topology[b.triangle].neighbors[static_cast<std::size_t>(b.opposite_vertex_index)] = a.triangle;
    }

    if (result.has_non_manifold_edges) {
        result.error = "Topology built, but non-manifold edges were detected.";
    }

    result.ok = true;
    return result;
}

BuildTopologyResult UpdateTopologyAfterLocalEdgeSplit(Mesh& mesh, const LocalEdgeSplitTopologyUpdate& update) {
    BuildTopologyResult result{};
    result.ok = true;

    if (!mesh.IsValidTriangleId(update.seed_triangle_id) || !mesh.IsValidTriangleId(update.seed_new_triangle_id)) {
        result.ok = false;
        result.error = "UpdateTopologyAfterLocalEdgeSplit: seed child triangle ids are invalid.";
        return result;
    }
    if (update.split_edge_a == kInvalidIndex || update.split_edge_b == kInvalidIndex || update.split_vertex == kInvalidIndex) {
        result.ok = false;
        result.error = "UpdateTopologyAfterLocalEdgeSplit: split edge description is incomplete.";
        return result;
    }
    if (update.neighbor_triangle_id != kInvalidIndex && !mesh.IsValidTriangleId(update.neighbor_triangle_id)) {
        result.ok = false;
        result.error = "UpdateTopologyAfterLocalEdgeSplit: seed neighbor triangle id is invalid.";
        return result;
    }
    if (update.neighbor_triangle_id != kInvalidIndex && !mesh.IsValidTriangleId(update.neighbor_new_triangle_id)) {
        result.ok = false;
        result.error = "UpdateTopologyAfterLocalEdgeSplit: neighbor child triangle id is invalid.";
        return result;
    }

    if (mesh.topology.size() < mesh.triangles.size()) {
        mesh.topology.resize(mesh.triangles.size());
    }

    std::vector<TriangleId> affected;
    affected.reserve(update.neighbor_triangle_id != kInvalidIndex ? 4 : 2);
    PushUniqueTriangleId(affected, update.seed_triangle_id);
    PushUniqueTriangleId(affected, update.seed_new_triangle_id);
    PushUniqueTriangleId(affected, update.neighbor_triangle_id);
    PushUniqueTriangleId(affected, update.neighbor_new_triangle_id);

    for (TriangleId tid : affected) {
        mesh.topology[tid] = TriangleNeighbors{};
    }

    for (std::size_t i = 0; i < affected.size(); ++i) {
        for (std::size_t j = i + 1; j < affected.size(); ++j) {
            const TriangleId a = affected[i];
            const TriangleId b = affected[j];
            VertexId edge_u = kInvalidIndex;
            VertexId edge_v = kInvalidIndex;
            if (!FindSharedEdge(mesh.triangles[a], mesh.triangles[b], edge_u, edge_v)) {
                continue;
            }
            SetBidirectionalNeighbor(mesh, a, b, edge_u, edge_v, result);
            if (!result.ok) {
                return result;
            }
        }
    }

    TransferPerimeterNeighbors(
        mesh,
        affected,
        update.seed_triangle_before,
        update.seed_neighbors_before,
        update.split_edge_a,
        update.split_edge_b,
        result);
    if (!result.ok) {
        return result;
    }

    if (update.neighbor_triangle_id != kInvalidIndex) {
        TransferPerimeterNeighbors(
            mesh,
            affected,
            update.neighbor_triangle_before,
            update.neighbor_neighbors_before,
            update.split_edge_a,
            update.split_edge_b,
            result);
        if (!result.ok) {
            return result;
        }
    }

    return result;
}

} // namespace svec
