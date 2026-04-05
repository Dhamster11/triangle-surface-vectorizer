#pragma once

#include <string>

#include "svec/surface/mesh.h"

namespace svec {

struct BuildTopologyResult {
    bool ok = false;
    bool has_non_manifold_edges = false;
    std::string error;
};

struct LocalEdgeSplitTopologyUpdate {
    TriangleId seed_triangle_id = kInvalidIndex;
    TriangleId seed_new_triangle_id = kInvalidIndex;

    TriangleId neighbor_triangle_id = kInvalidIndex;
    TriangleId neighbor_new_triangle_id = kInvalidIndex;

    Triangle seed_triangle_before{};
    TriangleNeighbors seed_neighbors_before{};

    Triangle neighbor_triangle_before{};
    TriangleNeighbors neighbor_neighbors_before{};

    VertexId split_edge_a = kInvalidIndex;
    VertexId split_edge_b = kInvalidIndex;
    VertexId split_vertex = kInvalidIndex;
};

[[nodiscard]] BuildTopologyResult BuildTriangleTopology(Mesh& mesh);
[[nodiscard]] BuildTopologyResult UpdateTopologyAfterLocalEdgeSplit(Mesh& mesh, const LocalEdgeSplitTopologyUpdate& update);
[[nodiscard]] i32 FindLocalOppositeVertexIndex(const Triangle& tri, VertexId a, VertexId b) noexcept;

} // namespace svec
