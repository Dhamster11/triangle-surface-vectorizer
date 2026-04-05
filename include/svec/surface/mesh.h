#pragma once

#include <array>
#include <limits>
#include <string>
#include <vector>

#include "svec/core/types.h"
#include "svec/math/color.h"
#include "svec/math/geometry.h"
#include "svec/math/vec2.h"

namespace svec {

using VertexId = u32;
using TriangleId = u32;

inline constexpr u32 kInvalidIndex = std::numeric_limits<u32>::max();

struct Vertex {
    Vec2 position{};
    ColorOKLaba color{};
};

struct Triangle {
    std::array<VertexId, 3> v{kInvalidIndex, kInvalidIndex, kInvalidIndex};

    [[nodiscard]] constexpr VertexId operator[](i32 i) const noexcept {
        return v[static_cast<std::size_t>(i)];
    }

    [[nodiscard]] bool IsIndexed() const noexcept {
        return v[0] != kInvalidIndex && v[1] != kInvalidIndex && v[2] != kInvalidIndex;
    }

    [[nodiscard]] bool HasDuplicateVertices() const noexcept {
        return v[0] == v[1] || v[0] == v[2] || v[1] == v[2];
    }
};

struct TriangleNeighbors {
    // neighbors[i] lies across the edge opposite local vertex i:
    // i=0 => edge (v1, v2)
    // i=1 => edge (v0, v2)
    // i=2 => edge (v0, v1)
    std::array<TriangleId, 3> neighbors{kInvalidIndex, kInvalidIndex, kInvalidIndex};

    [[nodiscard]] constexpr TriangleId operator[](i32 i) const noexcept {
        return neighbors[static_cast<std::size_t>(i)];
    }
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
    std::vector<TriangleNeighbors> topology;

    [[nodiscard]] bool Empty() const noexcept {
        return vertices.empty() || triangles.empty();
    }

    [[nodiscard]] bool HasTopology() const noexcept {
        return topology.size() == triangles.size();
    }

    void Clear() {
        vertices.clear();
        triangles.clear();
        topology.clear();
    }

    void ClearTopology() {
        topology.clear();
    }

    [[nodiscard]] bool IsValidVertexId(VertexId id) const noexcept {
        return id < vertices.size();
    }

    [[nodiscard]] bool IsValidTriangleId(TriangleId id) const noexcept {
        return id < triangles.size();
    }

    [[nodiscard]] const Vertex& GetVertex(VertexId id) const {
        return vertices.at(id);
    }

    [[nodiscard]] const Triangle& GetTriangle(TriangleId id) const {
        return triangles.at(id);
    }

    [[nodiscard]] Vertex& GetVertex(VertexId id) {
        return vertices.at(id);
    }

    [[nodiscard]] Triangle& GetTriangle(TriangleId id) {
        return triangles.at(id);
    }
};

[[nodiscard]] inline const Vec2& TriangleP0(const Mesh& mesh, const Triangle& tri) {
    return mesh.vertices.at(tri.v[0]).position;
}

[[nodiscard]] inline const Vec2& TriangleP1(const Mesh& mesh, const Triangle& tri) {
    return mesh.vertices.at(tri.v[1]).position;
}

[[nodiscard]] inline const Vec2& TriangleP2(const Mesh& mesh, const Triangle& tri) {
    return mesh.vertices.at(tri.v[2]).position;
}

[[nodiscard]] inline const ColorOKLaba& TriangleC0(const Mesh& mesh, const Triangle& tri) {
    return mesh.vertices.at(tri.v[0]).color;
}

[[nodiscard]] inline const ColorOKLaba& TriangleC1(const Mesh& mesh, const Triangle& tri) {
    return mesh.vertices.at(tri.v[1]).color;
}

[[nodiscard]] inline const ColorOKLaba& TriangleC2(const Mesh& mesh, const Triangle& tri) {
    return mesh.vertices.at(tri.v[2]).color;
}

[[nodiscard]] inline f64 ComputeTriangleArea(const Mesh& mesh, const Triangle& tri) {
    return TriangleArea(TriangleP0(mesh, tri), TriangleP1(mesh, tri), TriangleP2(mesh, tri));
}

[[nodiscard]] inline bool IsDegenerate(const Mesh& mesh, const Triangle& tri, f64 epsilon = kEpsilon) {
    return IsDegenerateTriangle(TriangleP0(mesh, tri), TriangleP1(mesh, tri), TriangleP2(mesh, tri), epsilon);
}

[[nodiscard]] bool ValidateMeshIndices(const Mesh& mesh, std::string* error = nullptr);
[[nodiscard]] bool ValidateMeshGeometry(const Mesh& mesh, std::string* error = nullptr, f64 epsilon = kEpsilon);

} // namespace svec
