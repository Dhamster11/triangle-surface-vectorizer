#include "svec/surface/mesh_builders.h"

namespace svec {

namespace {
[[nodiscard]] VertexId GridIndex(u32 ix, u32 iy, u32 verts_x) noexcept {
    return static_cast<VertexId>(iy * verts_x + ix);
}
}

Mesh CreateImageQuadMesh(const ImageSize& size, const ColorOKLaba& fill) {
    Mesh mesh;
    if (!size.IsValid()) {
        return mesh;
    }

    const f64 max_x = static_cast<f64>(size.width - 1);
    const f64 max_y = static_cast<f64>(size.height - 1);

    mesh.vertices = {
        Vertex{Vec2{0.0,   0.0},   fill},
        Vertex{Vec2{max_x, 0.0},   fill},
        Vertex{Vec2{max_x, max_y}, fill},
        Vertex{Vec2{0.0,   max_y}, fill}
    };

    mesh.triangles = {
        Triangle{{0, 1, 2}},
        Triangle{{0, 2, 3}}
    };

    return mesh;
}

Mesh CreateImageGridMesh(const ImageSize& size, u32 cells_x, u32 cells_y, const ColorOKLaba& fill) {
    Mesh mesh;
    if (!size.IsValid()) {
        return mesh;
    }

    cells_x = cells_x == 0 ? 1 : cells_x;
    cells_y = cells_y == 0 ? 1 : cells_y;

    const u32 verts_x = cells_x + 1;
    const u32 verts_y = cells_y + 1;
    const f64 max_x = static_cast<f64>(size.width - 1);
    const f64 max_y = static_cast<f64>(size.height - 1);

    mesh.vertices.reserve(static_cast<std::size_t>(verts_x) * static_cast<std::size_t>(verts_y));
    for (u32 iy = 0; iy < verts_y; ++iy) {
        const f64 fy = cells_y > 0 ? static_cast<f64>(iy) / static_cast<f64>(cells_y) : 0.0;
        const f64 py = fy * max_y;
        for (u32 ix = 0; ix < verts_x; ++ix) {
            const f64 fx = cells_x > 0 ? static_cast<f64>(ix) / static_cast<f64>(cells_x) : 0.0;
            const f64 px = fx * max_x;
            mesh.vertices.push_back(Vertex{Vec2{px, py}, fill});
        }
    }

    mesh.triangles.reserve(static_cast<std::size_t>(cells_x) * static_cast<std::size_t>(cells_y) * 2u);
    for (u32 iy = 0; iy < cells_y; ++iy) {
        for (u32 ix = 0; ix < cells_x; ++ix) {
            const VertexId v00 = GridIndex(ix + 0, iy + 0, verts_x);
            const VertexId v10 = GridIndex(ix + 1, iy + 0, verts_x);
            const VertexId v01 = GridIndex(ix + 0, iy + 1, verts_x);
            const VertexId v11 = GridIndex(ix + 1, iy + 1, verts_x);

            if (((ix + iy) & 1u) == 0u) {
                mesh.triangles.push_back(Triangle{{v00, v10, v11}});
                mesh.triangles.push_back(Triangle{{v00, v11, v01}});
            } else {
                mesh.triangles.push_back(Triangle{{v00, v10, v01}});
                mesh.triangles.push_back(Triangle{{v10, v11, v01}});
            }
        }
    }

    return mesh;
}

} // namespace svec
