#include <cassert>
#include <iostream>

#include "svec/core/types.h"
#include "svec/image/image.h"
#include "svec/render/reference_renderer.h"
#include "svec/surface/mesh_builders.h"

using namespace svec;

int main() {
    const ImageSize size{7, 7};
    Mesh mesh = CreateImageQuadMesh(size, {});

    assert(mesh.vertices.size() == 4);
    assert(mesh.triangles.size() == 2);

    mesh.vertices[0].color = {0.20, -0.05, -0.05, 1.0};
    mesh.vertices[1].color = {0.90,  0.10, -0.05, 1.0};
    mesh.vertices[2].color = {0.80,  0.00,  0.08, 1.0};
    mesh.vertices[3].color = {0.30, -0.10,  0.12, 1.0};

    const ReferenceRenderResult preview = RenderMeshReference(mesh, size);
    assert(preview.image.IsValid());
    assert(preview.stats.triangles_rasterized == 2);
    assert(preview.stats.triangles_skipped_degenerate == 0);
    assert(preview.stats.pixels_shaded > 0);

    const ColorOKLaba center = preview.image.At(3, 3);
    assert(center.alpha > 0.99);
    assert(center.L > 0.2 && center.L < 0.95);

    ReferenceRenderOptions debug_options{};
    debug_options.mode = ReferenceRenderMode::TriangleIdFlat;
    debug_options.overlay_wireframe = true;
    debug_options.wire_half_width_px = 0.55;

    const ReferenceRenderResult debug = RenderMeshReference(mesh, size, debug_options);
    assert(debug.image.IsValid());
    assert(debug.stats.wire_pixels_shaded > 0);

    std::cout << "step4_smoke OK\n";
    return 0;
}
