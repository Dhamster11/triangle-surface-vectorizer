#include <cassert>
#include <iostream>

#include "svec/error/error_estimator.h"
#include "svec/image/image.h"
#include "svec/render/reference_renderer.h"
#include "svec/surface/mesh_builders.h"

using namespace svec;

int main() {
    const ImageSize size{12, 8};

    Mesh mesh = CreateImageQuadMesh(size, {});
    mesh.vertices[0].color = {0.20, -0.05, -0.04, 1.0};
    mesh.vertices[1].color = {0.85,  0.10, -0.05, 1.0};
    mesh.vertices[2].color = {0.75,  0.02,  0.08, 1.0};
    mesh.vertices[3].color = {0.30, -0.11,  0.10, 1.0};

    const auto render = RenderMeshReference(mesh, size);
    assert(render.image.IsValid());

    const MeshErrorReport perfect = ComputeMeshError(mesh, render.image);
    assert(perfect.summary.sample_count_total > 0);
    assert(perfect.summary.weighted_rmse < 1e-9);
    assert(perfect.summary.worst_triangle_id != kInvalidIndex);
    assert(perfect.per_triangle.size() == mesh.triangles.size());

    ImageOKLaba modified = render.image;
    modified.At(2, 2).L += 0.20;
    modified.At(2, 2).a += 0.03;
    modified.At(2, 2).alpha -= 0.15;

    ErrorEstimatorOptions options{};
    options.samples_per_axis = 1;
    options.alpha_weight = 2.0;

    const MeshErrorReport disturbed = ComputeMeshError(mesh, modified, options);
    assert(disturbed.summary.weighted_rmse > 0.0);
    assert(disturbed.summary.worst_triangle_id != kInvalidIndex);
    assert(disturbed.summary.worst_triangle_weighted_rmse > 0.0);

    const ImageOKLaba heat = RenderTriangleErrorHeatmap(mesh, size, disturbed.per_triangle);
    assert(heat.IsValid());
    assert(heat.Width() == size.width);
    assert(heat.Height() == size.height);

    const TriangleErrorMetrics tri0 = ComputeTriangleError(mesh, 0, modified, options);
    const TriangleErrorMetrics tri1 = ComputeTriangleError(mesh, 1, modified, options);
    assert(tri0.HasSamples() || tri1.HasSamples());

    std::cout << "step5_smoke OK\n";
    return 0;
}
