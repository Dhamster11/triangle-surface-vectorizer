#include <cassert>
#include <cmath>
#include <iostream>

#include "svec/error/error_estimator.h"
#include "svec/refine/adaptive_refinement.h"
#include "svec/surface/mesh_builders.h"
#include "svec/surface/mesh_topology.h"

namespace {

svec::ImageOKLaba MakeQuadraticReference(svec::ImageSize size) {
    svec::ImageOKLaba img(size);

    const double w = static_cast<double>(size.width - 1);
    const double h = static_cast<double>(size.height - 1);

    for (int y = 0; y < size.height; ++y) {
        for (int x = 0; x < size.width; ++x) {
            const double nx = static_cast<double>(x) / w;
            const double ny = static_cast<double>(y) / h;
            img.At(x, y) = {
                0.10 + 0.75 * nx * nx,
                -0.08 + 0.16 * ny,
                0.06 * std::sin(nx * 3.14159265358979323846),
                1.0
            };
        }
    }
    return img;
}

void InitializeMeshVertexColorsFromReference(svec::Mesh& mesh, const svec::ImageOKLaba& ref) {
    for (auto& v : mesh.vertices) {
        v.color = svec::SampleImageOKLabaNearest(ref, v.position);
    }
}

} // namespace

int main() {
    using namespace svec;

    const ImageSize size{24, 24};
    const ImageOKLaba reference = MakeQuadraticReference(size);

    Mesh mesh = CreateImageQuadMesh(size);
    InitializeMeshVertexColorsFromReference(mesh, reference);

    const BuildTopologyResult topo = BuildTriangleTopology(mesh);
    assert(topo.ok);

    const ErrorEstimatorOptions error_opts{};
    const MeshErrorSummary initial_error = ComputeMeshError(mesh, reference, error_opts).summary;
    assert(initial_error.worst_triangle_id != kInvalidIndex);
    assert(initial_error.weighted_rmse > 0.0);

    const std::size_t initial_vertices = mesh.vertices.size();
    const std::size_t initial_triangles = mesh.triangles.size();

    const SingleRefineStepOptions step_opts{};
    const RefineStepResult one_step = RefineWorstTriangleOnce(mesh, reference, step_opts);
    assert(one_step.split_performed);
    assert(mesh.vertices.size() == initial_vertices + 1);
    assert(mesh.triangles.size() >= initial_triangles + 1);
    assert(one_step.error_after.weighted_rmse < one_step.error_before.weighted_rmse);

    AdaptiveRefinementOptions adaptive_opts{};
    adaptive_opts.max_iterations = 8;
    adaptive_opts.target_weighted_rmse = 0.0;

    AdaptiveRefinementReport report = AdaptiveRefineMesh(mesh, reference, adaptive_opts);
    assert(report.IterationsPerformed() > 0);
    assert(report.final_error.weighted_rmse < initial_error.weighted_rmse);
    assert(mesh.vertices.size() > initial_vertices);
    assert(mesh.triangles.size() > initial_triangles);

    std::cout << "step6_smoke ok\n";
    std::cout << "initial weighted_rmse: " << initial_error.weighted_rmse << "\n";
    std::cout << "final weighted_rmse:   " << report.final_error.weighted_rmse << "\n";
    std::cout << "vertices: " << mesh.vertices.size() << ", triangles: " << mesh.triangles.size() << "\n";
    return 0;
}
