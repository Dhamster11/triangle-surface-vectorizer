#include <cassert>
#include <cmath>
#include <iostream>

#include "svec/edge/edge_map.h"
#include "svec/error/error_estimator.h"
#include "svec/refine/edge_aware_refinement.h"
#include "svec/surface/mesh_builders.h"

using namespace svec;

int main() {
    ImageOKLaba reference({64, 64}, {0.10, 0.0, 0.0, 1.0});
    for (i32 y = 0; y < reference.Height(); ++y) {
        for (i32 x = 0; x < reference.Width(); ++x) {
            if (x > y) {
                reference.At(x, y) = {0.85, 0.05, -0.03, 1.0};
            }
        }
    }

    EdgeMap edge_map = ComputeEdgeMapSobel(reference);
    assert(edge_map.IsValid());
    assert(edge_map.MaxValue() > 0.0);

    Mesh mesh = CreateImageQuadMesh(reference.Size(), {0.0, 0.0, 0.0, 1.0});
    for (auto& v : mesh.vertices) {
        v.color = SampleImageOKLabaBilinear(reference, v.position);
    }

    ErrorEstimatorOptions error_opts{};
    const MeshErrorSummary initial_error = ComputeMeshError(mesh, reference, error_opts).summary;

    EdgeAwareRefinementOptions opts{};
    opts.max_iterations = 12;
    opts.stop_when_no_improvement = true;
    opts.step.error_options = error_opts;
    opts.select.error_options = error_opts;
    opts.select.edge_weight = 4.0;
    opts.select.strong_edge_bonus = 0.5;

    EdgeAwareRefinementReport report = EdgeAwareRefineMesh(mesh, reference, opts);

    std::cout << "initial weighted_rmse: " << initial_error.weighted_rmse << "\n";
    std::cout << "final weighted_rmse:   " << report.final_error.weighted_rmse << "\n";
    std::cout << "iterations:            " << report.IterationsPerformed() << "\n";

    assert(report.edge_map.IsValid());
    assert(!report.selections.empty());
    assert(!report.steps.empty());
    assert(report.final_error.weighted_rmse < initial_error.weighted_rmse);

    const auto& first_sel = report.selections.front();
    assert(first_sel.selected_triangle_id != kInvalidIndex);
    assert(first_sel.selected_score > 0.0);

    return 0;
}
