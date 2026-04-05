#include <cassert>
#include <iostream>

#include "svec/edge/edge_map.h"
#include "svec/error/error_estimator.h"
#include "svec/meshopt/mesh_optimization.h"
#include "svec/refine/edge_aware_refinement.h"
#include "svec/surface/mesh_builders.h"

using namespace svec;

int main() {
    ImageOKLaba reference({96, 72}, {0.0, 0.0, 0.0, 1.0});
    for (i32 y = 0; y < reference.Height(); ++y) {
        for (i32 x = 0; x < reference.Width(); ++x) {
            const f64 fx = static_cast<f64>(x) / static_cast<f64>(reference.Width() - 1);
            const f64 fy = static_cast<f64>(y) / static_cast<f64>(reference.Height() - 1);
            ColorOKLaba c{};
            c.L = 0.18 + 0.62 * fx;
            c.a = -0.05 + 0.12 * fy;
            c.b = 0.03 + 0.05 * fx;
            c.alpha = 1.0;
            if (x > 20 && x < 80 && y > 12 && y < 58 && x > y) {
                c.L += 0.10;
                c.a += 0.08;
                c.b -= 0.06;
            }
            if ((x - 48) * (x - 48) + (y - 36) * (y - 36) < 11 * 11) {
                c.L = 0.87;
                c.a = -0.02;
                c.b = 0.14;
            }
            reference.At(x, y) = c;
        }
    }

    Mesh mesh = CreateImageQuadMesh(reference.Size(), {0.0, 0.0, 0.0, 1.0});
    for (auto& v : mesh.vertices) {
        v.color = SampleImageOKLabaBilinear(reference, v.position);
    }

    EdgeAwareRefinementOptions refine_opts{};
    refine_opts.max_iterations = 20;
    refine_opts.stop_when_no_improvement = false;
    refine_opts.select.edge_weight = 3.5;
    refine_opts.select.strong_edge_bonus = 0.4;
    refine_opts.select.mean_edge_bonus = 0.25;
    refine_opts.step.error_options.samples_per_axis = 2;
    refine_opts.select.error_options.samples_per_axis = 2;

    EdgeAwareRefinementReport refined = EdgeAwareRefineMesh(mesh, reference, refine_opts);
    assert(refined.final_error.weighted_rmse < refined.initial_error.weighted_rmse);

    EdgeMap edge_map = ComputeEdgeMapSobel(reference);
    assert(edge_map.IsValid());

    const MeshErrorSummary before_error = ComputeMeshError(mesh, reference, refine_opts.step.error_options).summary;
    const f64 before_min_angle = ComputeMeshMinTriangleAngleDegrees(mesh, true);
    const f64 before_mean_angle = ComputeMeshMeanTriangleMinAngleDegrees(mesh, true);

    MeshOptimizationOptions opt{};
    opt.error_options = refine_opts.step.error_options;
    opt.outer_iterations = 2;
    opt.stop_when_no_changes = true;
    opt.flip.enabled = true;
    opt.flip.min_quality_improvement_degrees = 0.05;
    opt.flip.max_local_error_increase_ratio = 0.04;
    opt.flip.strong_edge_threshold = 0.45;
    opt.smooth.enabled = true;
    opt.smooth.iterations = 2;
    opt.smooth.lambda = 0.30;
    opt.smooth.max_move_distance = 2.0;
    opt.smooth.max_local_error_increase_ratio = 0.05;
    opt.smooth.strong_edge_threshold = 0.30;

    MeshOptimizationReport report = OptimizeMesh(mesh, reference, opt, &edge_map);

    std::cout << "before error:      " << before_error.weighted_rmse << "\n";
    std::cout << "after error:       " << report.final_error.weighted_rmse << "\n";
    std::cout << "before min angle:  " << before_min_angle << "\n";
    std::cout << "after min angle:   " << report.final_min_triangle_angle_deg << "\n";
    std::cout << "before mean angle: " << before_mean_angle << "\n";
    std::cout << "after mean angle:  " << report.final_mean_min_angle_deg << "\n";
    std::cout << "flips applied:     " << report.TotalFlipsApplied() << "\n";
    std::cout << "vertex moves:      " << report.TotalVertexMovesApplied() << "\n";

    assert(report.IterationsPerformed() >= 1);
    assert(report.TotalFlipsApplied() > 0 || report.TotalVertexMovesApplied() > 0);
    assert(report.final_error.weighted_rmse <= before_error.weighted_rmse * 1.10 + 1e-9);
    assert(report.final_min_triangle_angle_deg >= before_min_angle - 1e-9);

    return 0;
}
