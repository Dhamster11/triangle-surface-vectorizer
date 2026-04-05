#include <iostream>
#include "svec/image/image.h"
#include "svec/refine/edge_aware_refinement.h"
#include "svec/render/reference_renderer.h"
#include "svec/surface/mesh_builders.h"
#include "svec/refine/adaptive_refinement.h"

using namespace svec;

int main() {
    ImageOKLaba img({64,64}, {0.1,0.0,0.0,1.0});
    for (int y=0;y<img.Height();++y){
        for(int x=0;x<img.Width();++x){
            auto &c=img.At(x,y);
            c.L = 0.1 + 0.8 * (double)x / (img.Width()-1);
            if (x > 32) { c.a = 0.2; }
        }
    }
    Mesh mesh = CreateImageQuadMesh(img.Size(), {0,0,0,1});
    for (auto &v: mesh.vertices) v.color = SampleImageOKLabaBilinear(img, v.position);

    EdgeAwareRefinementOptions opts{};
    opts.max_iterations = 400;
    opts.step.error_options.samples_per_axis = 1;
    opts.select.error_options = opts.step.error_options;
    opts.select.mode = EdgeAwareSelectionMode::CachedHeap;
    auto rep = EdgeAwareRefineMesh(mesh, img, opts);
    std::cout << rep.initial_error.weighted_rmse << " -> " << rep.final_error.weighted_rmse << " triangles=" << mesh.triangles.size() << " steps=" << rep.steps.size() << " stale=" << rep.stale_heap_pops << " blocked=" << rep.blocked_split_attempts << "\n";
    return 0;
}
