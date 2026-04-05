#include <cstdlib>
#include <iostream>

#include "svec/core/types.h"
#include "svec/math/color.h"
#include "svec/surface/mesh.h"
#include "svec/surface/mesh_builders.h"
#include "svec/surface/mesh_topology.h"

using namespace svec;

namespace {

bool Check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[step3] FAIL: " << message << '\n';
        return false;
    }
    return true;
}

} // namespace

int main() {
    const ColorOKLaba fill{0.7, 0.01, -0.02, 1.0};
    Mesh mesh = CreateImageQuadMesh({16, 10}, fill);

    if (!Check(mesh.vertices.size() == 4, "quad mesh must have 4 vertices")) return EXIT_FAILURE;
    if (!Check(mesh.triangles.size() == 2, "quad mesh must have 2 triangles")) return EXIT_FAILURE;

    std::string error;
    if (!Check(ValidateMeshGeometry(mesh, &error), error.c_str())) return EXIT_FAILURE;

    const auto topo = BuildTriangleTopology(mesh);
    if (!Check(topo.ok, topo.error.c_str())) return EXIT_FAILURE;
    if (!Check(mesh.HasTopology(), "mesh topology must exist after build")) return EXIT_FAILURE;

    const Triangle& t0 = mesh.triangles[0];
    const Triangle& t1 = mesh.triangles[1];

    const f64 area0 = ComputeTriangleArea(mesh, t0);
    const f64 area1 = ComputeTriangleArea(mesh, t1);

    if (!Check(area0 > 0.0, "triangle 0 area must be positive")) return EXIT_FAILURE;
    if (!Check(area1 > 0.0, "triangle 1 area must be positive")) return EXIT_FAILURE;

    // Triangles share edge (0,2):
    const i32 t0_shared = FindLocalOppositeVertexIndex(t0, 0, 2);
    const i32 t1_shared = FindLocalOppositeVertexIndex(t1, 0, 2);

    if (!Check(t0_shared >= 0, "shared edge must exist in triangle 0")) return EXIT_FAILURE;
    if (!Check(t1_shared >= 0, "shared edge must exist in triangle 1")) return EXIT_FAILURE;

    if (!Check(mesh.topology[0].neighbors[static_cast<std::size_t>(t0_shared)] == 1,
               "triangle 0 must reference triangle 1 across shared edge")) {
        return EXIT_FAILURE;
    }

    if (!Check(mesh.topology[1].neighbors[static_cast<std::size_t>(t1_shared)] == 0,
               "triangle 1 must reference triangle 0 across shared edge")) {
        return EXIT_FAILURE;
    }

    std::cout << "[step3] smoke test passed\n";
    return EXIT_SUCCESS;
}
