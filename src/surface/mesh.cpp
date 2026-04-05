#include "svec/surface/mesh.h"

namespace svec {

bool ValidateMeshIndices(const Mesh& mesh, std::string* error) {
    for (std::size_t ti = 0; ti < mesh.triangles.size(); ++ti) {
        const Triangle& tri = mesh.triangles[ti];
        if (!tri.IsIndexed()) {
            if (error) {
                *error = "Triangle " + std::to_string(ti) + " contains invalid vertex index.";
            }
            return false;
        }

        if (tri.HasDuplicateVertices()) {
            if (error) {
                *error = "Triangle " + std::to_string(ti) + " contains duplicate vertex ids.";
            }
            return false;
        }

        for (VertexId vid : tri.v) {
            if (!mesh.IsValidVertexId(vid)) {
                if (error) {
                    *error = "Triangle " + std::to_string(ti) + " references out-of-range vertex id.";
                }
                return false;
            }
        }
    }

    return true;
}

bool ValidateMeshGeometry(const Mesh& mesh, std::string* error, f64 epsilon) {
    if (!ValidateMeshIndices(mesh, error)) {
        return false;
    }

    for (std::size_t ti = 0; ti < mesh.triangles.size(); ++ti) {
        const Triangle& tri = mesh.triangles[ti];
        if (IsDegenerate(mesh, tri, epsilon)) {
            if (error) {
                *error = "Triangle " + std::to_string(ti) + " is degenerate or has near-zero area.";
            }
            return false;
        }
    }

    return true;
}

} // namespace svec
