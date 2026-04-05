#pragma once

#include <cmath>
#include <vector>

#include "svec/core/types.h"
#include "svec/image/image.h"
#include "svec/image/scanline_integral_stats.h"
#include "svec/math/color.h"
#include "svec/surface/mesh.h"

namespace svec {

struct Plane1 {
    f64 c = 0.0;
    f64 cx = 0.0;
    f64 cy = 0.0;

    [[nodiscard]] f64 Evaluate(f64 x, f64 y) const noexcept {
        return c + cx * x + cy * y;
    }

    [[nodiscard]] f64 GradientMagnitude() const noexcept {
        return std::sqrt(cx * cx + cy * cy);
    }
};

struct TrianglePlane {
    TriangleId triangle_id = kInvalidIndex;
    Plane1 L{};
    Plane1 a{};
    Plane1 b{};
    Plane1 alpha{};
    f64 fit_rmse = 0.0;
    u32 fit_sample_count = 0;
};

struct TrianglePlaneFitOptions {
    bool include_vertices = true;
    bool include_edge_midpoints = true;
    bool include_centroid = true;
    u32 interior_barycentric_samples = 1;
    bool clamp_alpha = true;
    bool prefer_scanline_integral_stats = true;
};

[[nodiscard]] TrianglePlane FitTrianglePlane(
    const Mesh& mesh,
    TriangleId triangle_id,
    const ImageOKLaba& reference,
    const TrianglePlaneFitOptions& options = {});

[[nodiscard]] TrianglePlane FitTrianglePlane(
    const Mesh& mesh,
    TriangleId triangle_id,
    const ImageOKLaba& reference,
    const ScanlineIntegralStats* stats,
    const TrianglePlaneFitOptions& options);

[[nodiscard]] std::vector<TrianglePlane> FitAllTrianglePlanes(
    const Mesh& mesh,
    const ImageOKLaba& reference,
    const TrianglePlaneFitOptions& options = {});

[[nodiscard]] std::vector<TrianglePlane> FitAllTrianglePlanes(
    const Mesh& mesh,
    const ImageOKLaba& reference,
    const ScanlineIntegralStats* stats,
    const TrianglePlaneFitOptions& options);

[[nodiscard]] ColorOKLaba EvaluateTrianglePlane(const TrianglePlane& plane, f64 x, f64 y, bool clamp_alpha = true) noexcept;

} // namespace svec
