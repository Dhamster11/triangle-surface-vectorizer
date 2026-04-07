#pragma once

#include <vector>

#include "svec/core/types.h"
#include "svec/image/image.h"
#include "svec/surface/mesh.h"
#include "svec/surface/triangle_plane.h"

namespace svec {

enum class SurfacePlaneRenderMode {
    PlaneShaded = 0,
    PlaneFitErrorTint = 1
};

struct SurfaceRenderTransform {
    f64 scale_x = 1.0;
    f64 scale_y = 1.0;
    f64 offset_x = 0.0;
    f64 offset_y = 0.0;
};

struct SurfacePlaneRenderOptions {
    ColorOKLaba clear_color{0.0, 0.0, 0.0, 0.0};
    SurfacePlaneRenderMode mode = SurfacePlaneRenderMode::PlaneShaded;
    bool overlay_wireframe = false;
    ColorOKLaba wire_color{0.75, 0.18, 0.12, 1.0};
    f64 wire_half_width_px = 0.5;
    bool skip_degenerate_triangles = true;
    SurfaceRenderTransform transform{};
    bool smooth_internal_edges = true;
    f64 internal_edge_blend_radius_px = 0.85;
    f64 internal_edge_blend_strength = 1.0;
    bool preserve_discontinuities = true;
    f64 discontinuity_threshold = 0.045;
};

struct SurfacePlaneRenderStats {
    u32 triangles_total = 0;
    u32 triangles_rasterized = 0;
    u32 triangles_skipped_degenerate = 0;
    u64 pixels_shaded = 0;
    u64 wire_pixels_shaded = 0;
    u64 internal_edge_samples_blended = 0;
};

struct SurfacePlaneRenderResult {
    ImageOKLaba image;
    SurfacePlaneRenderStats stats;
};

[[nodiscard]] SurfacePlaneRenderResult RenderMeshPlaneSurface(
    const Mesh& mesh,
    const std::vector<TrianglePlane>& planes,
    const ImageSize& output_size,
    const SurfacePlaneRenderOptions& options = {});

} // namespace svec
