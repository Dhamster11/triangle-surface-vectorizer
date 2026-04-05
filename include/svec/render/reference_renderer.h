#pragma once

#include "svec/core/types.h"
#include "svec/image/image.h"
#include "svec/math/color.h"
#include "svec/surface/mesh.h"

namespace svec {

enum class ReferenceRenderMode {
    InterpolatedColor = 0,
    TriangleIdFlat = 1
};

struct ReferenceRenderOptions {
    ColorOKLaba clear_color{0.0, 0.0, 0.0, 0.0};
    ReferenceRenderMode mode = ReferenceRenderMode::InterpolatedColor;

    bool overlay_wireframe = false;
    ColorOKLaba wire_color{0.75, 0.18, 0.12, 1.0};
    f64 wire_half_width_px = 0.5;

    bool skip_degenerate_triangles = true;
};

struct ReferenceRenderStats {
    u32 triangles_total = 0;
    u32 triangles_rasterized = 0;
    u32 triangles_skipped_degenerate = 0;
    u64 pixels_shaded = 0;
    u64 wire_pixels_shaded = 0;
};

struct ReferenceRenderResult {
    ImageOKLaba image;
    ReferenceRenderStats stats;
};

[[nodiscard]] ReferenceRenderResult RenderMeshReference(
    const Mesh& mesh,
    const ImageSize& output_size,
    const ReferenceRenderOptions& options = {});

void RenderMeshReferenceTo(
    const Mesh& mesh,
    ImageOKLaba& target,
    const ReferenceRenderOptions& options = {},
    ReferenceRenderStats* out_stats = nullptr);

} // namespace svec
