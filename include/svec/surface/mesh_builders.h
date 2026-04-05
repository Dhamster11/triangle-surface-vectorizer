#pragma once

#include "svec/surface/mesh.h"

namespace svec {

// Image-space convention for now:
// pixel centers live on integer coordinates [0 .. width-1] x [0 .. height-1].
// The initial full-image mesh spans exactly that rectangle.
[[nodiscard]] Mesh CreateImageQuadMesh(const ImageSize& size, const ColorOKLaba& fill = {});
[[nodiscard]] Mesh CreateImageGridMesh(const ImageSize& size, u32 cells_x, u32 cells_y, const ColorOKLaba& fill = {});

} // namespace svec
