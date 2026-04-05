#pragma once

#include <memory>
#include <vector>

#include "svec/core/types.h"
#include "svec/edge/edge_map.h"
#include "svec/image/image.h"
#include "svec/image/integral_map.h"
#include "svec/image/scanline_integral_stats.h"
#include "svec/math/vec2.h"

namespace svec {

struct StructureTensorOptions {
    bool enabled = true;

    // Multi-channel structure tensor weights. These are independent from the
    // scalar edge-map weights because the tensor is used for orientation and
    // anisotropy guidance rather than for edge magnitude visualization.
    f64 weight_L = 1.00;
    f64 weight_a = 0.85;
    f64 weight_b = 0.85;
    f64 weight_alpha = 0.35;

    // Number of separable [1 2 1] smoothing passes applied to the raw tensor
    // components before extracting the dominant axis and coherence.
    u32 smoothing_passes = 1;

    f64 coherence_epsilon = 1e-12;
    f64 strength_epsilon = 1e-12;
};

struct StructureTensorField {
    // Dominant gradient-axis orientation encoded as a doubled-angle unit field.
    // This avoids sign ambiguity (axis ~= -axis) and interpolates robustly.
    EdgeMap orientation_cos2_map;
    EdgeMap orientation_sin2_map;

    // Coherence in [0, 1]: 0 = locally isotropic / ambiguous, 1 = strongly
    // anisotropic, well-defined local structure direction.
    EdgeMap coherence_map;

    // Dominant tensor strength. This is stored in edge-like units
    // (sqrt(lambda_max)) so it stays numerically close to gradient magnitudes.
    EdgeMap strength_map;

    [[nodiscard]] bool IsValid() const noexcept {
        return orientation_cos2_map.IsValid()
            && orientation_sin2_map.IsValid()
            && coherence_map.IsValid()
            && strength_map.IsValid();
    }
};

struct StructureTensorSample {
    Vec2 normal{1.0, 0.0};
    Vec2 tangent{0.0, 1.0};
    f64 coherence = 0.0;
    f64 strength = 0.0;

    [[nodiscard]] bool HasPreferredDirection(
        f64 min_coherence = 1e-6,
        f64 min_strength = 1e-9) const noexcept {
        return coherence >= min_coherence && strength >= min_strength;
    }
};

struct ErrorPyramidLevel {
    ImageOKLaba image;
    EdgeMap edge_map;
    EdgeMap gradient_energy_map;
    IntegralMap gradient_energy_integral;
    std::shared_ptr<ScanlineIntegralStats> scanline_stats;
    StructureTensorField structure_tensor;
    f64 gradient_energy_mean = 0.0;
    f64 scale = 1.0;
};

struct ErrorPyramid {
    std::vector<ErrorPyramidLevel> levels;

    [[nodiscard]] bool IsValid() const noexcept { return !levels.empty(); }
    [[nodiscard]] const ErrorPyramidLevel& Base() const { return levels.front(); }
};

struct ErrorPyramidOptions {
    u32 max_levels = 6;
    i32 min_level_extent = 16;
    EdgeMapOptions edge_options{};
    StructureTensorOptions tensor_options{};
};

[[nodiscard]] ErrorPyramid BuildErrorPyramid(const ImageOKLaba& image, const ErrorPyramidOptions& options = {});
[[nodiscard]] ImageOKLaba DownsampleImage2x(const ImageOKLaba& src);
[[nodiscard]] EdgeMap ComputeGradientEnergyMap(const EdgeMap& edge_map);
[[nodiscard]] StructureTensorField BuildStructureTensorField(
    const ImageOKLaba& image,
    const StructureTensorOptions& options = {});
[[nodiscard]] StructureTensorSample SampleStructureTensorBilinear(
    const StructureTensorField& field,
    const Vec2& p);
[[nodiscard]] StructureTensorSample SampleStructureTensorBilinear(
    const ErrorPyramidLevel& level,
    const Vec2& image_space_point);

} // namespace svec
