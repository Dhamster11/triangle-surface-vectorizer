#include "svec/pyramid/error_pyramid.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "svec/math/scalar.h"

namespace svec {
namespace {

constexpr i32 kSobelX[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

constexpr i32 kSobelY[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

[[nodiscard]] const ColorOKLaba& ClampedPixel(const ImageOKLaba& image, i32 x, i32 y) noexcept {
    const i32 cx = Clamp(x, 0, image.Width() - 1);
    const i32 cy = Clamp(y, 0, image.Height() - 1);
    return image.At(cx, cy);
}

[[nodiscard]] std::size_t ScalarIndex(const ImageSize& size, i32 x, i32 y) noexcept {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(size.width)
         + static_cast<std::size_t>(x);
}

void SmoothScalarFieldSeparable(ImageSize size, std::vector<f64>& values, u32 passes) {
    if (!size.IsValid() || values.empty() || passes == 0u) {
        return;
    }

    std::vector<f64> temp(values.size(), 0.0);

    for (u32 pass = 0; pass < passes; ++pass) {
        for (i32 y = 0; y < size.height; ++y) {
            for (i32 x = 0; x < size.width; ++x) {
                const i32 x0 = Max(0, x - 1);
                const i32 x1 = x;
                const i32 x2 = Min(size.width - 1, x + 1);
                temp[ScalarIndex(size, x, y)] =
                    0.25 * values[ScalarIndex(size, x0, y)] +
                    0.50 * values[ScalarIndex(size, x1, y)] +
                    0.25 * values[ScalarIndex(size, x2, y)];
            }
        }

        for (i32 y = 0; y < size.height; ++y) {
            const i32 y0 = Max(0, y - 1);
            const i32 y1 = y;
            const i32 y2 = Min(size.height - 1, y + 1);
            for (i32 x = 0; x < size.width; ++x) {
                values[ScalarIndex(size, x, y)] =
                    0.25 * temp[ScalarIndex(size, x, y0)] +
                    0.50 * temp[ScalarIndex(size, x, y1)] +
                    0.25 * temp[ScalarIndex(size, x, y2)];
            }
        }
    }
}

struct RawStructureTensor {
    f64 jxx = 0.0;
    f64 jxy = 0.0;
    f64 jyy = 0.0;
};

[[nodiscard]] RawStructureTensor ComputeRawStructureTensorAt(
    const ImageOKLaba& image,
    i32 x,
    i32 y,
    const StructureTensorOptions& options) noexcept {

    f64 gxL = 0.0, gyL = 0.0;
    f64 gxa = 0.0, gya = 0.0;
    f64 gxb = 0.0, gyb = 0.0;
    f64 gxAlpha = 0.0, gyAlpha = 0.0;

    for (i32 j = -1; j <= 1; ++j) {
        for (i32 i = -1; i <= 1; ++i) {
            const ColorOKLaba& c = ClampedPixel(image, x + i, y + j);
            const f64 sx = static_cast<f64>(kSobelX[j + 1][i + 1]);
            const f64 sy = static_cast<f64>(kSobelY[j + 1][i + 1]);

            gxL += sx * c.L;         gyL += sy * c.L;
            gxa += sx * c.a;         gya += sy * c.a;
            gxb += sx * c.b;         gyb += sy * c.b;
            gxAlpha += sx * c.alpha; gyAlpha += sy * c.alpha;
        }
    }

    RawStructureTensor out{};
    out.jxx =
        options.weight_L * gxL * gxL +
        options.weight_a * gxa * gxa +
        options.weight_b * gxb * gxb +
        options.weight_alpha * gxAlpha * gxAlpha;
    out.jxy =
        options.weight_L * gxL * gyL +
        options.weight_a * gxa * gya +
        options.weight_b * gxb * gyb +
        options.weight_alpha * gxAlpha * gyAlpha;
    out.jyy =
        options.weight_L * gyL * gyL +
        options.weight_a * gya * gya +
        options.weight_b * gyb * gyb +
        options.weight_alpha * gyAlpha * gyAlpha;
    return out;
}

[[nodiscard]] f64 SampleScalarMapBilinearClamped(const EdgeMap& map, const Vec2& p) noexcept {
    const f64 x = Clamp(p.x, 0.0, static_cast<f64>(map.Width() - 1));
    const f64 y = Clamp(p.y, 0.0, static_cast<f64>(map.Height() - 1));

    const i32 x0 = Clamp(static_cast<i32>(std::floor(x)), 0, map.Width() - 1);
    const i32 y0 = Clamp(static_cast<i32>(std::floor(y)), 0, map.Height() - 1);
    const i32 x1 = Clamp(x0 + 1, 0, map.Width() - 1);
    const i32 y1 = Clamp(y0 + 1, 0, map.Height() - 1);

    const f64 tx = x - static_cast<f64>(x0);
    const f64 ty = y - static_cast<f64>(y0);

    const f64 v00 = map.At(x0, y0);
    const f64 v10 = map.At(x1, y0);
    const f64 v01 = map.At(x0, y1);
    const f64 v11 = map.At(x1, y1);

    const f64 vx0 = Lerp(v00, v10, tx);
    const f64 vx1 = Lerp(v01, v11, tx);
    return Lerp(vx0, vx1, ty);
}

} // namespace

ImageOKLaba DownsampleImage2x(const ImageOKLaba& src) {
    if (!src.IsValid()) {
        throw std::runtime_error("DownsampleImage2x: source image is invalid.");
    }

    const i32 dst_w = Max(1, (src.Width() + 1) / 2);
    const i32 dst_h = Max(1, (src.Height() + 1) / 2);
    ImageOKLaba dst({dst_w, dst_h}, {0.0, 0.0, 0.0, 1.0});

    for (i32 y = 0; y < dst_h; ++y) {
        for (i32 x = 0; x < dst_w; ++x) {
            f64 sumL = 0.0;
            f64 suma = 0.0;
            f64 sumb = 0.0;
            f64 sumAlpha = 0.0;
            i32 count = 0;
            for (i32 j = 0; j < 2; ++j) {
                for (i32 i = 0; i < 2; ++i) {
                    const i32 sx = Clamp(2 * x + i, 0, src.Width() - 1);
                    const i32 sy = Clamp(2 * y + j, 0, src.Height() - 1);
                    const ColorOKLaba c = src.At(sx, sy);
                    sumL += c.L;
                    suma += c.a;
                    sumb += c.b;
                    sumAlpha += c.alpha;
                    ++count;
                }
            }
            const f64 inv = 1.0 / static_cast<f64>(count);
            dst.At(x, y) = {sumL * inv, suma * inv, sumb * inv, sumAlpha * inv};
        }
    }
    return dst;
}

EdgeMap ComputeGradientEnergyMap(const EdgeMap& edge_map) {
    if (!edge_map.IsValid()) {
        throw std::runtime_error("ComputeGradientEnergyMap: edge map is invalid.");
    }
    EdgeMap out(edge_map.Size(), 0.0);
    for (i32 y = 0; y < edge_map.Height(); ++y) {
        for (i32 x = 0; x < edge_map.Width(); ++x) {
            const f64 g = edge_map.At(x, y);
            out.At(x, y) = g * g;
        }
    }
    return out;
}

StructureTensorField BuildStructureTensorField(
    const ImageOKLaba& image,
    const StructureTensorOptions& options) {

    if (!image.IsValid()) {
        throw std::runtime_error("BuildStructureTensorField: image is invalid.");
    }

    StructureTensorField out{};
    if (!options.enabled) {
        return out;
    }

    const ImageSize size = image.Size();
    const std::size_t pixel_count = static_cast<std::size_t>(size.PixelCount());

    std::vector<f64> jxx(pixel_count, 0.0);
    std::vector<f64> jxy(pixel_count, 0.0);
    std::vector<f64> jyy(pixel_count, 0.0);

    for (i32 y = 0; y < size.height; ++y) {
        for (i32 x = 0; x < size.width; ++x) {
            const RawStructureTensor tensor = ComputeRawStructureTensorAt(image, x, y, options);
            const std::size_t idx = ScalarIndex(size, x, y);
            jxx[idx] = tensor.jxx;
            jxy[idx] = tensor.jxy;
            jyy[idx] = tensor.jyy;
        }
    }

    SmoothScalarFieldSeparable(size, jxx, options.smoothing_passes);
    SmoothScalarFieldSeparable(size, jxy, options.smoothing_passes);
    SmoothScalarFieldSeparable(size, jyy, options.smoothing_passes);

    out.orientation_cos2_map.Resize(size, 0.0);
    out.orientation_sin2_map.Resize(size, 0.0);
    out.coherence_map.Resize(size, 0.0);
    out.strength_map.Resize(size, 0.0);

    for (i32 y = 0; y < size.height; ++y) {
        for (i32 x = 0; x < size.width; ++x) {
            const std::size_t idx = ScalarIndex(size, x, y);
            const f64 a = jxx[idx];
            const f64 b = jxy[idx];
            const f64 c = jyy[idx];

            const f64 trace = Max(a + c, 0.0);
            const f64 disc2 = Max((a - c) * (a - c) + 4.0 * b * b, 0.0);
            const f64 disc = std::sqrt(disc2);
            const f64 lambda1 = 0.5 * (trace + disc);

            f64 cos2 = 1.0;
            f64 sin2 = 0.0;
            if (disc > options.coherence_epsilon) {
                cos2 = (a - c) / disc;
                sin2 = (2.0 * b) / disc;
            }

            const f64 coherence = trace > options.coherence_epsilon
                ? Saturate(disc / trace)
                : 0.0;
            const f64 strength = lambda1 > options.strength_epsilon
                ? std::sqrt(lambda1)
                : 0.0;

            out.orientation_cos2_map.At(x, y) = cos2;
            out.orientation_sin2_map.At(x, y) = sin2;
            out.coherence_map.At(x, y) = coherence;
            out.strength_map.At(x, y) = strength;
        }
    }

    return out;
}

StructureTensorSample SampleStructureTensorBilinear(
    const StructureTensorField& field,
    const Vec2& p) {

    if (!field.IsValid()) {
        throw std::runtime_error("SampleStructureTensorBilinear: structure tensor field is invalid.");
    }

    const f64 cos2 = SampleScalarMapBilinearClamped(field.orientation_cos2_map, p);
    const f64 sin2 = SampleScalarMapBilinearClamped(field.orientation_sin2_map, p);
    const f64 coherence = SampleScalarMapBilinearClamped(field.coherence_map, p);
    const f64 strength = SampleScalarMapBilinearClamped(field.strength_map, p);

    const f64 axis_len = std::sqrt(cos2 * cos2 + sin2 * sin2);
    const f64 axis_cos2 = axis_len > 1e-12 ? (cos2 / axis_len) : 1.0;
    const f64 axis_sin2 = axis_len > 1e-12 ? (sin2 / axis_len) : 0.0;

    const f64 theta = 0.5 * std::atan2(axis_sin2, axis_cos2);
    const Vec2 normal{std::cos(theta), std::sin(theta)};
    const Vec2 tangent{-normal.y, normal.x};

    return {normal, tangent, Saturate(coherence), Max(strength, 0.0)};
}

StructureTensorSample SampleStructureTensorBilinear(
    const ErrorPyramidLevel& level,
    const Vec2& image_space_point) {

    if (!level.structure_tensor.IsValid()) {
        throw std::runtime_error("SampleStructureTensorBilinear: pyramid level has no valid structure tensor field.");
    }
    const f64 inv_scale = level.scale > 0.0 ? (1.0 / level.scale) : 1.0;
    return SampleStructureTensorBilinear(level.structure_tensor, {image_space_point.x * inv_scale, image_space_point.y * inv_scale});
}

ErrorPyramid BuildErrorPyramid(const ImageOKLaba& image, const ErrorPyramidOptions& options) {
    if (!image.IsValid()) {
        throw std::runtime_error("BuildErrorPyramid: image is invalid.");
    }

    ErrorPyramid out{};
    out.levels.reserve(options.max_levels);

    ImageOKLaba current = image;
    f64 scale = 1.0;
    for (u32 level = 0; level < options.max_levels; ++level) {
        ErrorPyramidLevel pl{};
        pl.image = current;
        pl.scale = scale;
        pl.edge_map = ComputeEdgeMapSobel(current, options.edge_options);
        pl.gradient_energy_map = ComputeGradientEnergyMap(pl.edge_map);
        pl.gradient_energy_integral.BuildFromEdgeMap(pl.gradient_energy_map);
        pl.structure_tensor = BuildStructureTensorField(current, options.tensor_options);
        if (!pl.gradient_energy_map.Pixels().empty()) {
            f64 sum = 0.0;
            for (const f64 v : pl.gradient_energy_map.Pixels()) sum += v;
            pl.gradient_energy_mean = sum / static_cast<f64>(pl.gradient_energy_map.Pixels().size());
        }
        out.levels.push_back(std::move(pl));

        if (current.Width() <= options.min_level_extent || current.Height() <= options.min_level_extent) {
            break;
        }
        current = DownsampleImage2x(current);
        scale *= 2.0;
    }

    return out;
}

} // namespace svec
