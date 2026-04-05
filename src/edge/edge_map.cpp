#include "svec/edge/edge_map.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "svec/math/scalar.h"
#include "svec/math/vec2.h"

namespace svec {
namespace {

[[nodiscard]] const ColorOKLaba& ClampedPixel(const ImageOKLaba& image, i32 x, i32 y) {
    const i32 cx = Clamp(x, 0, image.Width() - 1);
    const i32 cy = Clamp(y, 0, image.Height() - 1);
    return image.At(cx, cy);
}

[[nodiscard]] f64 LuminanceHeat(f64 t) noexcept {
    return Saturate(t);
}

[[nodiscard]] ColorOKLaba HeatColor(f64 t) noexcept {
    const f64 x = Saturate(t);
    const f64 L = 0.15 + 0.80 * x;
    const f64 a = -0.10 + 0.36 * x;
    const f64 b = 0.12 + 0.12 * (1.0 - std::abs(2.0 * x - 1.0));
    return {L, a, b, 1.0};
}

} // namespace

void EdgeMap::Resize(ImageSize size, f64 fill) {
    if (!size.IsValid()) {
        m_size = {};
        m_values.clear();
        return;
    }
    m_size = size;
    m_values.assign(static_cast<std::size_t>(size.PixelCount()), fill);
}

bool EdgeMap::IsValid() const noexcept {
    return m_size.IsValid() && static_cast<i64>(m_values.size()) == m_size.PixelCount();
}

std::size_t EdgeMap::IndexOf(i32 x, i32 y) const {
    if (x < 0 || y < 0 || x >= m_size.width || y >= m_size.height) {
        throw std::out_of_range("EdgeMap::At index out of range");
    }
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(m_size.width) + static_cast<std::size_t>(x);
}

const f64& EdgeMap::At(i32 x, i32 y) const {
    return m_values.at(IndexOf(x, y));
}

f64& EdgeMap::At(i32 x, i32 y) {
    return m_values.at(IndexOf(x, y));
}

f64 EdgeMap::MaxValue() const noexcept {
    f64 m = 0.0;
    for (f64 v : m_values) {
        m = Max(m, v);
    }
    return m;
}

EdgeMap ComputeEdgeMapSobel(const ImageOKLaba& image, const EdgeMapOptions& options) {
    if (!image.IsValid()) {
        throw std::runtime_error("ComputeEdgeMapSobel: image is invalid.");
    }

    EdgeMap out(image.Size(), 0.0);

    constexpr i32 kx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    constexpr i32 ky[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    f64 max_value = 0.0;

    for (i32 y = 0; y < image.Height(); ++y) {
        for (i32 x = 0; x < image.Width(); ++x) {
            f64 gxL = 0.0, gyL = 0.0;
            f64 gxa = 0.0, gya = 0.0;
            f64 gxb = 0.0, gyb = 0.0;
            f64 gxaA = 0.0, gyaA = 0.0;

            for (i32 j = -1; j <= 1; ++j) {
                for (i32 i = -1; i <= 1; ++i) {
                    const ColorOKLaba& c = ClampedPixel(image, x + i, y + j);
                    const f64 sx = static_cast<f64>(kx[j + 1][i + 1]);
                    const f64 sy = static_cast<f64>(ky[j + 1][i + 1]);

                    gxL += sx * c.L;  gyL += sy * c.L;
                    gxa += sx * c.a;  gya += sy * c.a;
                    gxb += sx * c.b;  gyb += sy * c.b;
                    gxaA += sx * c.alpha; gyaA += sy * c.alpha;
                }
            }

            const f64 mag_L = std::sqrt(gxL * gxL + gyL * gyL);
            const f64 mag_a = std::sqrt(gxa * gxa + gya * gya);
            const f64 mag_b = std::sqrt(gxb * gxb + gyb * gyb);
            const f64 mag_alpha = std::sqrt(gxaA * gxaA + gyaA * gyaA);

            const f64 edge = std::sqrt(
                options.weight_L * mag_L * mag_L +
                options.weight_a * mag_a * mag_a +
                options.weight_b * mag_b * mag_b +
                options.weight_alpha * mag_alpha * mag_alpha);

            out.At(x, y) = edge;
            max_value = Max(max_value, edge);
        }
    }

    if (options.normalize_to_unit && max_value > options.normalization_epsilon) {
        const f64 inv = 1.0 / max_value;
        for (f64& v : out.Pixels()) {
            v *= inv;
        }
    }

    return out;
}

f64 SampleEdgeMapNearest(const EdgeMap& edge_map, const Vec2& p) {
    if (!edge_map.IsValid()) {
        throw std::runtime_error("SampleEdgeMapNearest: edge_map is invalid.");
    }

    const i32 x = Clamp(static_cast<i32>(std::llround(p.x)), 0, edge_map.Width() - 1);
    const i32 y = Clamp(static_cast<i32>(std::llround(p.y)), 0, edge_map.Height() - 1);
    return edge_map.At(x, y);
}

f64 SampleEdgeMapBilinear(const EdgeMap& edge_map, const Vec2& p) {
    if (!edge_map.IsValid()) {
        throw std::runtime_error("SampleEdgeMapBilinear: edge_map is invalid.");
    }

    const f64 x = Clamp(p.x, 0.0, static_cast<f64>(edge_map.Width() - 1));
    const f64 y = Clamp(p.y, 0.0, static_cast<f64>(edge_map.Height() - 1));

    const i32 x0 = Clamp(static_cast<i32>(std::floor(x)), 0, edge_map.Width() - 1);
    const i32 y0 = Clamp(static_cast<i32>(std::floor(y)), 0, edge_map.Height() - 1);
    const i32 x1 = Clamp(x0 + 1, 0, edge_map.Width() - 1);
    const i32 y1 = Clamp(y0 + 1, 0, edge_map.Height() - 1);

    const f64 tx = x - static_cast<f64>(x0);
    const f64 ty = y - static_cast<f64>(y0);

    const f64 v00 = edge_map.At(x0, y0);
    const f64 v10 = edge_map.At(x1, y0);
    const f64 v01 = edge_map.At(x0, y1);
    const f64 v11 = edge_map.At(x1, y1);

    const f64 vx0 = Lerp(v00, v10, tx);
    const f64 vx1 = Lerp(v01, v11, tx);
    return Lerp(vx0, vx1, ty);
}

ImageOKLaba RenderEdgeMapPreview(const EdgeMap& edge_map, bool use_heat) {
    if (!edge_map.IsValid()) {
        throw std::runtime_error("RenderEdgeMapPreview: edge_map is invalid.");
    }

    ImageOKLaba out(edge_map.Size(), {0.0, 0.0, 0.0, 1.0});
    const f64 norm = Max(edge_map.MaxValue(), 1e-12);

    for (i32 y = 0; y < out.Height(); ++y) {
        for (i32 x = 0; x < out.Width(); ++x) {
            const f64 t = edge_map.At(x, y) / norm;
            if (use_heat) {
                out.At(x, y) = HeatColor(t);
            } else {
                const f64 L = LuminanceHeat(t);
                out.At(x, y) = {L, 0.0, 0.0, 1.0};
            }
        }
    }

    return out;
}

} // namespace svec
