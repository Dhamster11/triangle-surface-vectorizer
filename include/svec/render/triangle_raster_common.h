#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "svec/image/image.h"
#include "svec/math/color.h"
#include "svec/math/geometry.h"
#include "svec/math/scalar.h"
#include "svec/math/vec2.h"

namespace svec::detail {

inline constexpr i32 kMsaaGridSize = 4;
inline constexpr i32 kMsaaSampleCount = kMsaaGridSize * kMsaaGridSize;
inline constexpr std::uint16_t kFullCoverageMask = static_cast<std::uint16_t>((1u << kMsaaSampleCount) - 1u);
inline constexpr f64 kRasterEdgeEpsilon = 1e-12;

[[nodiscard]] constexpr std::array<Vec2, kMsaaSampleCount> BuildMsaaPattern() noexcept {
    return {{
        {0.125, 0.125}, {0.375, 0.125}, {0.625, 0.125}, {0.875, 0.125},
        {0.125, 0.375}, {0.375, 0.375}, {0.625, 0.375}, {0.875, 0.375},
        {0.125, 0.625}, {0.375, 0.625}, {0.625, 0.625}, {0.875, 0.625},
        {0.125, 0.875}, {0.375, 0.875}, {0.625, 0.875}, {0.875, 0.875}
    }};
}

inline constexpr std::array<Vec2, kMsaaSampleCount> kMsaaPattern = BuildMsaaPattern();

struct TrianglePixelBounds {
    i32 min_x = 0;
    i32 max_x = -1;
    i32 min_y = 0;
    i32 max_y = -1;

    [[nodiscard]] bool IsEmpty() const noexcept {
        return max_x < min_x || max_y < min_y;
    }
};

struct EdgeEquation {
    f64 a = 0.0;
    f64 b = 0.0;
    f64 c = 0.0;
    bool inclusive = false;

    [[nodiscard]] f64 Evaluate(const Vec2& p) const noexcept {
        return a * p.x + b * p.y + c;
    }

    [[nodiscard]] bool Passes(const Vec2& p) const noexcept {
        const f64 value = Evaluate(p);
        if (value > kRasterEdgeEpsilon) {
            return true;
        }
        if (value < -kRasterEdgeEpsilon) {
            return false;
        }
        return inclusive;
    }
};

struct TriangleRasterSetup {
    Vec2 p0{};
    Vec2 p1{};
    Vec2 p2{};
    std::array<EdgeEquation, 3> edges{};
    f64 signed_double_area = 0.0;
    bool valid = false;
};

struct AffineScalarField2D {
    f64 ax = 0.0;
    f64 by = 0.0;
    f64 c = 0.0;
    bool valid = false;

    [[nodiscard]] f64 Evaluate(const Vec2& p) const noexcept {
        return ax * p.x + by * p.y + c;
    }
};

struct AffineColorField2D {
    AffineScalarField2D l{};
    AffineScalarField2D a{};
    AffineScalarField2D b{};
    AffineScalarField2D alpha{};
    bool valid = false;

    [[nodiscard]] ColorOKLaba Evaluate(const Vec2& p) const noexcept {
        return {
            l.Evaluate(p),
            a.Evaluate(p),
            b.Evaluate(p),
            alpha.Evaluate(p)
        };
    }
};

[[nodiscard]] inline ColorOKLaba MakeZeroColor() noexcept {
    return {0.0, 0.0, 0.0, 0.0};
}

inline void ClearImage(ImageOKLaba& image, const ColorOKLaba& color) {
    for (i32 y = 0; y < image.Height(); ++y) {
        for (i32 x = 0; x < image.Width(); ++x) {
            image.At(x, y) = color;
        }
    }
}

inline void AddScaledColor(ColorOKLaba& dst, const ColorOKLaba& src, f64 scale) noexcept {
    dst.L += src.L * scale;
    dst.a += src.a * scale;
    dst.b += src.b * scale;
    dst.alpha += src.alpha * scale;
}

inline void ScaleColorInPlace(ColorOKLaba& value, f64 scale) noexcept {
    value.L *= scale;
    value.a *= scale;
    value.b *= scale;
    value.alpha *= scale;
}

[[nodiscard]] inline f64 EdgeFunction(const Vec2& a, const Vec2& b, const Vec2& p) noexcept {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

[[nodiscard]] inline bool IsInclusiveEdge(const Vec2& a, const Vec2& b) noexcept {
    const f64 dy = b.y - a.y;
    const f64 dx = b.x - a.x;
    return dy > 0.0 || (std::abs(dy) <= kRasterEdgeEpsilon && dx < 0.0);
}

[[nodiscard]] inline EdgeEquation BuildEdgeEquation(const Vec2& a, const Vec2& b) noexcept {
    return {
        a.y - b.y,
        b.x - a.x,
        a.x * b.y - a.y * b.x,
        IsInclusiveEdge(a, b)
    };
}

[[nodiscard]] inline TriangleRasterSetup BuildTriangleRasterSetup(
    Vec2 p0,
    Vec2 p1,
    Vec2 p2) noexcept {

    TriangleRasterSetup setup{};
    const f64 signed_double_area = EdgeFunction(p0, p1, p2);
    if (std::abs(signed_double_area) <= kRasterEdgeEpsilon) {
        return setup;
    }

    if (signed_double_area < 0.0) {
        std::swap(p1, p2);
    }

    setup.p0 = p0;
    setup.p1 = p1;
    setup.p2 = p2;
    setup.signed_double_area = std::abs(signed_double_area);
    setup.edges[0] = BuildEdgeEquation(setup.p0, setup.p1);
    setup.edges[1] = BuildEdgeEquation(setup.p1, setup.p2);
    setup.edges[2] = BuildEdgeEquation(setup.p2, setup.p0);
    setup.valid = true;
    return setup;
}

[[nodiscard]] inline AffineScalarField2D BuildAffineScalarFieldThroughTriangle(
    const Vec2& p0,
    f64 s0,
    const Vec2& p1,
    f64 s1,
    const Vec2& p2,
    f64 s2) noexcept {

    AffineScalarField2D field{};
    const f64 det = EdgeFunction(p0, p1, p2);
    if (std::abs(det) <= kRasterEdgeEpsilon) {
        field.c = s0;
        return field;
    }

    field.ax = (s0 * (p1.y - p2.y) + s1 * (p2.y - p0.y) + s2 * (p0.y - p1.y)) / det;
    field.by = (s0 * (p2.x - p1.x) + s1 * (p0.x - p2.x) + s2 * (p1.x - p0.x)) / det;
    field.c = (s0 * (p1.x * p2.y - p2.x * p1.y) +
               s1 * (p2.x * p0.y - p0.x * p2.y) +
               s2 * (p0.x * p1.y - p1.x * p0.y)) / det;
    field.valid = true;
    return field;
}

[[nodiscard]] inline AffineColorField2D BuildAffineColorFieldThroughTriangle(
    const Vec2& p0,
    const ColorOKLaba& c0,
    const Vec2& p1,
    const ColorOKLaba& c1,
    const Vec2& p2,
    const ColorOKLaba& c2) noexcept {

    AffineColorField2D field{};
    field.l = BuildAffineScalarFieldThroughTriangle(p0, c0.L, p1, c1.L, p2, c2.L);
    field.a = BuildAffineScalarFieldThroughTriangle(p0, c0.a, p1, c1.a, p2, c2.a);
    field.b = BuildAffineScalarFieldThroughTriangle(p0, c0.b, p1, c1.b, p2, c2.b);
    field.alpha = BuildAffineScalarFieldThroughTriangle(p0, c0.alpha, p1, c1.alpha, p2, c2.alpha);
    field.valid = field.l.valid && field.a.valid && field.b.valid && field.alpha.valid;
    return field;
}

[[nodiscard]] inline f64 DistancePointToSegment(const Vec2& p, const Vec2& a, const Vec2& b) noexcept {
    const Vec2 ab = b - a;
    const f64 denom = ab.LengthSquared();
    if (denom <= kEpsilon) {
        return Distance(p, a);
    }

    const f64 t = Saturate(Dot(p - a, ab) / denom);
    const Vec2 q = a + ab * t;
    return Distance(p, q);
}

[[nodiscard]] inline f64 DistancePointToTriangleEdges(
    const Vec2& p,
    const Vec2& p0,
    const Vec2& p1,
    const Vec2& p2) noexcept {

    return Min(
        DistancePointToSegment(p, p0, p1),
        Min(DistancePointToSegment(p, p1, p2), DistancePointToSegment(p, p2, p0)));
}

[[nodiscard]] inline f64 DistancePointToTriangleEdges(
    const Vec2& p,
    const TriangleRasterSetup& setup) noexcept {

    return DistancePointToTriangleEdges(p, setup.p0, setup.p1, setup.p2);
}

[[nodiscard]] inline bool PointInsideTriangle(const Vec2& p, const TriangleRasterSetup& setup) noexcept {
    if (!setup.valid) {
        return false;
    }

    return setup.edges[0].Passes(p) &&
           setup.edges[1].Passes(p) &&
           setup.edges[2].Passes(p);
}

[[nodiscard]] inline bool PointInsideTriangle(
    const Vec2& p,
    const Vec2& p0,
    const Vec2& p1,
    const Vec2& p2) noexcept {

    return PointInsideTriangle(p, BuildTriangleRasterSetup(p0, p1, p2));
}

[[nodiscard]] inline bool PixelSquareFullyInsideTriangle(
    i32 x,
    i32 y,
    const TriangleRasterSetup& setup) noexcept {

    const f64 fx = static_cast<f64>(x);
    const f64 fy = static_cast<f64>(y);

    return PointInsideTriangle({fx, fy}, setup) &&
           PointInsideTriangle({fx + 1.0, fy}, setup) &&
           PointInsideTriangle({fx, fy + 1.0}, setup) &&
           PointInsideTriangle({fx + 1.0, fy + 1.0}, setup);
}

[[nodiscard]] inline bool PixelSquareFullyInsideTriangle(
    i32 x,
    i32 y,
    const Vec2& p0,
    const Vec2& p1,
    const Vec2& p2) noexcept {

    return PixelSquareFullyInsideTriangle(x, y, BuildTriangleRasterSetup(p0, p1, p2));
}

[[nodiscard]] inline TrianglePixelBounds ComputeTrianglePixelBounds(
    const Vec2& p0,
    const Vec2& p1,
    const Vec2& p2,
    i32 target_width,
    i32 target_height,
    f64 grow = 0.0) noexcept {

    TrianglePixelBounds bounds{};

    const f64 min_x_f = Min(p0.x, Min(p1.x, p2.x)) - grow;
    const f64 max_x_f = Max(p0.x, Max(p1.x, p2.x)) + grow;
    const f64 min_y_f = Min(p0.y, Min(p1.y, p2.y)) - grow;
    const f64 max_y_f = Max(p0.y, Max(p1.y, p2.y)) + grow;

    bounds.min_x = Clamp(static_cast<i32>(std::floor(min_x_f)), 0, target_width - 1);
    bounds.max_x = Clamp(static_cast<i32>(std::ceil(max_x_f) - 1), 0, target_width - 1);
    bounds.min_y = Clamp(static_cast<i32>(std::floor(min_y_f)), 0, target_height - 1);
    bounds.max_y = Clamp(static_cast<i32>(std::ceil(max_y_f) - 1), 0, target_height - 1);
    return bounds;
}

[[nodiscard]] inline TrianglePixelBounds ComputeTrianglePixelBounds(
    const TriangleRasterSetup& setup,
    i32 target_width,
    i32 target_height,
    f64 grow = 0.0) noexcept {

    return ComputeTrianglePixelBounds(setup.p0, setup.p1, setup.p2, target_width, target_height, grow);
}

inline void ResolveHighQualityImage(
    ImageOKLaba& target,
    const std::vector<std::uint16_t>& sample_masks,
    const ColorOKLaba& clear_color) {

    const i32 width = target.Width();
    const i32 height = target.Height();

    for (i32 y = 0; y < height; ++y) {
        const std::size_t row_offset = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        for (i32 x = 0; x < width; ++x) {
            const std::size_t pixel_index = row_offset + static_cast<std::size_t>(x);
            const std::uint16_t mask = sample_masks[pixel_index];
            if (mask == 0u) {
                target.At(x, y) = clear_color;
                continue;
            }

            if (mask != kFullCoverageMask) {
                const u32 covered = std::popcount(static_cast<unsigned>(mask));
                if (covered > 0u) {
                    const f64 renorm = static_cast<f64>(kMsaaSampleCount) / static_cast<f64>(covered);
                    ScaleColorInPlace(target.At(x, y), renorm);
                }
            }
        }
    }
}

} // namespace svec::detail
