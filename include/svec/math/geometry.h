#pragma once

#include <array>
#include <optional>

#include "svec/core/types.h"
#include "svec/math/scalar.h"
#include "svec/math/vec2.h"
#include "svec/math/vec3.h"
#include "svec/math/vec4.h"

namespace svec {

struct Barycentric {
    f64 u = 0.0;
    f64 v = 0.0;
    f64 w = 0.0;

    [[nodiscard]] inline bool IsFinite() const noexcept {
        return std::isfinite(u) && std::isfinite(v) && std::isfinite(w);
    }

    [[nodiscard]] inline bool IsInside(f64 epsilon = kEpsilon) const noexcept {
        return u >= -epsilon && v >= -epsilon && w >= -epsilon;
    }

    [[nodiscard]] inline f64 Sum() const noexcept {
        return u + v + w;
    }
};

[[nodiscard]] constexpr f64 TriangleAreaSigned2(const Vec2& a, const Vec2& b, const Vec2& c) noexcept {
    return Cross(b - a, c - a);
}

[[nodiscard]] inline f64 TriangleAreaSigned(const Vec2& a, const Vec2& b, const Vec2& c) noexcept {
    return 0.5 * TriangleAreaSigned2(a, b, c);
}

[[nodiscard]] inline f64 TriangleArea(const Vec2& a, const Vec2& b, const Vec2& c) noexcept {
    return std::abs(TriangleAreaSigned(a, b, c));
}

[[nodiscard]] inline bool IsDegenerateTriangle(const Vec2& a, const Vec2& b, const Vec2& c, f64 epsilon = kEpsilon) noexcept {
    return std::abs(TriangleAreaSigned2(a, b, c)) <= epsilon;
}

[[nodiscard]] inline std::optional<Barycentric> ComputeBarycentric(
    const Vec2& p,
    const Vec2& a,
    const Vec2& b,
    const Vec2& c,
    f64 epsilon = kEpsilon) noexcept {

    const f64 denom = TriangleAreaSigned2(a, b, c);
    if (std::abs(denom) <= epsilon) {
        return std::nullopt;
    }

    const f64 invDenom = 1.0 / denom;
    const f64 u = TriangleAreaSigned2(p, b, c) * invDenom;
    const f64 v = TriangleAreaSigned2(a, p, c) * invDenom;
    const f64 w = TriangleAreaSigned2(a, b, p) * invDenom;

    return Barycentric{u, v, w};
}

template <typename T>
[[nodiscard]] inline T InterpolateBarycentric(const T& av, const T& bv, const T& cv, const Barycentric& bc) noexcept {
    return av * bc.u + bv * bc.v + cv * bc.w;
}

[[nodiscard]] inline Vec2 TriangleCentroid(const Vec2& a, const Vec2& b, const Vec2& c) noexcept {
    return (a + b + c) / 3.0;
}

[[nodiscard]] inline std::array<f64, 3> EdgeLengths(const Vec2& a, const Vec2& b, const Vec2& c) noexcept {
    return {
        Distance(b, c),
        Distance(a, c),
        Distance(a, b)
    };
}

[[nodiscard]] inline i32 LongestEdgeIndex(const Vec2& a, const Vec2& b, const Vec2& c) noexcept {
    const auto lengths = EdgeLengths(a, b, c);
    if (lengths[0] >= lengths[1] && lengths[0] >= lengths[2]) {
        return 0; // edge BC
    }
    if (lengths[1] >= lengths[2]) {
        return 1; // edge AC
    }
    return 2;     // edge AB
}

[[nodiscard]] inline Vec2 Midpoint(const Vec2& a, const Vec2& b) noexcept {
    return (a + b) * 0.5;
}

[[nodiscard]] inline bool PointInTriangle(const Vec2& p, const Vec2& a, const Vec2& b, const Vec2& c, f64 epsilon = kEpsilon) noexcept {
    const auto bc = ComputeBarycentric(p, a, b, c, epsilon);
    return bc.has_value() && bc->IsInside(epsilon);
}

} // namespace svec
