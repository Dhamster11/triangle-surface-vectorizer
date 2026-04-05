#pragma once

#include "svec/core/types.h"
#include "svec/math/scalar.h"
#include "svec/math/vec3.h"
#include "svec/math/vec4.h"

namespace svec {

struct ColorRGBf {
    f64 r = 0.0;
    f64 g = 0.0;
    f64 b = 0.0;
};

struct ColorRGBAf {
    f64 r = 0.0;
    f64 g = 0.0;
    f64 b = 0.0;
    f64 a = 1.0;
};

struct ColorOKLab {
    f64 L = 0.0;
    f64 a = 0.0;
    f64 b = 0.0;
};

struct ColorOKLaba {
    f64 L = 0.0;
    f64 a = 0.0;
    f64 b = 0.0;
    f64 alpha = 1.0;
};

[[nodiscard]] constexpr Vec3 ToVec3(const ColorOKLab& c) noexcept {
    return {c.L, c.a, c.b};
}

[[nodiscard]] constexpr Vec4 ToVec4(const ColorOKLaba& c) noexcept {
    return {c.L, c.a, c.b, c.alpha};
}

[[nodiscard]] constexpr ColorOKLab FromVec3(const Vec3& v) noexcept {
    return {v.x, v.y, v.z};
}

[[nodiscard]] constexpr ColorOKLaba FromVec4(const Vec4& v) noexcept {
    return {v.x, v.y, v.z, v.w};
}

[[nodiscard]] inline ColorRGBAf Clamp01(const ColorRGBAf& c) noexcept {
    return {
        Saturate(c.r),
        Saturate(c.g),
        Saturate(c.b),
        Saturate(c.a)
    };
}

[[nodiscard]] inline ColorOKLaba Lerp(const ColorOKLaba& a, const ColorOKLaba& b, f64 t) noexcept {
    return {
        svec::Lerp(a.L, b.L, t),
        svec::Lerp(a.a, b.a, t),
        svec::Lerp(a.b, b.b, t),
        svec::Lerp(a.alpha, b.alpha, t)
    };
}

} // namespace svec
