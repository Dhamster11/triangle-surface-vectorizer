#pragma once

#include <cmath>

#include "svec/core/types.h"
#include "svec/math/scalar.h"

namespace svec {

struct Vec4 {
    f64 x = 0.0;
    f64 y = 0.0;
    f64 z = 0.0;
    f64 w = 0.0;

    constexpr Vec4() noexcept = default;
    constexpr Vec4(f64 x_, f64 y_, f64 z_, f64 w_) noexcept : x(x_), y(y_), z(z_), w(w_) {}

    [[nodiscard]] constexpr Vec4 operator+() const noexcept { return *this; }
    [[nodiscard]] constexpr Vec4 operator-() const noexcept { return {-x, -y, -z, -w}; }

    [[nodiscard]] constexpr Vec4 operator+(const Vec4& rhs) const noexcept { return {x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w}; }
    [[nodiscard]] constexpr Vec4 operator-(const Vec4& rhs) const noexcept { return {x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w}; }
    [[nodiscard]] constexpr Vec4 operator*(f64 scalar) const noexcept { return {x * scalar, y * scalar, z * scalar, w * scalar}; }
    [[nodiscard]] constexpr Vec4 operator/(f64 scalar) const noexcept { return {x / scalar, y / scalar, z / scalar, w / scalar}; }

    constexpr Vec4& operator+=(const Vec4& rhs) noexcept {
        x += rhs.x; y += rhs.y; z += rhs.z; w += rhs.w; return *this;
    }

    constexpr Vec4& operator-=(const Vec4& rhs) noexcept {
        x -= rhs.x; y -= rhs.y; z -= rhs.z; w -= rhs.w; return *this;
    }

    constexpr Vec4& operator*=(f64 scalar) noexcept {
        x *= scalar; y *= scalar; z *= scalar; w *= scalar; return *this;
    }

    constexpr Vec4& operator/=(f64 scalar) noexcept {
        x /= scalar; y /= scalar; z /= scalar; w /= scalar; return *this;
    }

    [[nodiscard]] inline f64 LengthSquared() const noexcept { return x * x + y * y + z * z + w * w; }
    [[nodiscard]] inline f64 Length() const noexcept { return std::sqrt(LengthSquared()); }

    [[nodiscard]] inline Vec4 Normalized(f64 epsilon = kEpsilon) const noexcept {
        const f64 len = Length();
        return len <= epsilon ? Vec4{} : (*this / len);
    }
};

[[nodiscard]] constexpr Vec4 operator*(f64 scalar, const Vec4& v) noexcept {
    return v * scalar;
}

[[nodiscard]] constexpr bool operator==(const Vec4& lhs, const Vec4& rhs) noexcept {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
}

[[nodiscard]] constexpr bool operator!=(const Vec4& lhs, const Vec4& rhs) noexcept {
    return !(lhs == rhs);
}

[[nodiscard]] constexpr f64 Dot(const Vec4& a, const Vec4& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

[[nodiscard]] inline Vec4 Normalize(const Vec4& v, f64 epsilon = kEpsilon) noexcept {
    return v.Normalized(epsilon);
}

} // namespace svec
