#pragma once

#include <cmath>

#include "svec/core/types.h"
#include "svec/math/scalar.h"

namespace svec {

struct Vec2 {
    f64 x = 0.0;
    f64 y = 0.0;

    constexpr Vec2() noexcept = default;
    constexpr Vec2(f64 x_, f64 y_) noexcept : x(x_), y(y_) {}

    [[nodiscard]] constexpr Vec2 operator+() const noexcept { return *this; }
    [[nodiscard]] constexpr Vec2 operator-() const noexcept { return {-x, -y}; }

    [[nodiscard]] constexpr Vec2 operator+(const Vec2& rhs) const noexcept { return {x + rhs.x, y + rhs.y}; }
    [[nodiscard]] constexpr Vec2 operator-(const Vec2& rhs) const noexcept { return {x - rhs.x, y - rhs.y}; }
    [[nodiscard]] constexpr Vec2 operator*(f64 scalar) const noexcept { return {x * scalar, y * scalar}; }
    [[nodiscard]] constexpr Vec2 operator/(f64 scalar) const noexcept { return {x / scalar, y / scalar}; }

    constexpr Vec2& operator+=(const Vec2& rhs) noexcept {
        x += rhs.x; y += rhs.y; return *this;
    }

    constexpr Vec2& operator-=(const Vec2& rhs) noexcept {
        x -= rhs.x; y -= rhs.y; return *this;
    }

    constexpr Vec2& operator*=(f64 scalar) noexcept {
        x *= scalar; y *= scalar; return *this;
    }

    constexpr Vec2& operator/=(f64 scalar) noexcept {
        x /= scalar; y /= scalar; return *this;
    }

    [[nodiscard]] inline f64 LengthSquared() const noexcept { return x * x + y * y; }
    [[nodiscard]] inline f64 Length() const noexcept { return std::sqrt(LengthSquared()); }

    [[nodiscard]] inline Vec2 Normalized(f64 epsilon = kEpsilon) const noexcept {
        const f64 len = Length();
        return len <= epsilon ? Vec2{} : (*this / len);
    }
};

[[nodiscard]] constexpr Vec2 operator*(f64 scalar, const Vec2& v) noexcept {
    return v * scalar;
}

[[nodiscard]] constexpr bool operator==(const Vec2& lhs, const Vec2& rhs) noexcept {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

[[nodiscard]] constexpr bool operator!=(const Vec2& lhs, const Vec2& rhs) noexcept {
    return !(lhs == rhs);
}

[[nodiscard]] constexpr f64 Dot(const Vec2& a, const Vec2& b) noexcept {
    return a.x * b.x + a.y * b.y;
}

[[nodiscard]] constexpr f64 Cross(const Vec2& a, const Vec2& b) noexcept {
    return a.x * b.y - a.y * b.x;
}

[[nodiscard]] inline f64 DistanceSquared(const Vec2& a, const Vec2& b) noexcept {
    return (a - b).LengthSquared();
}

[[nodiscard]] inline f64 Distance(const Vec2& a, const Vec2& b) noexcept {
    return (a - b).Length();
}

[[nodiscard]] inline Vec2 Normalize(const Vec2& v, f64 epsilon = kEpsilon) noexcept {
    return v.Normalized(epsilon);
}

} // namespace svec
