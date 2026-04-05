#pragma once

#include <cmath>

#include "svec/core/types.h"
#include "svec/math/scalar.h"

namespace svec {

struct Vec3 {
    f64 x = 0.0;
    f64 y = 0.0;
    f64 z = 0.0;

    constexpr Vec3() noexcept = default;
    constexpr Vec3(f64 x_, f64 y_, f64 z_) noexcept : x(x_), y(y_), z(z_) {}

    [[nodiscard]] constexpr Vec3 operator+() const noexcept { return *this; }
    [[nodiscard]] constexpr Vec3 operator-() const noexcept { return {-x, -y, -z}; }

    [[nodiscard]] constexpr Vec3 operator+(const Vec3& rhs) const noexcept { return {x + rhs.x, y + rhs.y, z + rhs.z}; }
    [[nodiscard]] constexpr Vec3 operator-(const Vec3& rhs) const noexcept { return {x - rhs.x, y - rhs.y, z - rhs.z}; }
    [[nodiscard]] constexpr Vec3 operator*(f64 scalar) const noexcept { return {x * scalar, y * scalar, z * scalar}; }
    [[nodiscard]] constexpr Vec3 operator/(f64 scalar) const noexcept { return {x / scalar, y / scalar, z / scalar}; }

    constexpr Vec3& operator+=(const Vec3& rhs) noexcept {
        x += rhs.x; y += rhs.y; z += rhs.z; return *this;
    }

    constexpr Vec3& operator-=(const Vec3& rhs) noexcept {
        x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this;
    }

    constexpr Vec3& operator*=(f64 scalar) noexcept {
        x *= scalar; y *= scalar; z *= scalar; return *this;
    }

    constexpr Vec3& operator/=(f64 scalar) noexcept {
        x /= scalar; y /= scalar; z /= scalar; return *this;
    }

    [[nodiscard]] inline f64 LengthSquared() const noexcept { return x * x + y * y + z * z; }
    [[nodiscard]] inline f64 Length() const noexcept { return std::sqrt(LengthSquared()); }

    [[nodiscard]] inline Vec3 Normalized(f64 epsilon = kEpsilon) const noexcept {
        const f64 len = Length();
        return len <= epsilon ? Vec3{} : (*this / len);
    }
};

[[nodiscard]] constexpr Vec3 operator*(f64 scalar, const Vec3& v) noexcept {
    return v * scalar;
}

[[nodiscard]] constexpr bool operator==(const Vec3& lhs, const Vec3& rhs) noexcept {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

[[nodiscard]] constexpr bool operator!=(const Vec3& lhs, const Vec3& rhs) noexcept {
    return !(lhs == rhs);
}

[[nodiscard]] constexpr f64 Dot(const Vec3& a, const Vec3& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] constexpr Vec3 Cross(const Vec3& a, const Vec3& b) noexcept {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

[[nodiscard]] inline Vec3 Normalize(const Vec3& v, f64 epsilon = kEpsilon) noexcept {
    return v.Normalized(epsilon);
}

} // namespace svec
