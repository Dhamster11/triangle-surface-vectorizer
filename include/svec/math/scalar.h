#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace svec {

inline constexpr double kEpsilon = 1e-9;
inline constexpr float  kEpsilonF = 1e-6f;
inline constexpr double kPi = 3.141592653589793238462643383279502884;

template <typename T>
[[nodiscard]] constexpr T Abs(T value) noexcept {
    return value < T{} ? -value : value;
}

template <typename T>
[[nodiscard]] constexpr T Min(T a, T b) noexcept {
    return a < b ? a : b;
}

template <typename T>
[[nodiscard]] constexpr T Max(T a, T b) noexcept {
    return a > b ? a : b;
}

template <typename T>
[[nodiscard]] constexpr T Clamp(T value, T minValue, T maxValue) noexcept {
    return value < minValue ? minValue : (value > maxValue ? maxValue : value);
}

template <typename T>
[[nodiscard]] constexpr T Saturate(T value) noexcept {
    return Clamp(value, T{0}, T{1});
}

template <typename T, typename U>
[[nodiscard]] constexpr auto Lerp(const T& a, const T& b, U t) noexcept {
    return a + (b - a) * t;
}

template <typename T>
[[nodiscard]] inline bool NearlyEqual(T a, T b, T epsilon = static_cast<T>(kEpsilon)) noexcept {
    static_assert(std::is_floating_point_v<T>, "NearlyEqual expects a floating point type");
    return std::abs(a - b) <= epsilon;
}

template <typename T>
[[nodiscard]] inline T Sqrt(T value) noexcept {
    static_assert(std::is_floating_point_v<T>, "Sqrt expects a floating point type");
    return std::sqrt(value);
}

template <typename T>
[[nodiscard]] inline T Rsqrt(T value) noexcept {
    static_assert(std::is_floating_point_v<T>, "Rsqrt expects a floating point type");
    return static_cast<T>(1) / std::sqrt(value);
}

template <typename T>
[[nodiscard]] inline T SafeDiv(T numerator, T denominator, T fallback = T{}) noexcept {
    static_assert(std::is_floating_point_v<T>, "SafeDiv expects a floating point type");
    return std::abs(denominator) <= std::numeric_limits<T>::epsilon() ? fallback : numerator / denominator;
}

} // namespace svec
