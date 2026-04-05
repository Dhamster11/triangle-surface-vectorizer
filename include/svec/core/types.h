#pragma once

#include <cstdint>
#include <string_view>

namespace svec {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using f32 = float;
using f64 = double;

struct ImageSize {
    i32 width  = 0;
    i32 height = 0;

    [[nodiscard]] constexpr bool IsValid() const noexcept {
        return width > 0 && height > 0;
    }

    [[nodiscard]] constexpr i64 PixelCount() const noexcept {
        return static_cast<i64>(width) * static_cast<i64>(height);
    }
};

struct Version {
    i32 major = 0;
    i32 minor = 0;
    i32 patch = 0;
};

inline constexpr Version kEngineVersion{0, 1, 0};
inline constexpr std::string_view kEngineName = "surface-vectorizer";

} // namespace svec
