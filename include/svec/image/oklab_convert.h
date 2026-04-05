#pragma once

#include "svec/image/image.h"

namespace svec {

[[nodiscard]] f64 SrgbToLinear(f64 srgb01) noexcept;
[[nodiscard]] f64 LinearToSrgb(f64 linear01) noexcept;

[[nodiscard]] ColorRGBf  DecodeSRGB(const ColorRGBA8& c) noexcept;
[[nodiscard]] ColorRGBA8 EncodeSRGB8(const ColorRGBAf& c) noexcept;

[[nodiscard]] ColorOKLab  LinearSRGBToOKLab(const ColorRGBf& linear) noexcept;
[[nodiscard]] ColorRGBf   OKLabToLinearSRGB(const ColorOKLab& oklab) noexcept;

[[nodiscard]] ColorOKLaba RGBA8ToOKLaba(const ColorRGBA8& rgba) noexcept;
[[nodiscard]] ColorRGBA8  OKLabaToRGBA8(const ColorOKLaba& oklab) noexcept;

[[nodiscard]] ImageOKLaba ConvertToOKLab(const ImageRGBA8& image);
[[nodiscard]] ImageRGBA8  ConvertToRGBA8(const ImageOKLaba& image);

} // namespace svec
