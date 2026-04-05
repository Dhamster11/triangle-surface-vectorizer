#include "svec/image/oklab_convert.h"

#include <cmath>

#include "svec/math/scalar.h"

namespace svec {

namespace {

[[nodiscard]] inline f64 CubeRootSigned(f64 x) noexcept {
    return x >= 0.0 ? std::cbrt(x) : -std::cbrt(-x);
}

[[nodiscard]] inline u8 ToByte(f64 x) noexcept {
    const f64 scaled = Saturate(x) * 255.0;
    return static_cast<u8>(std::lround(scaled));
}

} // namespace

f64 SrgbToLinear(f64 srgb01) noexcept {
    const f64 x = Saturate(srgb01);
    if (x <= 0.04045) {
        return x / 12.92;
    }
    return std::pow((x + 0.055) / 1.055, 2.4);
}

f64 LinearToSrgb(f64 linear01) noexcept {
    const f64 x = Saturate(linear01);
    if (x <= 0.0031308) {
        return x * 12.92;
    }
    return 1.055 * std::pow(x, 1.0 / 2.4) - 0.055;
}

ColorRGBf DecodeSRGB(const ColorRGBA8& c) noexcept {
    return {
        SrgbToLinear(static_cast<f64>(c.r) / 255.0),
        SrgbToLinear(static_cast<f64>(c.g) / 255.0),
        SrgbToLinear(static_cast<f64>(c.b) / 255.0)
    };
}

ColorRGBA8 EncodeSRGB8(const ColorRGBAf& c) noexcept {
    return {
        ToByte(LinearToSrgb(c.r)),
        ToByte(LinearToSrgb(c.g)),
        ToByte(LinearToSrgb(c.b)),
        ToByte(c.a)
    };
}

ColorOKLab LinearSRGBToOKLab(const ColorRGBf& linear) noexcept {
    const f64 l = 0.4122214708 * linear.r + 0.5363325363 * linear.g + 0.0514459929 * linear.b;
    const f64 m = 0.2119034982 * linear.r + 0.6806995451 * linear.g + 0.1073969566 * linear.b;
    const f64 s = 0.0883024619 * linear.r + 0.2817188376 * linear.g + 0.6299787005 * linear.b;

    const f64 l_ = CubeRootSigned(l);
    const f64 m_ = CubeRootSigned(m);
    const f64 s_ = CubeRootSigned(s);

    return {
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    };
}

ColorRGBf OKLabToLinearSRGB(const ColorOKLab& oklab) noexcept {
    const f64 l_ = oklab.L + 0.3963377774 * oklab.a + 0.2158037573 * oklab.b;
    const f64 m_ = oklab.L - 0.1055613458 * oklab.a - 0.0638541728 * oklab.b;
    const f64 s_ = oklab.L - 0.0894841775 * oklab.a - 1.2914855480 * oklab.b;

    const f64 l = l_ * l_ * l_;
    const f64 m = m_ * m_ * m_;
    const f64 s = s_ * s_ * s_;

    return {
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    };
}

ColorOKLaba RGBA8ToOKLaba(const ColorRGBA8& rgba) noexcept {
    const ColorRGBf linear = DecodeSRGB(rgba);
    const ColorOKLab lab = LinearSRGBToOKLab(linear);
    return {
        lab.L,
        lab.a,
        lab.b,
        static_cast<f64>(rgba.a) / 255.0
    };
}

ColorRGBA8 OKLabaToRGBA8(const ColorOKLaba& oklab) noexcept {
    const ColorRGBf linear = OKLabToLinearSRGB({oklab.L, oklab.a, oklab.b});
    return EncodeSRGB8({linear.r, linear.g, linear.b, oklab.alpha});
}

ImageOKLaba ConvertToOKLab(const ImageRGBA8& image) {
    ImageOKLaba out(image.Size());
    for (i32 y = 0; y < image.Height(); ++y) {
        for (i32 x = 0; x < image.Width(); ++x) {
            out.At(x, y) = RGBA8ToOKLaba(image.At(x, y));
        }
    }
    return out;
}

ImageRGBA8 ConvertToRGBA8(const ImageOKLaba& image) {
    ImageRGBA8 out(image.Size());
    for (i32 y = 0; y < image.Height(); ++y) {
        for (i32 x = 0; x < image.Width(); ++x) {
            out.At(x, y) = OKLabaToRGBA8(image.At(x, y));
        }
    }
    return out;
}

} // namespace svec
