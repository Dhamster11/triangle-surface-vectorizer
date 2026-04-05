#include <cassert>
#include <cmath>
#include <iostream>

#include "svec/image/image.h"
#include "svec/image/oklab_convert.h"

int main() {
    using namespace svec;

    ImageRGBA8 rgba({2, 2});
    rgba.At(0, 0) = {255,   0,   0, 255};
    rgba.At(1, 0) = {  0, 255,   0, 255};
    rgba.At(0, 1) = {  0,   0, 255, 255};
    rgba.At(1, 1) = {255, 255, 255, 128};

    const ImageOKLaba lab = ConvertToOKLab(rgba);
    assert(lab.IsValid());
    assert(lab.Size().width == 2 && lab.Size().height == 2);

    const ImageRGBA8 roundtrip = ConvertToRGBA8(lab);
    assert(roundtrip.IsValid());

    auto byteDiff = [](u8 a, u8 b) -> int {
        return std::abs(static_cast<int>(a) - static_cast<int>(b));
    };

    for (i32 y = 0; y < rgba.Height(); ++y) {
        for (i32 x = 0; x < rgba.Width(); ++x) {
            const auto src = rgba.At(x, y);
            const auto dst = roundtrip.At(x, y);
            assert(byteDiff(src.r, dst.r) <= 2);
            assert(byteDiff(src.g, dst.g) <= 2);
            assert(byteDiff(src.b, dst.b) <= 2);
            assert(byteDiff(src.a, dst.a) <= 1);
        }
    }

    std::cout << "Step 2 smoke test passed. RGBA8 <-> OKLaba roundtrip is stable.\n";
    return 0;
}
