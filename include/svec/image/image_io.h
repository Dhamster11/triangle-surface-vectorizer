#pragma once

#include <string>

#include "svec/image/image.h"

namespace svec {

struct ImageLoadOptions {
    bool flipVertical = false;
};

struct ImageSaveOptions {
    bool flipVertical = false;
    i32 jpgQuality = 95;
};

struct ImageLoadResult {
    ImageRGBA8 image{};
    std::string error{};

    [[nodiscard]] bool Ok() const noexcept {
        return error.empty() && image.IsValid();
    }
};

[[nodiscard]] ImageLoadResult LoadRGBA8FromFile(const std::string& path, const ImageLoadOptions& options = {});
[[nodiscard]] bool SaveRGBA8ToFile(const std::string& path, const ImageRGBA8& image, const ImageSaveOptions& options = {}, std::string* outError = nullptr);

} // namespace svec
