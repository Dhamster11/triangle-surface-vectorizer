#include "svec/image/image_io.h"

#include <algorithm>
#include <cctype>
#include <cstring>

#define STBI_WINDOWS_UTF8
#include "stb_image.h"
#include "stb_image_write.h"

namespace svec {

namespace {

[[nodiscard]] std::string ToLowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

[[nodiscard]] std::string ExtensionOf(const std::string& path) {
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return {};
    }
    return ToLowerCopy(path.substr(dot + 1));
}

[[nodiscard]] bool HasSupportedExtension(const std::string& ext) {
    return ext == "png" || ext == "bmp" || ext == "tga" || ext == "jpg" || ext == "jpeg";
}

} // namespace

ImageLoadResult LoadRGBA8FromFile(const std::string& path, const ImageLoadOptions& options) {
    ImageLoadResult result{};

    stbi_set_flip_vertically_on_load(options.flipVertical ? 1 : 0);

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (data == nullptr) {
        result.error = stbi_failure_reason() ? stbi_failure_reason() : "stbi_load failed";
        return result;
    }

    result.image.Resize({width, height});
    const std::size_t pixelCount = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    for (std::size_t i = 0; i < pixelCount; ++i) {
        const std::size_t base = i * 4;
        result.image.Pixels()[i] = ColorRGBA8{data[base + 0], data[base + 1], data[base + 2], data[base + 3]};
    }

    stbi_image_free(data);
    return result;
}

bool SaveRGBA8ToFile(const std::string& path, const ImageRGBA8& image, const ImageSaveOptions& options, std::string* outError) {
    if (!image.IsValid()) {
        if (outError != nullptr) {
            *outError = "SaveRGBA8ToFile: image is invalid";
        }
        return false;
    }

    const std::string ext = ExtensionOf(path);
    if (!HasSupportedExtension(ext)) {
        if (outError != nullptr) {
            *outError = "SaveRGBA8ToFile: unsupported extension";
        }
        return false;
    }

    stbi_flip_vertically_on_write(options.flipVertical ? 1 : 0);

    const int width = image.Width();
    const int height = image.Height();
    const int strideBytes = width * 4;
    const void* raw = static_cast<const void*>(image.Pixels().data());

    int ok = 0;
    if (ext == "png") {
        ok = stbi_write_png(path.c_str(), width, height, 4, raw, strideBytes);
    } else if (ext == "bmp") {
        ok = stbi_write_bmp(path.c_str(), width, height, 4, raw);
    } else if (ext == "tga") {
        ok = stbi_write_tga(path.c_str(), width, height, 4, raw);
    } else {
        ok = stbi_write_jpg(path.c_str(), width, height, 4, raw, options.jpgQuality);
    }

    if (ok == 0) {
        if (outError != nullptr) {
            *outError = "stbi_write_* failed";
        }
        return false;
    }

    return true;
}

} // namespace svec
