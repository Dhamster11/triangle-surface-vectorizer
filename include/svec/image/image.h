#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "svec/core/types.h"
#include "svec/math/color.h"

namespace svec {

struct ColorRGBA8 {
    u8 r = 0;
    u8 g = 0;
    u8 b = 0;
    u8 a = 255;
};

class ImageRGBA8 {
public:
    ImageRGBA8() = default;
    explicit ImageRGBA8(ImageSize size, ColorRGBA8 fill = {}) {
        Resize(size, fill);
    }

    void Resize(ImageSize size, ColorRGBA8 fill = {}) {
        if (!size.IsValid()) {
            m_size = {};
            m_pixels.clear();
            return;
        }
        m_size = size;
        m_pixels.assign(static_cast<std::size_t>(size.PixelCount()), fill);
    }

    [[nodiscard]] bool IsValid() const noexcept {
        return m_size.IsValid() && static_cast<i64>(m_pixels.size()) == m_size.PixelCount();
    }

    [[nodiscard]] ImageSize Size() const noexcept { return m_size; }
    [[nodiscard]] i32 Width() const noexcept { return m_size.width; }
    [[nodiscard]] i32 Height() const noexcept { return m_size.height; }
    [[nodiscard]] i64 PixelCount() const noexcept { return m_size.PixelCount(); }

    [[nodiscard]] const std::vector<ColorRGBA8>& Pixels() const noexcept { return m_pixels; }
    [[nodiscard]] std::vector<ColorRGBA8>& Pixels() noexcept { return m_pixels; }

    [[nodiscard]] const ColorRGBA8& At(i32 x, i32 y) const {
        return m_pixels.at(IndexOf(x, y));
    }

    [[nodiscard]] ColorRGBA8& At(i32 x, i32 y) {
        return m_pixels.at(IndexOf(x, y));
    }

private:
    [[nodiscard]] std::size_t IndexOf(i32 x, i32 y) const {
        if (x < 0 || y < 0 || x >= m_size.width || y >= m_size.height) {
            throw std::out_of_range("ImageRGBA8::At index out of range");
        }
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(m_size.width) + static_cast<std::size_t>(x);
    }

    ImageSize m_size{};
    std::vector<ColorRGBA8> m_pixels{};
};

class ImageOKLaba {
public:
    ImageOKLaba() = default;
    explicit ImageOKLaba(ImageSize size, ColorOKLaba fill = {}) {
        Resize(size, fill);
    }

    void Resize(ImageSize size, ColorOKLaba fill = {}) {
        if (!size.IsValid()) {
            m_size = {};
            m_pixels.clear();
            return;
        }
        m_size = size;
        m_pixels.assign(static_cast<std::size_t>(size.PixelCount()), fill);
    }

    [[nodiscard]] bool IsValid() const noexcept {
        return m_size.IsValid() && static_cast<i64>(m_pixels.size()) == m_size.PixelCount();
    }

    [[nodiscard]] ImageSize Size() const noexcept { return m_size; }
    [[nodiscard]] i32 Width() const noexcept { return m_size.width; }
    [[nodiscard]] i32 Height() const noexcept { return m_size.height; }
    [[nodiscard]] i64 PixelCount() const noexcept { return m_size.PixelCount(); }

    [[nodiscard]] const std::vector<ColorOKLaba>& Pixels() const noexcept { return m_pixels; }
    [[nodiscard]] std::vector<ColorOKLaba>& Pixels() noexcept { return m_pixels; }

    [[nodiscard]] const ColorOKLaba& At(i32 x, i32 y) const {
        return m_pixels.at(IndexOf(x, y));
    }

    [[nodiscard]] ColorOKLaba& At(i32 x, i32 y) {
        return m_pixels.at(IndexOf(x, y));
    }

private:
    [[nodiscard]] std::size_t IndexOf(i32 x, i32 y) const {
        if (x < 0 || y < 0 || x >= m_size.width || y >= m_size.height) {
            throw std::out_of_range("ImageOKLaba::At index out of range");
        }
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(m_size.width) + static_cast<std::size_t>(x);
    }

    ImageSize m_size{};
    std::vector<ColorOKLaba> m_pixels{};
};

} // namespace svec
