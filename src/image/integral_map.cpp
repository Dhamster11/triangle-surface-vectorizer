#include "svec/image/integral_map.h"

#include <stdexcept>

#include "svec/math/scalar.h"

namespace svec {

void IntegralMap::Resize(ImageSize source_size) {
    if (!source_size.IsValid()) {
        m_source_size = {};
        m_pitch = 0;
        m_prefix.clear();
        return;
    }
    m_source_size = source_size;
    m_pitch = source_size.width + 1;
    const std::size_t h = static_cast<std::size_t>(source_size.height + 1);
    m_prefix.assign(static_cast<std::size_t>(m_pitch) * h, 0.0);
}

bool IntegralMap::IsValid() const noexcept {
    if (!m_source_size.IsValid() || m_pitch != m_source_size.width + 1) {
        return false;
    }
    const std::size_t expected = static_cast<std::size_t>(m_pitch) * static_cast<std::size_t>(m_source_size.height + 1);
    return m_prefix.size() == expected;
}

void IntegralMap::BuildFromScalars(ImageSize source_size, const std::vector<f64>& values) {
    if (!source_size.IsValid()) {
        throw std::runtime_error("IntegralMap::BuildFromScalars: invalid source size.");
    }
    if (static_cast<i64>(values.size()) != source_size.PixelCount()) {
        throw std::runtime_error("IntegralMap::BuildFromScalars: value count mismatch.");
    }

    Resize(source_size);
    for (i32 y = 0; y < source_size.height; ++y) {
        f64 row_accum = 0.0;
        for (i32 x = 0; x < source_size.width; ++x) {
            row_accum += values[static_cast<std::size_t>(y) * static_cast<std::size_t>(source_size.width) + static_cast<std::size_t>(x)];
            m_prefix[IndexOf(x + 1, y + 1)] = m_prefix[IndexOf(x + 1, y)] + row_accum;
        }
    }
}

void IntegralMap::BuildFromEdgeMap(const EdgeMap& map) {
    BuildFromScalars(map.Size(), map.Pixels());
}

f64 IntegralMap::Prefix(i32 x, i32 y) const {
    if (!IsValid()) {
        throw std::runtime_error("IntegralMap::Prefix: map is invalid.");
    }
    x = Clamp(x, 0, m_source_size.width);
    y = Clamp(y, 0, m_source_size.height);
    return m_prefix[IndexOf(x, y)];
}

f64 IntegralMap::SumRect(i32 x0, i32 y0, i32 x1, i32 y1) const {
    if (!IsValid()) {
        throw std::runtime_error("IntegralMap::SumRect: map is invalid.");
    }
    x0 = Clamp(x0, 0, m_source_size.width);
    y0 = Clamp(y0, 0, m_source_size.height);
    x1 = Clamp(x1, 0, m_source_size.width);
    y1 = Clamp(y1, 0, m_source_size.height);
    if (x1 < x0) std::swap(x0, x1);
    if (y1 < y0) std::swap(y0, y1);
    return Prefix(x1, y1) - Prefix(x0, y1) - Prefix(x1, y0) + Prefix(x0, y0);
}

f64 IntegralMap::MeanRect(i32 x0, i32 y0, i32 x1, i32 y1) const {
    const i32 w = Max(0, x1 - x0);
    const i32 h = Max(0, y1 - y0);
    const i32 area = w * h;
    if (area <= 0) {
        return 0.0;
    }
    return SumRect(x0, y0, x1, y1) / static_cast<f64>(area);
}

} // namespace svec
