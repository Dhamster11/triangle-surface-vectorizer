#pragma once

#include <vector>

#include "svec/core/types.h"
#include "svec/edge/edge_map.h"

namespace svec {

class IntegralMap {
public:
    IntegralMap() = default;

    void Resize(ImageSize source_size);
    [[nodiscard]] bool IsValid() const noexcept;

    [[nodiscard]] i32 SourceWidth() const noexcept { return m_source_size.width; }
    [[nodiscard]] i32 SourceHeight() const noexcept { return m_source_size.height; }

    void BuildFromScalars(ImageSize source_size, const std::vector<f64>& values);
    void BuildFromEdgeMap(const EdgeMap& map);

    [[nodiscard]] f64 Prefix(i32 x, i32 y) const;
    [[nodiscard]] f64 SumRect(i32 x0, i32 y0, i32 x1, i32 y1) const;
    [[nodiscard]] f64 MeanRect(i32 x0, i32 y0, i32 x1, i32 y1) const;

private:
    [[nodiscard]] std::size_t IndexOf(i32 x, i32 y) const noexcept {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(m_pitch) + static_cast<std::size_t>(x);
    }

    ImageSize m_source_size{};
    i32 m_pitch = 0;
    std::vector<f64> m_prefix{};
};

} // namespace svec
