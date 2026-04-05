#pragma once

#include <vector>

#include "svec/core/types.h"
#include "svec/image/image.h"
#include "svec/math/vec2.h"

namespace svec {

class EdgeMap {
public:
    EdgeMap() = default;
    explicit EdgeMap(ImageSize size, f64 fill = 0.0) {
        Resize(size, fill);
    }

    void Resize(ImageSize size, f64 fill = 0.0);

    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] ImageSize Size() const noexcept { return m_size; }
    [[nodiscard]] i32 Width() const noexcept { return m_size.width; }
    [[nodiscard]] i32 Height() const noexcept { return m_size.height; }
    [[nodiscard]] i64 PixelCount() const noexcept { return m_size.PixelCount(); }

    [[nodiscard]] const std::vector<f64>& Pixels() const noexcept { return m_values; }
    [[nodiscard]] std::vector<f64>& Pixels() noexcept { return m_values; }

    [[nodiscard]] const f64& At(i32 x, i32 y) const;
    [[nodiscard]] f64& At(i32 x, i32 y);

    [[nodiscard]] f64 MaxValue() const noexcept;

private:
    [[nodiscard]] std::size_t IndexOf(i32 x, i32 y) const;

    ImageSize m_size{};
    std::vector<f64> m_values{};
};

struct EdgeMapOptions {
    f64 weight_L = 1.0;
    f64 weight_a = 1.0;
    f64 weight_b = 1.0;
    f64 weight_alpha = 0.25;

    bool normalize_to_unit = true;
    f64 normalization_epsilon = 1e-12;
};

[[nodiscard]] EdgeMap ComputeEdgeMapSobel(const ImageOKLaba& image, const EdgeMapOptions& options = {});
[[nodiscard]] f64 SampleEdgeMapNearest(const EdgeMap& edge_map, const Vec2& p);
[[nodiscard]] f64 SampleEdgeMapBilinear(const EdgeMap& edge_map, const Vec2& p);
[[nodiscard]] ImageOKLaba RenderEdgeMapPreview(const EdgeMap& edge_map, bool use_heat = true);

} // namespace svec
