#pragma once

#include <vector>

#include "svec/core/types.h"
#include "svec/image/image.h"
#include "svec/math/vec2.h"

namespace svec {

struct TriangleScanlineMoments {
    f64 count = 0.0;
    f64 sum_x = 0.0;
    f64 sum_y = 0.0;
    f64 sum_x2 = 0.0;
    f64 sum_y2 = 0.0;
    f64 sum_xy = 0.0;

    f64 sum_L = 0.0;
    f64 sum_a = 0.0;
    f64 sum_b = 0.0;
    f64 sum_alpha = 0.0;

    f64 sum_xL = 0.0;
    f64 sum_xa = 0.0;
    f64 sum_xb = 0.0;
    f64 sum_xalpha = 0.0;

    f64 sum_yL = 0.0;
    f64 sum_ya = 0.0;
    f64 sum_yb = 0.0;
    f64 sum_yalpha = 0.0;

    f64 sum_L2 = 0.0;
    f64 sum_a2 = 0.0;
    f64 sum_b2 = 0.0;
    f64 sum_alpha2 = 0.0;

    [[nodiscard]] bool IsEmpty() const noexcept { return count <= 0.0; }
};

class ScanlineIntegralStats {
public:
    ScanlineIntegralStats() = default;
    explicit ScanlineIntegralStats(const ImageOKLaba& image) { Build(image); }

    void Build(const ImageOKLaba& image);
    [[nodiscard]] bool IsValid() const noexcept;
    [[nodiscard]] ImageSize Size() const noexcept { return m_size; }

    [[nodiscard]] TriangleScanlineMoments AccumulateTriangle(const Vec2& p0, const Vec2& p1, const Vec2& p2) const;

public:
    struct RowPrefix {
        std::vector<f64> sum;
        std::vector<f64> sum_xz;
        std::vector<f64> sum_z2;
    };

private:
    ImageSize m_size{};
    std::vector<RowPrefix> m_L_rows{};
    std::vector<RowPrefix> m_a_rows{};
    std::vector<RowPrefix> m_b_rows{};
    std::vector<RowPrefix> m_alpha_rows{};

    [[nodiscard]] static f64 PrefixDiff(const std::vector<f64>& prefix, i32 x0, i32 x1_exclusive);
    [[nodiscard]] static f64 SumInt(i32 x0, i32 x1_inclusive) noexcept;
    [[nodiscard]] static f64 SumIntSquares(i32 x0, i32 x1_inclusive) noexcept;
    [[nodiscard]] static bool ComputeRowSpan(const Vec2& p0, const Vec2& p1, const Vec2& p2, i32 y, i32 width, i32& out_x0, i32& out_x1);
};

} // namespace svec
