#include "svec/image/scanline_integral_stats.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "svec/math/scalar.h"

namespace svec {
namespace {

void BuildRowPrefixesForChannel(const ImageOKLaba& image,
                                std::vector<ScanlineIntegralStats::RowPrefix>& rows,
                                f64 (*getter)(const ColorOKLaba&)) {
    rows.resize(static_cast<std::size_t>(image.Height()));
    for (i32 y = 0; y < image.Height(); ++y) {
        auto& row = rows[static_cast<std::size_t>(y)];
        row.sum.assign(static_cast<std::size_t>(image.Width() + 1), 0.0);
        row.sum_xz.assign(static_cast<std::size_t>(image.Width() + 1), 0.0);
        row.sum_z2.assign(static_cast<std::size_t>(image.Width() + 1), 0.0);

        f64 acc_z = 0.0;
        f64 acc_xz = 0.0;
        f64 acc_z2 = 0.0;
        for (i32 x = 0; x < image.Width(); ++x) {
            const f64 z = getter(image.At(x, y));
            acc_z += z;
            acc_xz += static_cast<f64>(x) * z;
            acc_z2 += z * z;
            const std::size_t idx = static_cast<std::size_t>(x + 1);
            row.sum[idx] = acc_z;
            row.sum_xz[idx] = acc_xz;
            row.sum_z2[idx] = acc_z2;
        }
    }
}

f64 GetL(const ColorOKLaba& c) { return c.L; }
f64 Geta(const ColorOKLaba& c) { return c.a; }
f64 Getb(const ColorOKLaba& c) { return c.b; }
f64 GetAlpha(const ColorOKLaba& c) { return c.alpha; }

} // namespace

void ScanlineIntegralStats::Build(const ImageOKLaba& image) {
    if (!image.IsValid()) {
        throw std::runtime_error("ScanlineIntegralStats::Build: invalid image.");
    }
    m_size = image.Size();
    BuildRowPrefixesForChannel(image, m_L_rows, GetL);
    BuildRowPrefixesForChannel(image, m_a_rows, Geta);
    BuildRowPrefixesForChannel(image, m_b_rows, Getb);
    BuildRowPrefixesForChannel(image, m_alpha_rows, GetAlpha);
}

bool ScanlineIntegralStats::IsValid() const noexcept {
    return m_size.IsValid() &&
           m_L_rows.size() == static_cast<std::size_t>(m_size.height) &&
           m_a_rows.size() == static_cast<std::size_t>(m_size.height) &&
           m_b_rows.size() == static_cast<std::size_t>(m_size.height) &&
           m_alpha_rows.size() == static_cast<std::size_t>(m_size.height);
}

f64 ScanlineIntegralStats::PrefixDiff(const std::vector<f64>& prefix, i32 x0, i32 x1_exclusive) {
    const i32 lo = Max(0, Min(x0, x1_exclusive));
    const i32 hi = Max(0, Max(x0, x1_exclusive));
    const std::size_t slo = static_cast<std::size_t>(lo);
    const std::size_t shi = static_cast<std::size_t>(hi);
    return prefix[shi] - prefix[slo];
}

f64 ScanlineIntegralStats::SumInt(i32 x0, i32 x1_inclusive) noexcept {
    if (x1_inclusive < x0) return 0.0;
    const f64 a = static_cast<f64>(x0);
    const f64 b = static_cast<f64>(x1_inclusive);
    return (a + b) * (b - a + 1.0) * 0.5;
}

f64 ScanlineIntegralStats::SumIntSquares(i32 x0, i32 x1_inclusive) noexcept {
    if (x1_inclusive < x0) return 0.0;
    auto f = [](f64 n) {
        return n * (n + 1.0) * (2.0 * n + 1.0) / 6.0;
    };
    return f(static_cast<f64>(x1_inclusive)) - f(static_cast<f64>(x0 - 1));
}

bool ScanlineIntegralStats::ComputeRowSpan(
    const Vec2& p0,
    const Vec2& p1,
    const Vec2& p2,
    i32 y,
    i32 width,
    i32& out_x0,
    i32& out_x1) {

    const f64 yy = static_cast<f64>(y);

    f64 min_x = 0.0;
    f64 max_x = 0.0;
    bool has_any = false;

    auto add_x = [&](f64 x) {
        if (!has_any) {
            min_x = x;
            max_x = x;
            has_any = true;
        }
        else {
            min_x = Min(min_x, x);
            max_x = Max(max_x, x);
        }
        };

    auto add_intersection = [&](const Vec2& a, const Vec2& b) {
        // Горизонтальное ребро
        if (std::abs(a.y - b.y) <= 1e-12) {
            if (std::abs(yy - a.y) <= 0.5) {
                add_x(Min(a.x, b.x));
                add_x(Max(a.x, b.x));
            }
            return;
        }

        const f64 ymin = Min(a.y, b.y);
        const f64 ymax = Max(a.y, b.y);

        if (yy < ymin || yy >= ymax) {
            return;
        }

        const f64 t = (yy - a.y) / (b.y - a.y);
        add_x(a.x + t * (b.x - a.x));
        };

    add_intersection(p0, p1);
    add_intersection(p1, p2);
    add_intersection(p2, p0);

    if (!has_any) {
        return false;
    }

    out_x0 = Clamp(static_cast<i32>(std::ceil(min_x)), 0, width - 1);
    out_x1 = Clamp(static_cast<i32>(std::floor(max_x)), 0, width - 1);

    return out_x1 >= out_x0;
}

TriangleScanlineMoments ScanlineIntegralStats::AccumulateTriangle(const Vec2& p0, const Vec2& p1, const Vec2& p2) const {
    if (!IsValid()) {
        throw std::runtime_error("ScanlineIntegralStats::AccumulateTriangle: stats are invalid.");
    }

    TriangleScanlineMoments out{};
    const i32 y0 = Clamp(static_cast<i32>(std::ceil(Min(p0.y, Min(p1.y, p2.y)))), 0, m_size.height - 1);
    const i32 y1 = Clamp(static_cast<i32>(std::floor(Max(p0.y, Max(p1.y, p2.y)))), 0, m_size.height - 1);

    for (i32 y = y0; y <= y1; ++y) {
        i32 x0 = 0;
        i32 x1 = -1;
        if (!ComputeRowSpan(p0, p1, p2, y, m_size.width, x0, x1)) continue;
        const f64 n = static_cast<f64>(x1 - x0 + 1);
        const f64 sum_x = SumInt(x0, x1);
        const f64 sum_x2 = SumIntSquares(x0, x1);
        const f64 yf = static_cast<f64>(y);

        out.count += n;
        out.sum_x += sum_x;
        out.sum_y += yf * n;
        out.sum_x2 += sum_x2;
        out.sum_y2 += yf * yf * n;
        out.sum_xy += yf * sum_x;

        const std::size_t ry = static_cast<std::size_t>(y);
        const i32 x1e = x1 + 1;

        const f64 sumL = PrefixDiff(m_L_rows[ry].sum, x0, x1e);
        const f64 sumA = PrefixDiff(m_a_rows[ry].sum, x0, x1e);
        const f64 sumB = PrefixDiff(m_b_rows[ry].sum, x0, x1e);
        const f64 sumAlpha = PrefixDiff(m_alpha_rows[ry].sum, x0, x1e);

        out.sum_L += sumL;
        out.sum_a += sumA;
        out.sum_b += sumB;
        out.sum_alpha += sumAlpha;

        out.sum_xL += PrefixDiff(m_L_rows[ry].sum_xz, x0, x1e);
        out.sum_xa += PrefixDiff(m_a_rows[ry].sum_xz, x0, x1e);
        out.sum_xb += PrefixDiff(m_b_rows[ry].sum_xz, x0, x1e);
        out.sum_xalpha += PrefixDiff(m_alpha_rows[ry].sum_xz, x0, x1e);

        out.sum_yL += yf * sumL;
        out.sum_ya += yf * sumA;
        out.sum_yb += yf * sumB;
        out.sum_yalpha += yf * sumAlpha;

        out.sum_L2 += PrefixDiff(m_L_rows[ry].sum_z2, x0, x1e);
        out.sum_a2 += PrefixDiff(m_a_rows[ry].sum_z2, x0, x1e);
        out.sum_b2 += PrefixDiff(m_b_rows[ry].sum_z2, x0, x1e);
        out.sum_alpha2 += PrefixDiff(m_alpha_rows[ry].sum_z2, x0, x1e);
    }

    return out;
}

} // namespace svec
