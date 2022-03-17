#include "Triangle.hpp"

namespace lvr2
{

template <typename Vec, typename Scalar>
Triangle<Vec, Scalar>::Triangle(Vec a, Vec b, Vec c)
    : m_a(a), m_b(b), m_c(c)
{
    this->init();
}


template <typename Vec, typename Scalar>
Triangle<Vec, Scalar>::Triangle(const std::array<Vec, 3UL>& array)
    : Triangle(array[0], array[1], array[2])
{}

template <typename Vec, typename Scalar>
inline void Triangle<Vec, Scalar>::init()
{
    // Calculate the area using herons formula
    Scalar lenA = (m_b - m_a).norm();
    Scalar lenB = (m_c - m_b).norm();
    Scalar lenC = (m_a - m_c).norm();
    // Semi perimeter of the triangle
    Scalar sp = (lenA + lenB + lenC) / 2;
    m_area = std::sqrt((sp - lenA) * (sp - lenB) * (sp - lenC) * sp);

    m_areaInverse = (Scalar) 1.0 / m_area;
}


/**
 * Explanation of barycentric coordinates 
 * https://mathworld.wolfram.com/BarycentricCoordinates.html
 */
template <typename Vec, typename Scalar>
typename Triangle<Vec, Scalar>::BarycentricCoords Triangle<Vec, Scalar>::barycentric(Vec point) const
{
    // A is area opposite of point a
    auto A = Triangle(m_b, m_c, point);
    auto B = Triangle(m_c, m_a, point);
    auto C = Triangle(m_a, m_b, point);

    // The barycentrics are normalized with respect to this triangles area
    Scalar t1 = A.area() * m_areaInverse;
    Scalar t2 = B.area() * m_areaInverse;
    Scalar t3 = C.area() * m_areaInverse;
    
    return BarycentricCoords(t1, t2, t3);
}

template <typename Vec, typename Scalar>
Vec Triangle<Vec, Scalar>::point(BarycentricCoords point) const
{
    return point.x() * m_a + point.y() * m_b + point.z() * m_c;
}

template <typename Vec, typename Scalar>
std::pair<Vec, Vec> Triangle<Vec, Scalar>::getAABoundingBox() const
{
    Vec min, max;
    for (size_t i = 0; i < min.rows(); i++)
    {
        min[i] = std::min({m_a[i], m_b[i], m_c[i]});
        max[i] = std::max({m_a[i], m_b[i], m_c[i]});
    }
    return std::make_pair(min, max);
}

} // namespace lvr2

