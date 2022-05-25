#include "BoundingRectangle.hpp"

namespace lvr2
{

template <typename CoordType>
BaseVector<CoordType> BoundingRectangle<CoordType>::center() const
{
    BaseVector<CoordType> ret;

    CoordType a_offset = (m_maxDistA - m_minDistA) / 2;
    CoordType b_offset = (m_maxDistB - m_minDistB) / 2;

    ret = m_supportVector + m_vec1 * a_offset + m_vec2 * b_offset;

    return ret;
}

} // namespace lvr2