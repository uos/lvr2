#include "lvr2/registration/AABB.hpp"

namespace lvr2
{

AABB::AABB()
    : m_count(0)
{
    m_min.setConstant(std::numeric_limits<double>::infinity());
    m_max.setConstant(-std::numeric_limits<double>::infinity());
    m_sum.setConstant(0.0);
}

const Vector3d& AABB::min() const
{
    return m_min;
}

const Vector3d& AABB::max() const
{
    return m_max;
}

Vector3d AABB::avg() const
{
    return m_sum / m_count;
}

size_t AABB::count() const
{
    return m_count;
}

double AABB::difference(int axis) const
{
    return m_max(axis) - m_min(axis);
}

int AABB::longestAxis() const
{
    int splitAxis = 0;
    for (int axis = 1; axis < 3; axis++)
    {
        if (difference(axis) > difference(splitAxis))
        {
            splitAxis = axis;
        }
    }
    return splitAxis;
}

} // namespace lvr2