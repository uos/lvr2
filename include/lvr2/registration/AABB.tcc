namespace lvr2
{

template<typename T>
AABB<T>::AABB()
    : m_count(0)
{
    m_min.setConstant(std::numeric_limits<T>::max());
    m_max.setConstant(std::numeric_limits<T>::lowest());
    m_sum.setConstant(0.0);
}

template<typename T>
const Vector3<T>& AABB<T>::min() const
{
    return m_min;
}

template<typename T>
const Vector3<T>& AABB<T>::max() const
{
    return m_max;
}

template<typename T>
Vector3<T> AABB<T>::avg() const
{
    return m_sum / m_count;
}

template<typename T>
size_t AABB<T>::count() const
{
    return m_count;
}

template<typename T>
T AABB<T>::difference(int axis) const
{
    return m_max(axis) - m_min(axis);
}

template<typename T>
int AABB<T>::longestAxis() const
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