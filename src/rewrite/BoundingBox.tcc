/*
 * BoundingBox.cpp
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename T>
BoundingBox<T>::BoundingBox()
{
	T max_val = numeric_limits<T>::max;
	T min_val = numeric_limits<T>::min;

	m_min = Vertex<T>(max_val, max_val, max_val);
	m_max = Vertex<T>(min_val, min_val, min_val);
}

template<typename T>
BoundingBox<T>::BoundingBox(Vertex<T> v1, Vertex<T> v2)
{
	m_min = v1;
	m_max = v2;
}

template<typename T>
BoundingBox<T>::BoundingBox(T x_min, T y_min, T z_min,
		                    T x_max, T y_max, T z_max)
{
	m_min = Vertex<T>(x_min, y_min, z_min);
	m_max = Vertex<T>(x_max, y_max, z_max);
}

template<typename T>
bool BoundingBox<T>::isValid()
{
    T max_val = numeric_limits<T>::max;
    T min_val = numeric_limits<T>::min;

	Vertex<T> v_min(min_val, min_val, min_val);
	Vertex<T> v_max(max_val, max_val, max_val);
	return (m_min != v_max && m_max != v_min);
}

template<typename T>
T BoundingBox<T>::getRadius()
{
	// Shift bounding box to (0,0,0)
	Vertex<T> m_min0 = m_min - m_centroid;
	Vertex<T> m_max0 = m_max - m_centroid;

	// Return radius
	if(m_min0.length() > m_max0.length())
		return m_min0.length();
	else
		return m_max0.length();
}

template<typename T>
inline void BoundingBox<T>::expand(Vertex<T> v)
{
    m_min.x = std::min(v.x, m_min.x);
    m_min.y = std::min(v.y, m_min.y);
    m_min.z = std::min(v.z, m_min.z);

    m_max.x = std::max(v.x, m_max.x);
    m_max.y = std::max(v.y, m_max.y);
    m_max.z = std::max(v.z, m_max.z);

    m_xSize = fabs(m_max.x - m_min.x);
    m_ySize = fabs(m_max.y - m_min.y);
    m_zSize = fabs(m_max.z - m_min.z);

    m_centroid = Vertex<T>(m_max.x - m_min.x,
                      m_max.y - m_min.y,
                      m_max.z - m_min.z);

}

template<typename T>
inline void BoundingBox<T>::expand(T x, T y, T z)
{
    m_min.x = std::min(x, m_min.x);
    m_min.y = std::min(y, m_min.y);
    m_min.z = std::min(z, m_min.z);

    m_max.x = std::max(x, m_max.x);
    m_max.y = std::max(y, m_max.y);
    m_max.z = std::max(z, m_max.z);

    m_xSize = fabs(m_max.x - m_min.x);
    m_ySize = fabs(m_max.y - m_min.y);
    m_zSize = fabs(m_max.z - m_min.z);

    m_centroid = Vertex<T>(m_min.x + 0.5 * m_xSize,
                      m_min.y + 0.5 * m_ySize,
                      m_min.z + 0.5 * m_zSize);

}

template<typename T>
inline void BoundingBox<T>::expand(BoundingBox<T>& bb)
{
    //expand(bb.m_centroid);
    expand(bb.m_min);
    expand(bb.m_max);
}

} // namespace lssr
