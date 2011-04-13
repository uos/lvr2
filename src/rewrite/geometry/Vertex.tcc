/*
 * Vertex.tcc
 *
 *  Created on: 10.02.2011
 *      Author: Thomas Wiemann
 */

#include <stdexcept>

namespace lssr
{

template<typename CoordType>
CoordType Vertex<CoordType>::operator[](const int &index) const
{
    CoordType ret = 0.0;

    switch(index){
    case 0:
        ret = m_x;
        break;
    case 1:
        ret = m_y;
        break;
    case 2:
        ret = m_z;
        break;
    default:
        throw std::overflow_error("Access index out of range.");
    }
    return ret;
}

template<typename CoordType>
CoordType& Vertex<CoordType>::operator[](const int &index)
{
    switch(index){
    case 0:
        return m_x;
    case 1:
        return m_y;
    case 2:
        return m_z;
    default:
        throw std::overflow_error("Access index out of range.");
    }
}

template<typename CoordType>
bool Vertex<CoordType>::operator==(const Vertex &other) const
{
    return fabs(m_x - other.m_x) <= Vertex::epsilon &&
           fabs(m_y - other.m_y) <= Vertex::epsilon &&
           fabs(m_z - other.m_z) <= Vertex::epsilon;
}

template<typename CoordType>
void Vertex<CoordType>::operator/=(const CoordType &scale)
{
    if(scale != 0)
    {
        m_x /= scale;
        m_y /= scale;
        m_z /= scale;
    }
    else
    {
        m_x = m_y = m_z = 0;
    }
}

template<typename CoordType>
void Vertex<CoordType>::operator*=(const CoordType &scale)
      {
    m_x *= scale;
    m_y *= scale;
    m_z *= scale;
}

template<typename CoordType>
void Vertex<CoordType>::operator+=(const Vertex &other)
{
    m_x += other.m_x;
    m_y += other.m_y;
    m_z += other.m_z;
}

template<typename CoordType>
void Vertex<CoordType>::operator-=(const Vertex &other)
{
    m_x -= other.m_x;
    m_y -= other.m_y;
    m_z -= other.m_z;
}

template<typename CoordType>
Vertex<CoordType> Vertex<CoordType>::operator-(const Vertex &other) const
{
    return Vertex<CoordType>(m_x - other.m_x, m_y - other.m_y, m_z - other.m_z);
}

template<typename CoordType>
Vertex<CoordType> Vertex<CoordType>::operator+(const Vertex &other) const
{
    return Vertex<CoordType>(m_x + other.m_x, m_y + other.m_y, m_z + other.m_z);
}

template<typename CoordType>
Vertex<CoordType> Vertex<CoordType>::operator*(const CoordType &scale) const
{
    return Vertex<CoordType>(m_x * scale, m_y * scale, m_z * scale);
}

template<typename CoordType>
CoordType Vertex<CoordType>::operator*(const Vertex<CoordType> &other) const
{
    return m_x * other.m_x + m_y * other.m_y + m_z * other.m_z;
}

template<typename CoordType>
void Vertex<CoordType>::crossTo(const Vertex<CoordType>  &other)
{
    m_x = m_y * other.m_z - m_z * other.m_y;
    m_y = m_z * other.m_x - m_x * other.m_z;
    m_z = m_x * other.m_y - m_y * other.m_x;
}

template<typename CoordType>
void Vertex<CoordType>::rotate(const Matrix4<CoordType> &m)
{
    m_x = m_x * m[0 ] + m_y * m[1 ] + m_z * m[2 ];
    m_y = m_x * m[4 ] + m_y * m[5 ] + m_z * m[6 ];
    m_z = m_x * m[8 ] + m_y * m[9 ] + m_z * m[10];
}

template<typename CoordType>
Vertex<CoordType> Vertex<CoordType>::cross(const Vertex<CoordType> &other) const
{
    CoordType tx = m_y * other.m_z - m_z * other.m_y;
    CoordType ty = m_z * other.m_x - m_x * other.m_z;
    CoordType tz = m_x * other.m_y - m_y * other.m_x;

    return Vertex<CoordType>(tx, ty, tz);
}

template<typename CoordType>
CoordType Vertex<CoordType>::length()
{
    return sqrt(m_x * m_x + m_y * m_y + m_z * m_z);
}


} // namespace lssr
