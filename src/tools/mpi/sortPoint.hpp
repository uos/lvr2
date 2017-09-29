#ifndef LAS_VEGAS_SORTPOINT_HPP
#define LAS_VEGAS_SORTPOINT_HPP

namespace lvr
{
template<typename pointType>
class sortPoint
{
public:
    sortPoint() : m_first(0), m_pos(0)
    {

    }
    sortPoint(pointType* p, size_t pos = 0) : m_first(p), m_pos(pos)
    {

    }


    bool operator<(const sortPoint& rhs) const
    {
        if(x() != rhs.x()) return x() < rhs.x();
        if(y() != rhs.y()) return y() < rhs.y();
        return z() < rhs.z();
    }
    bool operator> (const sortPoint& rhs) const
    {
        return ! (this->operator <(rhs));
    }


    bool operator==(const sortPoint& rhs) const
    {
//        float dist = ( ((x() - rhs.x())*(x() - rhs.x())) + ((y() - rhs.y())*(y() - rhs.y())) + ((z() - rhs.z())*(z() - rhs.z())) );
//        if(fabs(dist) < 0.00001) return true;
//        return false;
        return std::fabs((float)x() - rhs.x()) <= tollerance &&
               std::fabs((float)y() - rhs.y()) <= tollerance &&
               std::fabs((float)z() - rhs.z()) <= tollerance;
    }

    const pointType x() const{return m_first[0];}
    const pointType y() const{return m_first[1];}
    const pointType z() const{return m_first[2];}
    const size_t id() const{return m_pos;}
    static float tollerance;
private:
    pointType * m_first;
    size_t m_pos;

};

template<typename pointType>
float lvr::sortPoint<pointType>::tollerance = 0.0005;

template<>
inline bool sortPoint<unsigned int>::operator ==(const sortPoint& rhs) const
{
    return x() == rhs.x() && y() == rhs.y() && z() == rhs.z();

}


}
#endif //LAS_VEGAS_SORTPOINT_HPP
