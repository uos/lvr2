#ifndef POINTBUFFER2_HPP
#define POINTBUFFER2_HPP

#include <lvr2/io/DataStruct.hpp>

#include <map>
#include <string>

#include <boost/shared_array.hpp>


namespace lvr2
{

template<typename T>
class AttributeChannel
{
public:
    using DataPtr = boost::shared_array<T>;

    AttributeChannel(size_t n, unsigned width)
        : m_width(width), m_numAttributes(n)
    {
        m_data = DataPtr(new T[m_numAttributes * width]);
    }

    AttributeChannel(size_t n, unsigned width, DataPtr ptr)
        : m_numAttributes(n),
          m_width(width),
          m_data(ptr)
    {

    }

    DataPtr     get() { return m_data;}
    unsigned    width() { return m_width;}
    size_t      n() { return m_numAttributes;}

private:
    size_t          m_numAttributes;
    unsigned        m_width;
    DataPtr         m_data;
};

template<typename T>
class AttributeProxy
{
public:
    template<typename BaseVecT>
    AttributeProxy operator=(const BaseVecT& v)
    {
        if( m_ptr && (m_w > 2) && (&v != this))
        {
            m_ptr[0] = v.x;
            m_ptr[1] = v.y;
            m_ptr[2] = v.z;
        }
        return *this;
    }

    AttributeProxy(T* pos = nullptr, unsigned w = 0) : m_ptr(pos), m_w(0) {}
private:

    T*              m_ptr;
    unsigned        m_w;
};


class PointBuffer2
{
public:
    using FloatChannel = AttributeChannel<float>;
    using UCharChannel = AttributeChannel<unsigned char>;

    using FloatProxy = AttributeProxy<float>;
    using UCharProxy = AttributeProxy<unsigned char>;

    using FloatChannelPtr = std::shared_ptr<FloatChannel>;
    using UCharChannelPtr = std::shared_ptr<UCharChannel>;

    PointBuffer2(floatArr points, size_t n);
    PointBuffer2(floatArr points, floatArr normals, size_t n);

    void createFloatChannel(
            FloatChannelPtr data,
            std::string name,
            size_t n,
            unsigned width);

    void createUCharChannel(
            UCharChannelPtr data,
            std::string name,
            size_t n,
            unsigned width);

    bool hasUCharChannel(std::string name);
    bool hasFloatChannel(std::string name);

    unsigned ucharChannelWidth(std::string name);
    unsigned floatChannelWidth(std::string name);

    FloatProxy getFloatHandle(int idx, const std::string& name);
    UCharProxy getUCharHandle(int idx, const std::string& name);

    FloatProxy point(int idx);
    FloatProxy normal(int idx);

private:

    void addFloatChannel(FloatChannelPtr data, std::string name);
    void addUCharChannel(UCharChannelPtr data, std::string name);

    // Point channel, 'cached' to allow faster access
    FloatChannelPtr     m_points;

    // Number of points in buffer
    size_t              m_numPoints;

    std::map<std::string, FloatChannelPtr>  m_floatChannels;
    std::map<std::string, UCharChannelPtr>  m_ucharChannels;

    using FloatChannelMap = std::map<std::string, FloatChannelPtr>;
    using UCharChannelMap = std::map<std::string, UCharChannelPtr>;

};

} // namespace lvr2

#endif // POINTBUFFER2_HPP
