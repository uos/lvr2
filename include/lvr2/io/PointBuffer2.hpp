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

    AttributeChannel(size_t n, unsigned width);
    AttributeChannel(size_t n, unsigned width, DataPtr ptr);

    DataPtr     get() { return m_data;}
    unsigned    width() { return m_width;}
    size_t      n() { return m_numAttributes;}

private:
    size_t          m_numAttributes;
    unsigned        m_width;
    DataPtr         m_data;
};

template<typename T>
class AttributeHandle
{
public:
    template<typename BaseVecT>
    AttributeHandle operator=(const BaseVecT& v)
    {
        if( m_ptr && (m_w > 2) && (&v != this))
        {
            m_ptr[0] = v.x;
            m_ptr[1] = v.y;
            m_ptr[2] = v.z;
        }
        return *this;
    }

    AttributeHandle(T* pos = 0, unsigned w = 0) : m_ptr(pos), m_w(0) {}
private:
    AttributeHandle();
    T*              m_ptr;
    unsigned        m_w;
};


class PointBuffer2
{
    using FloatChannel = AttributeChannel<float>;
    using UCharChannel = AttributeChannel<unsigned char>;

    using FloatHandle = AttributeHandle<float>;
    using UCharHandle = AttributeHandle<unsigned char>;

    using FloatChannelPtr = std::shared_ptr<FloatChannel>;
    using ucharChannelPtr = std::shared_ptr<UCharChannel>;

    PointBuffer2(floatArr points, size_t n);

    void addFloatChannel(std::string name, size_t n, unsigned width);
    void addUCharChannel(std::string name, size_t n, unsigned width);

    bool hasUCharChannel(std::string name);
    bool hasFloatChannel(std::string name);

    unsigned ucharChannelWidth(std::string name);
    unsigned floatChannelWidth(std::string name);

    FloatHandle getFloatHandle(int idx, unsigned w);
    UCharHandle getUCharHandle(int idx, unsigned w);

    FloatHandle point(int idx);
    FloatHandle normal(int idx);

private:

    std::map<std::string, FloatChannelPtr>  m_floatChannels;
    std::map<std::string, ucharChannelPtr>  m_ucharChannels;

};

} // namespace lvr2

#endif // POINTBUFFER2_HPP
