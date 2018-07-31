#ifndef BASEBUFFER_HPP
#define BASEBUFFER_HPP

#include <lvr2/io/AttributeManager.hpp>

namespace lvr2
{

///
/// \brief Base class to handle buffers with several attribute channels
///
class BaseBuffer
{
public:

    /// Constructs an empty buffer
    BaseBuffer();

    ///
    /// \brief addFloatChannel  Adds an channel with floating point data to the buffer
    /// \param data             The float array with attributes
    /// \param name             Name of the channel
    /// \param n                Number of attributes in the channel
    /// \param w                Number of elements
    ///
    virtual void addFloatChannel(floatArr data, std::string name, size_t n, unsigned w);
    virtual void addUCharChannel(ucharArr data, std::string name, size_t n, unsigned w);

    floatArr getFloatArray(const std::string& name, unsigned& w);
    ucharArr getUcharArray(const std::string& name, unsigned& w);

    FloatChannel getFloatChannel(const std::string& name);
    UCharChannel getUcharChannel(const std::string& name);

    virtual ~BaseBuffer() {}

protected:
    AttributeManager        m_channels;
};

}

#endif // BASEBUFFER_HPP
