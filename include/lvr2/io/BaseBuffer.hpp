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
    /// \param w                Number of elements per attribute
    ///
    virtual void addFloatChannel(floatArr data, std::string name, size_t n, unsigned w);


    ///
    /// \brief addFloatChannel  Adds an channel with byte aligned (unsigned char) data to the buffer
    /// \param data             The float array with attributes
    /// \param name             Name of the channel
    /// \param n                Number of attributes in the channel
    /// \param w                Number of elements per attribute
    ///
    virtual void addUCharChannel(ucharArr data, std::string name, size_t n, unsigned w);

    ///
    /// \brief getFloatArray    Returns a float array representation of the given channel
    /// \param name             Name of the channel
    /// \param n                Number of attributes in the channel
    /// \param w                Number of elements per attribute
    /// \return                 Array of attributes
    ///
    floatArr getFloatArray(const std::string& name, size_t& n, unsigned& w);

    ///
    /// \brief getFloatArray    Returns a unsigned char arrey representation of the given channel
    /// \param name             Name of the channel
    /// \param n                Number of attributes in the channel
    /// \param w                Number of elements per attribute
    /// \return                 Array of attributes
    ///
    ucharArr getUcharArray(const std::string& name, size_t& n, unsigned& w);

    ///
    /// \brief getFloatChannel  Returns a float channel representation of the
    ///                         given attribute layer. The float channel representation
    ///                         provides access operators with conversion to
    ///                         LVRs base vector types to easily allow mathematical
    ///                         computations on the channel. See \ref AttributeManager class
    ///                         for reference.
    /// \param name             Name of the channel
    /// \return                 A FloatChannel representation of the attributes
    ///
    FloatChannelOptional getFloatChannel(const std::string& name);

    ///
    /// \brief getUCharChannel  Returns a UCHarChannel representation of the
    ///                         given attribute layer. The float channel representation
    ///                         provides access operators with conversion to
    ///                         LVRs base vector types to easily allow mathematical
    ///                         computations on the channel. See \ref AttributeManager class
    ///                         for reference.
    /// \param name             Name of the channel
    /// \return                 A FloatChannel representation of the attributes
    ///
    UCharChannelOptional getUcharChannel(const std::string& name);

    /// Destructor
    virtual ~BaseBuffer() {}

protected:

    /// Manager class to handle different attribute layers
    AttributeManager        m_channels;
};

}

#endif // BASEBUFFER_HPP
