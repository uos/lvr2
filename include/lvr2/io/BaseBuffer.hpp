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
    /// \brief addUCharChannel  Adds an channel with byte aligned (unsigned char) data to the buffer
    /// \param data             The unsigned char array with attributes
    /// \param name             Name of the channel
    /// \param n                Number of attributes in the channel
    /// \param w                Number of elements per attribute
    ///
    virtual void addUCharChannel(ucharArr data, std::string name, size_t n, unsigned w);

    ///
    /// \brief addIndexChannel  Adds an channel with unsigned int data to the buffer
    /// \param data             The unsigned int array with attributes
    /// \param name             Name of the channel
    /// \param n                Number of attributes in the channel
    /// \param w                Number of elements per attribute
    ///
    virtual void addIndexChannel(indexArray data, std::string name, size_t n, unsigned w);

    void addFloatAttribute(float data, std::string name);
    void addUCharAttribute(unsigned char data, std::string name);
    void addIntAttribute(int data, std::string name);

    ///
    /// \brief getFloatArray    Returns a float array representation of the given channel
    /// \param name             Name of the channel
    /// \param n                Number of attributes in the channel
    /// \param w                Number of elements per attribute
    /// \return                 Array of attributes
    ///
    floatArr getFloatArray(const std::string& name, size_t& n, unsigned& w);

    ///
    /// \brief getUCharArray    Returns an unsigned char array representation of the given channel
    /// \param name             Name of the channel
    /// \param n                Number of attributes in the channel
    /// \param w                Number of elements per attribute
    /// \return                 Array of attributes
    ///
    ucharArr getUCharArray(const std::string& name, size_t& n, unsigned& w);

    ///
    /// \brief getIndexArray    Returns a unsigned int array representation of the given channel
    /// \param name             Name of the channel
    /// \param n                Number of attributes in the channel
    /// \param w                Number of elements per attribute
    /// \return                 Array of attributes
    ///
    indexArray getIndexArray(const std::string& name, size_t& n, unsigned& w);

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
    ///                         given attribute layer. The UChar channel representation
    ///                         provides access operators with conversion to
    ///                         LVRs base vector types to easily allow mathematical
    ///                         computations on the channel. See \ref AttributeManager class
    ///                         for reference.
    /// \param name             Name of the channel
    /// \return                 A UCharChannel representation of the attributes
    ///
    UCharChannelOptional getUCharChannel(const std::string& name);

    ///
    /// \brief getIndexChannel  Returns a IndexChannel representation of the
    ///                         given attribute layer. The index channel representation
    ///                         provides access operators with conversion to
    ///                         LVRs base vector types to easily allow mathematical
    ///                         computations on the channel. See \ref AttributeManager class
    ///                         for reference.
    /// \param name             Name of the channel
    /// \return                 A IndexChannel representation of the attributes
    ///
    IndexChannelOptional getIndexChannel(const std::string& name);

    floatOptional getFloatAttribute(std::string name);
    ucharOptional getUCharAttribute(std::string name);
    intOptional getIntAttribute(std::string name);

    /// Destructor
    virtual ~BaseBuffer() {}

protected:

    /// Manager class to handle different attribute layers
    AttributeManager        m_channels;
};

}

#endif // BASEBUFFER_HPP
