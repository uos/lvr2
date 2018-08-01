#include <lvr2/io/BaseBuffer.hpp>

namespace lvr2
{

BaseBuffer::BaseBuffer()
{

}

floatArr BaseBuffer::getFloatArray(const std::string& name, size_t& n, unsigned& w)
{
    return m_channels.getFloatArray(n, w, name);
}

ucharArr BaseBuffer::getUcharArray(const std::string& name, size_t& n, unsigned& w)
{
    return m_channels.getUCharArray(n, w, name);
}

FloatChannel BaseBuffer::getFloatChannel(const std::string& name)
{
    return m_channels.getFloatChannel(name);
}

UCharChannel BaseBuffer::getUcharChannel(const std::string& name)
{
    return m_channels.getUCharChannel(name);
}

void BaseBuffer::addFloatChannel(floatArr data, std::string name, size_t n, unsigned w)
{
    m_channels.addFloatChannel(data, name, n, w);
}


void BaseBuffer::addUCharChannel(ucharArr data, std::string name, size_t n, unsigned w)
{
    m_channels.addUCharChannel(data, name, n, w);
}

}
