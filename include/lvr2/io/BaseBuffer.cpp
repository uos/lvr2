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

ucharArr BaseBuffer::getUCharArray(const std::string& name, size_t& n, unsigned& w)
{
    return m_channels.getUCharArray(n, w, name);
}

indexArray BaseBuffer::getIndexArray(const std::string& name, size_t& n, unsigned& w)
{
    return m_channels.getIndexArray(n, w, name);
}

FloatChannelOptional BaseBuffer::getFloatChannel(const std::string& name)
{
    return m_channels.getFloatChannel(name);
}

UCharChannelOptional BaseBuffer::getUCharChannel(const std::string& name)
{
    return m_channels.getUCharChannel(name);
}

IndexChannelOptional BaseBuffer::getIndexChannel(const std::string& name)
{
    return m_channels.getIndexChannel(name);
}

floatOptional BaseBuffer::getFloatAttribute(std::string name)
{
    return m_channels.getFloatAttribute(name);
}

ucharOptional BaseBuffer::getUCharAttribute(std::string name)
{
    return m_channels.getUCharAttribute(name);
}

intOptional BaseBuffer::getIntAttribute(std::string name)
{
    return m_channels.getIntAttribute(name);
}


void BaseBuffer::addFloatChannel(floatArr data, std::string name, size_t n, unsigned w)
{
    m_channels.addFloatChannel(data, name, n, w);
}


void BaseBuffer::addUCharChannel(ucharArr data, std::string name, size_t n, unsigned w)
{
    m_channels.addUCharChannel(data, name, n, w);
}

void BaseBuffer::addIndexChannel(indexArray data, std::string name, size_t n, unsigned w)
{
    m_channels.addIndexChannel(data, name, n, w);
}

void BaseBuffer::addFloatAttribute(float data, std::string name)
{
    m_channels.addFloatAttribute(data, name);
}

void BaseBuffer::addUCharAttribute(unsigned char data, std::string name)
{
    m_channels.addUCharAttribute(data, name);
}

void BaseBuffer::addIntAttribute(int data, std::string name)
{
    m_channels.addIntAttribute(data, name);
}

}
