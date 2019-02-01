/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef BASEBUFFER_HPP
#define BASEBUFFER_HPP

#include <lvr2/io/AttributeManager.hpp>
#include <lvr2/geometry/Handles.hpp>
#include <lvr2/attrmaps/AttrMaps.hpp>
#include <lvr2/geometry/BaseVector.hpp>

using Index = uint32_t;

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

    ///
    /// \brief addAttributeMap  Stores an attribute map of vectors
    /// \tparam BaseVecT        The base vector type, with x, y, z attributes
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param keys_name        The name of the map keys for the attribute manager
    /// \param values_name      The name of the map values for the attribute manager
    bool addAttributeMap(const AttributeMap<BaseHandle<Index>, BaseVector<float> >& map,
        const std::string& keys_name, const std::string& values_name);
    ///
    /// \brief addAttributeMap  Stores an attribute map of unsigned chars
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param keys_name        The name of the map keys for the attribute manager
    /// \param values_name      The name of the map values for the attribute manager
    bool addAttributeMap(
        const AttributeMap<BaseHandle<Index>, unsigned char>& map,
        const std::string& keys_name, const std::string& values_name)
        {return addAttributeMap<unsigned char>(map, keys_name, values_name);}


    ///
    /// \brief addAttributeMap  Stores an attribute map of unsigned ints
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param keys_name        The name of the map keys for the attribute manager
    /// \param values_name      The name of the map values for the attribute manager
    bool addAttributeMap(
        const AttributeMap<BaseHandle<Index>, unsigned int>& map,
        const std::string& keys_name, const std::string& values_name)
        {return addAttributeMap<unsigned int>(map, keys_name, values_name);}


    ///
    /// \brief addAttributeMap  Stores an attribute map of floats
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param keys_name        The name of the map keys for the attribute manager
    /// \param values_name      The name of the map values for the attribute manager
    bool addAttributeMap(
        const AttributeMap<BaseHandle<Index>, float>& map,
        const std::string& keys_name, const std::string& values_name)
        {return addAttributeMap<float>(map, keys_name, values_name);}


    ///
    /// \brief getAttributeMap  Reads an attribute map of vectors
    /// \tparam BaseVecT        The base vector type, with x, y, z attributes
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param keys_name        The name of the map keys for the attribute manager
    /// \param values_name      The name of the map values for the attribute manager
    /// \return                 true if the conversion and the reading succeeded
    bool getAttributeMap(AttributeMap<BaseHandle<Index>, BaseVector<float> >& map,
        const std::string& keys_name, const std::string& values_name);

    ///
    /// \brief getAttributeMap  Reads an attribute map of unsigned ints
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param keys_name        The name of the map keys for the attribute manager
    /// \param values_name      The name of the map values for the attribute manager
    /// \return                 true if the conversion and the reading succeeded
    bool getAttributeMap(
        AttributeMap<BaseHandle<Index>, unsigned int>& map,
        const std::string& keys_name, const std::string& values_name)
        {return getAttributeMap<unsigned int>(map, keys_name, values_name);}


    ///
    /// \brief getAttributeMap  Reads an attribute map of unsigned chars
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param keys_name        The name of the map keys for the attribute manager
    /// \param values_name      The name of the map values for the attribute manager
    /// \return                 true if the conversion and the reading succeeded
    bool getAttributeMap(
        AttributeMap<BaseHandle<Index>, unsigned char>& map,
        const std::string& keys_name, const std::string& values_name)
        {return getAttributeMap<unsigned char>(map, keys_name, values_name);}

    ///
    /// \brief getAttributeMap  Reads an attribute map of floats
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param keys_name        The name of the map keys for the attribute manager
    /// \param values_name      The name of the map values for the attribute manager
    /// \return                 true if the conversion and the reading succeeded
    bool getAttributeMap(
        AttributeMap<BaseHandle<Index>, float>& map,
        const std::string& keys_name, const std::string& values_name)
        {return getAttributeMap<float>(map, keys_name, values_name);}

    ///
    /// \brief addAttributeMap  Stores a dense attribute map of vectors
    /// \tparam BaseVecT        The base vector type, with x, y, z attributes
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param name             The name of the map values for the attribute manager
    bool addAttributeMap(const DenseAttrMap<BaseHandle<Index>, BaseVector<float> >& map, const std::string& name);

    ///
    /// \brief addAttributeMap  Stores a dense attribute map of unsigned chars
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param name             The name of the map values for the attribute manager
    bool addAttributeMap(const DenseAttrMap<BaseHandle<Index>, unsigned char>& map, const std::string& name)
    {return addAttributeMap<unsigned char>(map, name);}


    ///
    /// \brief addAttributeMap  Stores a dense attribute map of unsigned ints
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param name             The name of the map values for the attribute manager
    bool addAttributeMap(const DenseAttrMap<BaseHandle<Index>, unsigned int>& map, const std::string& name)
    {return addAttributeMap<unsigned int>(map, name);}

    ///
    /// \brief addAttributeMap  Stores a dense attribute map of floats
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param name             The name of the map values for the attribute manager
    bool addAttributeMap(const DenseAttrMap<BaseHandle<Index>, float>& map, const std::string& name)
    {return addAttributeMap<float>(map, name);}

    ///
    /// \brief getAttributeMap  Reads a dense attribute map of vectors
    /// \tparam BaseVecT        The base vector type, with x, y, z attributes
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param name             The name of the map values for the attribute manager
    /// \return                 true if the conversion and the reading succeeded
    bool getAttributeMap(DenseAttrMap<BaseHandle<Index>, BaseVector<float> >& map, const std::string& name);

    ///
    /// \brief getAttributeMap  Reads a dense attribute map of unsigned ints
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param name             The name of the map values for the attribute manager
    /// \return                 true if the conversion and the reading succeeded
    bool getAttributeMap(DenseAttrMap <BaseHandle<Index>, unsigned int>& map, const std::string& name)
    {return getAttributeMap<unsigned int>(map, name);}

    ///
    /// \brief getAttributeMap  Reads a dense attribute map of unsigned chars
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param name             The name of the map values for the attribute manager
    /// \return                 true if the conversion and the reading succeeded
    bool getAttributeMap(DenseAttrMap<BaseHandle<Index>, unsigned char>& map, const std::string& name)
    {return getAttributeMap<unsigned char>(map, name);}

    ///
    /// \brief getAttributeMap  Reads a dense attribute map of floats
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param name             The name of the map values for the attribute manager
    /// \return                 true if the conversion and the reading succeeded
    bool getAttributeMap(DenseAttrMap<BaseHandle<Index>, float>& map, const std::string& name)
    {return getAttributeMap<float>(map, name);}

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

private:

    ///
    /// \brief addAttributeMap  Stores an attribute map
    /// \tparam DataType        The data type to store
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param keys_name        The name of the map keys for the attribute manager
    /// \param values_name      The name of the map values for the attribute manager
    template<typename DataType>
    bool addAttributeMap(const AttributeMap<BaseHandle<Index>, DataType>& map,
                         const std::string& keys_name, const std::string& values_name)
    {
      boost::shared_array<DataType> values(new DataType[map.numValues()]);
      indexArray keys(new unsigned int[map.numValues()]);
      Index i = 0;
      for(auto handle: map)
      {
        values[i] = map[handle];
        keys[i++] = handle.idx();
      }

      m_channels.addChannel(values, values_name, map.numValues(), 1);
      m_channels.addIndexChannel(keys, keys_name, map.numValues(), 1);
      return true;
    }

    ///
    /// \brief addAttributeMap  Stores a dense attribute map
    /// \tparam DataType        The data type to store
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param name             The name of the map values for the attribute manager
    template<typename DataType>
    bool addAttributeMap(const DenseAttrMap <BaseHandle<Index>, DataType>& map, const std::string& name)
    {
        boost::shared_array<DataType> values(new DataType[map.numValues()]);
        Index i = 0;
        for(auto handle: map) values[i++] = map[handle];
        m_channels.addChannel(values, name, map.numValues(), 1);
        return true;
    }

    ///
    /// \brief getAttributeMap  Reads and attribute map
    /// \tparam DataType        The data type to store
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param name             The name of the map values for the attribute manager
    /// \return                 true if the conversion and the reading succeeded
    template<typename DataType>
    bool getAttributeMap(AttributeMap<BaseHandle<Index>, DataType>& map,
                         const std::string& keys_name, const std::string& values_name)
    {
        IndexChannelOptional keys_opt = m_channels.getIndexChannel(keys_name);
        boost::optional<AttributeChannel<DataType>&> values_opt;
        m_channels.getChannel(values_name, values_opt);

        if(keys_opt && values_opt &&
            keys_opt.get().width() == 1 &&
            values_opt.get().width() == 1 &&
            keys_opt.get().numAttributes() == values_opt.get().numAttributes())
        {
            AttributeChannel<DataType>& values = values_opt.get();
            IndexChannel& keys = keys_opt.get();
            map.clear();
            for(size_t i=0; i<keys.numAttributes(); i++)
            {
                map.insert(BaseHandle<Index>(keys[i][0]), values[i][0]);
            }
            return true;
        }
        else return false;
    }

    ///
    /// \brief getAttributeMap  Reads a dense attribute map
    /// \tparam DataType        The data type to store
    /// \param map              The attribute map, which can be a specific attribute map implementation
    /// \param name             The name of the map values for the attribute manager
    /// \return                 true if the conversion and the reading succeeded
    template<typename DataType>
    bool getAttributeMap(DenseAttrMap<BaseHandle<Index>, DataType>& map, const std::string& name)
    {
        boost::optional<AttributeChannel<DataType>&> values_opt;
        m_channels.getChannel(name, values_opt);

        if(values_opt && values_opt.get().width() == 1)
        {
            map.clear();
            AttributeChannel<DataType>& values = values_opt.get();
            for(size_t i=0; i<values.numAttributes(); i++) map.insert(BaseHandle<Index>(i), values[i][0]);
            //TODO map = DenseAttrMap<BaseHandle<Index>, DataType>(values_opt.get().dataPtr());
            return true;
        }
        else return false;
    }


};

}

#endif // BASEBUFFER_HPP
