/**
 * Copyright (c) 2019, University Osnabrück
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

#ifndef LAS_VEGAS_IOMESHINTERFACE_H
#define LAS_VEGAS_IOMESHINTERFACE_H

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <lvr2/geometry/Handles.hpp>
#include <lvr2/attrmaps/AttrMaps.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/io/AttributeManager.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>

namespace lvr2{

using BaseVec = BaseVector<float>;

template<typename ValueType> struct channel_type{};
template<> struct channel_type<float>            { static const unsigned int w = 1; typedef float type; };
template<> struct channel_type<unsigned int>     { static const unsigned int w = 1; typedef unsigned int type; };
template<> struct channel_type<unsigned char>    { static const unsigned int w = 1; typedef unsigned char type; };
template<> struct channel_type<Normal<BaseVec>>  { static const unsigned int w = 3; typedef float type; };
template<> struct channel_type<Vector<BaseVec>>  { static const unsigned int w = 3; typedef float type; };

template<typename HandleType> struct attribute_type{};;
template<> struct attribute_type<EdgeHandle>            { static const std::string attr_group; };
template<> struct attribute_type<OptionalEdgeHandle>    { static const std::string attr_group; };
template<> struct attribute_type<VertexHandle>          { static const std::string attr_group; };
template<> struct attribute_type<OptionalVertexHandle>  { static const std::string attr_group; };
template<> struct attribute_type<FaceHandle>            { static const std::string attr_group; };
template<> struct attribute_type<OptionalFaceHandle>    { static const std::string attr_group; };
template<> struct attribute_type<ClusterHandle>         { static const std::string attr_group; };
template<> struct attribute_type<OptionalClusterHandle> { static const std::string attr_group; };

class IOMeshInterface
{
 public:

  static const std::string vertex_attributes;
  static const std::string face_attributes;
  static const std::string edge_attributes;
  static const std::string cluster_attributes;

  HalfEdgeMesh<BaseVec> getMesh()
  {
    HalfEdgeMesh<BaseVec> hem;
    auto vertices = getVertices();
    auto indices = getIndices();

    for (size_t i = 0; i < vertices.size();)
    {
      hem.addVertex(BaseVec(
          vertices[i++],
          vertices[i++],
          vertices[i++]));
    }

    for (size_t i = 0; i < indices.size(); i++)
    {
      VertexHandle v1(indices[i++]);
      VertexHandle v2(indices[i++]);
      VertexHandle v3(indices[i++]);
      hem.addFace(v1, v2, v3);
    }
    return hem;
  }

  ///
  /// \brief addDenseValueAttributeMap    Stores a dense attribute map of primitive values
  /// \tparam DataType                    The data type to store
  /// \param map                          The attribute map, which can be a specific attribute map implementation
  /// \param keys_name                    The name of the map keys for the attribute manager
  /// \param values_name                  The name of the map values for the attribute manager
  template<typename MapT>
  bool addDenseAttributeMap(const MapT& map, const std::string& group, const std::string& name);

  ///
  /// \brief getValueAttributeMap  Reads a dense attribute map of premitive values
  /// \tparam BaseVecT        The base vector type, with x, y, z attributes
  /// \param map              The attribute map, which can be a specific attribute map implementation
  /// \param name             The name of the map values for the attribute manager
  /// \return                 true if the conversion and the reading succeeded
  template <typename MapT>
  boost::optional<MapT> getDenseAttributeMap(const std::string& group, const std::string& name);

  ///
  /// \brief addAttributeMap  Stores an attribute map
  /// \tparam DataType        The data type to store
  /// \param map              The attribute map, which can be a specific attribute map implementation
  /// \param keys_name        The name of the map keys for the attribute manager
  /// \param values_name      The name of the map values for the attribute manager
  template<typename MapT>
  bool addAttributeMap(const MapT& map, const std::string& group, const std::string& name);

  ///
  /// \brief getAttributeMap  Reads and attribute map
  /// \tparam DataType        The data type to store
  /// \param map              The attribute map, which can be a specific attribute map implementation
  /// \param name             The name of the map values for the attribute manager
  /// \return                 true if the conversion and the reading succeeded
  template<typename MapT>
  boost::optional<MapT> getAttributeMap(const std::string& group, const std::string& name);

 private:

  virtual std::vector<float> getVertices() = 0;
  virtual std::vector<unsigned int> getIndices() = 0;

  virtual bool getChannel(const std::string group, const std::string name, FloatChannel::Ptr& channel) = 0;
  virtual bool getChannel(const std::string group, const std::string name, IndexChannel::Ptr& channel) = 0;
  virtual bool getChannel(const std::string group, const std::string name, UCharChannel::Ptr& channel) = 0;

  virtual bool addChannel(const std::string group, const std::string name, const FloatChannel::Ptr& channel) = 0;
  virtual bool addChannel(const std::string group, const std::string name, const IndexChannel::Ptr& channel) = 0;
  virtual bool addChannel(const std::string group, const std::string name, const UCharChannel::Ptr& channel) = 0;


};

}

#include "IOMeshInterface.tcc"

#endif //LAS_VEGAS_IOMESHINTERFACE_H
