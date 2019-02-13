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

class IOMeshInterface
{
 public:

  using BaseVec = BaseVector<float>;

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
  /// \brief getAttributeMap  Reads a dense attribute map of floats
  /// \param map              The attribute map, which can be a specific attribute map implementation
  /// \param name             The name of the map values for the attribute manager
  /// \return                 true if the conversion and the reading succeeded
  virtual bool getAttributeMap(DenseFaceMap<Vector<BaseVector<float>>>& map, const std::string& name){
    return getDenseVectorAttributeMap<DenseFaceMap<Vector<BaseVector<float> > > >(map, face_attributes, name);
  }

  virtual bool getAttributeMap(DenseFaceMap<Normal<BaseVector<float>>>& map, const std::string& name){
    return getDenseVectorAttributeMap<DenseFaceMap<Normal<BaseVector<float> > > >(map, face_attributes, name);
  }

  virtual bool getAttributeMap(DenseFaceMap<float>& map, const std::string& name){
    return getDenseValueAttributeMap<DenseFaceMap<float> >(map, face_attributes, name);
  }

  virtual bool getAttributeMap(DenseFaceMap<unsigned int>& map, const std::string& name){
    return getDenseValueAttributeMap<DenseFaceMap<unsigned int>>(map, face_attributes, name);
  }

  virtual bool getAttributeMap(DenseFaceMap<unsigned char>& map, const std::string& name){
    return getDenseValueAttributeMap<DenseFaceMap<unsigned char>>(map, face_attributes, name);
  }

  virtual bool addAttributeMap(const DenseFaceMap<float>& map, const std::string& name){
    return addDenseValueAttributeMap<DenseFaceMap<float>>(map, face_attributes, name);
  }

  virtual bool addAttributeMap(const DenseFaceMap<unsigned int>& map, const std::string& name){
    return addDenseValueAttributeMap<DenseFaceMap<unsigned int>>(map, face_attributes, name);
  }

  virtual bool addAttributeMap(const DenseFaceMap<unsigned char>& map, const std::string& name){
    return addDenseValueAttributeMap<DenseFaceMap<unsigned char>>(map, face_attributes, name);
  }

  virtual bool addAttributeMap(const DenseFaceMap<Vector<BaseVector<float>>>& map, const std::string& name){
    return addDenseVectorAttributeMap<DenseFaceMap<Vector<BaseVector<float>>>>(map, face_attributes, name);
  }

  virtual bool addAttributeMap(const DenseFaceMap<Normal<BaseVector<float>>>& map, const std::string& name){
    return addDenseVectorAttributeMap<DenseFaceMap<Normal<BaseVector<float>>>>(map, face_attributes, name);
  }

  ///
  /// VertexMaps
  ///

  virtual bool getAttributeMap(DenseVertexMap<Vector<BaseVector<float>>>& map, const std::string& name){
    return getDenseVectorAttributeMap<DenseVertexMap<Vector<BaseVector<float> > > >(map, vertex_attributes, name);
  }

  virtual bool getAttributeMap(DenseVertexMap<Normal<BaseVector<float>>>& map, const std::string& name){
    return getDenseVectorAttributeMap<DenseVertexMap<Normal<BaseVector<float> > > >(map, vertex_attributes, name);
  }

  virtual bool getAttributeMap(DenseVertexMap<float>& map, const std::string& name){
    return getDenseValueAttributeMap<DenseVertexMap<float>>(map, vertex_attributes, name);
  }

  virtual bool getAttributeMap(DenseVertexMap<unsigned int>& map, const std::string& name){
    return getDenseValueAttributeMap<DenseVertexMap<unsigned int>>(map, vertex_attributes, name);
  }

  virtual bool getAttributeMap(DenseVertexMap<unsigned char>& map, const std::string& name){
    return getDenseValueAttributeMap<DenseVertexMap<unsigned char>>(map, vertex_attributes, name);
  }

  /// \brief      adds a DenseVertexMap with the associated name to the io buffer channels. This method will store
  ///             the dense vertex map on the vertex property level, since a VertexMap is always associated with the
  ///             vertices in the mesh.
  /// \param map  The map which needs to be stored persistently.
  /// \param name The corresponding map name
  /// \return     True if the map as been added successfully, false otherwise.
  virtual bool addAttributeMap(const DenseVertexMap<float>& map, const std::string& name){
    return addDenseValueAttributeMap<DenseVertexMap<float>>(map, vertex_attributes, name);
  }

  virtual bool addAttributeMap(const DenseVertexMap<unsigned int>& map, const std::string& name){
    return addDenseValueAttributeMap<DenseVertexMap<unsigned int>>(map, vertex_attributes, name);
  }

  virtual bool addAttributeMap(const DenseVertexMap<unsigned char>& map, const std::string& name){
    return addDenseValueAttributeMap<DenseVertexMap<unsigned char>>(map, vertex_attributes, name);
  }

  virtual bool addAttributeMap(const DenseVertexMap<Vector<BaseVector<float>>>& map, const std::string& name){
    return addDenseVectorAttributeMap<DenseVertexMap<Vector<BaseVector<float>>>>(map, vertex_attributes, name);
  }

  virtual bool addAttributeMap(const DenseVertexMap<Normal<BaseVector<float>>>& map, const std::string& name){
    return addDenseVectorAttributeMap<DenseVertexMap<Normal<BaseVector<float>>>>(map, vertex_attributes, name);
  }

  /// Edge Maps
  virtual bool getAttributeMap(DenseEdgeMap<Vector<BaseVector<float>>>& map, const std::string& name){
    return getDenseVectorAttributeMap<DenseEdgeMap<Vector<BaseVector<float> > > >(map, edge_attributes, name);
  }

  virtual bool getAttributeMap(DenseEdgeMap<Normal<BaseVector<float>>>& map, const std::string& name){
    return getDenseVectorAttributeMap<DenseEdgeMap<Normal<BaseVector<float> > > >(map, edge_attributes, name);
  }

  virtual bool getAttributeMap(DenseEdgeMap<float>& map, const std::string& name){
    return getDenseValueAttributeMap<DenseEdgeMap<float>>(map, edge_attributes, name);
  }

  virtual bool getAttributeMap(DenseEdgeMap<unsigned int>& map, const std::string& name){
    return getDenseValueAttributeMap<DenseEdgeMap<unsigned int>>(map, edge_attributes, name);
  }

  virtual bool getAttributeMap(DenseEdgeMap<unsigned char>& map, const std::string& name){
    return getDenseValueAttributeMap<DenseEdgeMap<unsigned char>>(map, edge_attributes, name);
  }

  virtual bool addAttributeMap(const DenseEdgeMap<float>& map, const std::string& name){
    return addDenseValueAttributeMap<DenseEdgeMap<float>>(map, edge_attributes, name);
  }

  virtual bool addAttributeMap(const DenseEdgeMap<unsigned int>& map, const std::string& name){
    return addDenseValueAttributeMap<DenseEdgeMap<unsigned int>>(map, edge_attributes, name);
  }

  virtual bool addAttributeMap(const DenseEdgeMap<unsigned char>& map, const std::string& name){
    return addDenseValueAttributeMap<DenseEdgeMap<unsigned char>>(map, edge_attributes, name);
  }

  virtual bool addAttributeMap(const DenseEdgeMap<Vector<BaseVector<float>>>& map, const std::string& name){
    return addDenseVectorAttributeMap<DenseEdgeMap<Vector<BaseVector<float>>>>(map, edge_attributes, name);
  }

  virtual bool addAttributeMap(const DenseEdgeMap<Normal<BaseVector<float>>>& map, const std::string& name){
    return addDenseVectorAttributeMap<DenseEdgeMap<Normal<BaseVector<float>>>>(map, edge_attributes, name);
  }

  /// Cluster Maps
  virtual bool getAttributeMap(DenseClusterMap<Vector<BaseVector<float>>>& map, const std::string& name){
    return getDenseVectorAttributeMap<DenseClusterMap<Vector<BaseVector<float> > > >(map, cluster_attributes, name);
  }

  virtual bool getAttributeMap(DenseClusterMap<Normal<BaseVector<float>>>& map, const std::string& name){
    return getDenseVectorAttributeMap<DenseClusterMap<Normal<BaseVector<float> > > >(map, cluster_attributes, name);
  }

  virtual bool getAttributeMap(DenseClusterMap<float>& map, const std::string& name){
    return getDenseValueAttributeMap<DenseClusterMap<float>>(map, cluster_attributes, name);
  }

  virtual bool getAttributeMap(DenseClusterMap<unsigned int>& map, const std::string& name){
    return getDenseValueAttributeMap<DenseClusterMap<unsigned int>>(map, cluster_attributes, name);
  }

  virtual bool getAttributeMap(DenseClusterMap<unsigned char>& map, const std::string& name){
    return getDenseValueAttributeMap<DenseClusterMap<unsigned char>>(map, cluster_attributes, name);
  }

  virtual bool addAttributeMap(const DenseClusterMap<float>& map, const std::string& name){
    return addDenseValueAttributeMap<DenseClusterMap<float>>(map, cluster_attributes, name);
  }

  virtual bool addAttributeMap(const DenseClusterMap<unsigned int>& map, const std::string& name){
    return addDenseValueAttributeMap<DenseClusterMap<unsigned int>>(map, cluster_attributes, name);
  }

  virtual bool addAttributeMap(const DenseClusterMap<unsigned char>& map, const std::string& name){
    return addDenseValueAttributeMap<DenseClusterMap<unsigned char>>(map, cluster_attributes, name);
  }

  virtual bool addAttributeMap(const DenseClusterMap<Vector<BaseVector<float>>>& map, const std::string& name){
    return addDenseVectorAttributeMap<DenseClusterMap<Vector<BaseVector<float>>>>(map, cluster_attributes, name);
  }

  virtual bool addAttributeMap(const DenseClusterMap<Normal<BaseVector<float>>>& map, const std::string& name){
    return addDenseVectorAttributeMap<DenseClusterMap<Normal<BaseVector<float>>>>(map, cluster_attributes, name);
  }

 private:

  virtual std::vector<float> getVertices() = 0;
  virtual std::vector<unsigned int> getIndices() = 0;

  virtual bool getChannel(const std::string group, const std::string name, FloatChannel::Ptr& channel) = 0;
  virtual bool getChannel(const std::string group, const std::string name, IndexChannel::Ptr& channel) = 0;
  virtual bool getChannel(const std::string group, const std::string name, UCharChannel::Ptr& channel) = 0;

  virtual bool addChannel(const std::string group, const std::string name, const FloatChannel::Ptr& channel) = 0;
  virtual bool addChannel(const std::string group, const std::string name, const IndexChannel::Ptr& channel) = 0;
  virtual bool addChannel(const std::string group, const std::string name, const UCharChannel::Ptr& channel) = 0;

  ///
  /// \brief getValueAttributeMap  Reads a dense attribute map of premitive values
  /// \tparam BaseVecT        The base vector type, with x, y, z attributes
  /// \param map              The attribute map, which can be a specific attribute map implementation
  /// \param name             The name of the map values for the attribute manager
  /// \return                 true if the conversion and the reading succeeded
  template <typename MapT>
  bool getDenseValueAttributeMap(MapT& map, const std::string& group, const std::string& name)
  {
    typename AttributeChannel< typename MapT::ValueType >::Ptr channel_ptr;
    if(getChannel(group, name, channel_ptr) && channel_ptr && channel_ptr->width() == 1)
    {
      map.clear();
      for(size_t i=0; i<channel_ptr->numAttributes(); i++)
      {
        typename MapT::HandleType handle(i);
        map.insert(handle, (*channel_ptr)[i]);
      }
      return true;
    }
    return false;
  }

  ///
  /// \brief getAttributeMap  Reads a dense attribute map of vectors
  /// \tparam BaseVecT        The base vector type, with x, y, z attributes
  /// \param map              The attribute map, which can be a specific attribute map implementation
  /// \param name             The name of the map values for the attribute manager
  /// \return                 true if the conversion and the reading succeeded
  template <typename MapT>
  bool getDenseVectorAttributeMap(MapT& map, const std::string& group, const std::string& name)
  {
    FloatChannel::Ptr channel;
    if(getChannel(group, name, channel) && channel->width() == 3)
    {
      map.clear();
      for(size_t i=0; i<channel->numAttributes(); i++)
      {
        typename MapT::HandleType handle(i);
        map.insert(handle, (*channel)[i]);
      }
      return true;
    }
    else
    {
      return false;
    }
  }

  ///
  /// \brief addDenseValueAttributeMap    Stores a dense attribute map of primitive values
  /// \tparam DataType                    The data type to store
  /// \param map                          The attribute map, which can be a specific attribute map implementation
  /// \param keys_name                    The name of the map keys for the attribute manager
  /// \param values_name                  The name of the map values for the attribute manager
  template<typename MapT>
  bool addDenseValueAttributeMap(const MapT& map, const std::string& group, const std::string& name)
  {
    typename AttributeChannel<typename MapT::ValueType>::Ptr values_channel_ptr(
        new AttributeChannel<typename MapT::ValueType>(map.numValues(), 1));

    auto& values = *values_channel_ptr;
    Index i = 0;
    for(auto handle: map)
    {
      values[i++] = map[handle]; //TODO handle deleted map values.
    }
    addChannel(group, name, values_channel_ptr);
    return true;
  }

  ///
  /// \brief addDenseVectorAttributeMap   Stores a dense attribute map of vectors
  /// \tparam DataType                    The data type to store
  /// \param map                          The attribute map, which can be a specific attribute map implementation
  /// \param keys_name                    The name of the map keys for the attribute manager
  /// \param values_name                  The name of the map values for the attribute manager
  template<typename MapT>
  bool addDenseVectorAttributeMap(const MapT& map, const std::string& group, const std::string& name)
  {
    typename AttributeChannel<float>::Ptr values_channel_ptr(new AttributeChannel<float>(map.numValues(), 3));
    auto& channel = *values_channel_ptr;

    Index i = 0;
    for(auto handle: map)
    {
      channel[i++] = map[handle]; //TODO handle deleted map values.
    }
    addChannel(group, name, values_channel_ptr);
    return true;
  }

  ///
  /// \brief addAttributeMap  Stores an attribute map
  /// \tparam DataType        The data type to store
  /// \param map              The attribute map, which can be a specific attribute map implementation
  /// \param keys_name        The name of the map keys for the attribute manager
  /// \param values_name      The name of the map values for the attribute manager
  template<typename MapT>
  bool addAttributeMap(const MapT& map, const std::string& group, const std::string& name)
  {
    typename AttributeChannel<typename MapT::ValueType>::Ptr values_channel_ptr(
        new AttributeChannel<typename MapT::ValueType>(map.numValues(), 1));
    IndexChannel::Ptr index_channel_ptr(new IndexChannel(map.numValues(), 1));

    auto& values = *values_channel_ptr;
    auto& indices = *index_channel_ptr;

    Index i = 0;
    for(auto handle: map)
    {
      values[i] = map[handle];
      indices[i++] = handle.idx();
    }

    addChannel(group, name, values_channel_ptr);
    addChannel(group, name + "_idx", index_channel_ptr);
    return true;
  }

  ///
  /// \brief getAttributeMap  Reads and attribute map
  /// \tparam DataType        The data type to store
  /// \param map              The attribute map, which can be a specific attribute map implementation
  /// \param name             The name of the map values for the attribute manager
  /// \return                 true if the conversion and the reading succeeded
  template<typename MapT>
  bool getAttributeMap(MapT& map, const std::string& group, const std::string& name)
  {
    typename IndexChannel::Ptr index_channel_ptr;
    typename AttributeChannel<typename MapT::ValueType>::Ptr value_channel_ptr;
    if(getChannel(group, name+"_idx", index_channel_ptr) && getChannel(group, name, value_channel_ptr)
      && index_channel_ptr->width() == 1 && value_channel_ptr->width() == 1
      && index_channel_ptr->numAttributes() == value_channel_ptr->numAttributes())
    {
      map.clear();
      for(size_t i=0; i<index_channel_ptr->numAttributes(); i++)
      {
        map.insert(typename MapT::HandleType((*index_channel_ptr)[i]), (*value_channel_ptr)[i]);
      }
      return true;
    }
    return false;
  }
};

}

#endif //LAS_VEGAS_IOMESHINTERFACE_H
