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

#ifndef LAS_VEGAS_MESHIOINTERFACE_TCC
#define LAS_VEGAS_MESHIOINTERFACE_TCC

namespace lvr2{

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

const std::string attribute_type<EdgeHandle>::attr_group            = "edge_attributes";
const std::string attribute_type<OptionalEdgeHandle>::attr_group    = "edge_attributes";
const std::string attribute_type<VertexHandle>::attr_group          = "vertex_attributes";
const std::string attribute_type<OptionalVertexHandle>::attr_group  = "vertex_attributes";
const std::string attribute_type<FaceHandle>::attr_group            = "face_attributes";
const std::string attribute_type<OptionalFaceHandle>::attr_group    = "face_attributes";
const std::string attribute_type<ClusterHandle>::attr_group         = "cluster_attributes";
const std::string attribute_type<OptionalClusterHandle>::attr_group = "cluster_attributes";

template<typename MapT>
bool MeshIOInterface::addDenseAttributeMap(const MapT& map, const std::string& name)
{
  AttributeChannel<typename channel_type<typename MapT::ValueType>::type> values(
          map.numValues(), channel_type<typename MapT::ValueType>::w);

  Index i = 0;
  for(auto handle: map)
  {
    values[i++] = map[handle]; //TODO handle deleted map values.
  }
  return addChannel(attribute_type<typename MapT::HandleType>::attr_group, name, values);
}

template <typename MapT>
boost::optional<MapT> MeshIOInterface::getDenseAttributeMap(const std::string& name)
{
  typename AttributeChannel<typename channel_type<typename MapT::ValueType>::type>::Optional channel_opt;
  if(getChannel(attribute_type<typename MapT::HandleType>::attr_group, name, channel_opt) && channel_opt
      && channel_opt.get().width() == channel_type<typename MapT::ValueType>::w)
  {
    auto& channel = channel_opt.get();
    MapT map;
    map.reserve(channel.numAttributes());
    for(size_t i=0; i<channel.numAttributes(); i++)
    {
      typename MapT::HandleType handle(i);
      map.insert(handle, channel[i]);
    }
    return map;
  }
  return boost::none;
}

template<typename MapT>
bool MeshIOInterface::addAttributeMap(const MapT& map, const std::string& name)
{
  AttributeChannel<typename channel_type<typename MapT::ValueType>::type> values(
      map.numValues(), channel_type<typename MapT::ValueType>::w);
  IndexChannel indices(map.numValues(), 1);

  Index i = 0;
  for(auto handle: map)
  {
    values[i] = map[handle];
    indices[i++] = handle.idx();
  }

  return addChannel(attribute_type<typename MapT::HandleType>::attr_group, name, values)
    && addChannel(attribute_type<typename MapT::HandleType>::attr_group, name + "_idx", indices);
}

template<typename MapT>
boost::optional<MapT> MeshIOInterface::getAttributeMap(const std::string& name)
{
  typename AttributeChannel<typename channel_type<typename MapT::ValueType>::type>::Optional values_opt;
  typename IndexChannel::Optional indices_opt;
  if(getChannel(attribute_type<typename MapT::HandleType>::attr_group, name+"_idx", indices_opt)
    && getChannel(attribute_type<typename MapT::HandleType>::attr_group, name, values_opt)
      && indices_opt && values_opt && indices_opt.get().width() == 1
      && values_opt.get().width() == channel_type<typename MapT::ValueType>::w
      && indices_opt.get().numAttributes() == values_opt.get().numAttributes())
  {
    auto& indices =  indices_opt.get();
    auto& values = values_opt.get();
    MapT map;

    map.reserve(indices.numAttributes());
    for(size_t i=0; i<indices.numAttributes(); i++)
    {
      map.insert(typename MapT::HandleType(indices[i]), values[i]);
    }
    return map;
  }
  return boost::none;
}

} /* namespace lvr2 */

#endif //LAS_VEGAS_MESHIOINTERFACE_TCC
