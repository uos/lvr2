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

#include "lvr2/util/ClusterBiMap.hpp"

namespace lvr2
{

template <typename ValueType>
struct channel_type
{
};
template <>
struct channel_type<float>
{
  static const unsigned int w = 1;
  typedef float type;
};
template <>
struct channel_type<unsigned int>
{
  static const unsigned int w = 1;
  typedef unsigned int type;
};
template <>
struct channel_type<unsigned char>
{
  static const unsigned int w = 1;
  typedef unsigned char type;
};
template <>
struct channel_type<Normal<float>>
{
  static const unsigned int w = 3;
  typedef float type;
};
template <>
struct channel_type<BaseVector<float>>
{
  static const unsigned int w = 3;
  typedef float type;
};
template <typename T, size_t size>
struct channel_type<std::array<T, size>>
{
  static const unsigned int w = size;
  typedef T type;
};


template <typename HandleType>
struct attribute_type
{
};
template <>
struct attribute_type<EdgeHandle>
{
  static const std::string attr_group;
  template <typename BaseVecT>
  using IteratorProxy = EdgeIteratorProxy<BaseVecT>;
};
template <>
struct attribute_type<OptionalEdgeHandle>
{
  static const std::string attr_group;
  template <typename BaseVecT>
  using IteratorProxy = EdgeIteratorProxy<BaseVecT>;
};
template <>
struct attribute_type<VertexHandle>
{
  static const std::string attr_group;
  template <typename BaseVecT>
  using IteratorProxy = VertexIteratorProxy<BaseVecT>;
};
template <>
struct attribute_type<OptionalVertexHandle>
{
  static const std::string attr_group;
  template <typename BaseVecT>
  using IteratorProxy = VertexIteratorProxy<BaseVecT>;
};
template <>
struct attribute_type<FaceHandle>
{
  static const std::string attr_group;
  template <typename BaseVecT>
  using IteratorProxy = FaceIteratorProxy<BaseVecT>;
};
template <>
struct attribute_type<OptionalFaceHandle>
{
  static const std::string attr_group;
  template <typename BaseVecT>
  using IteratorProxy = FaceIteratorProxy<BaseVecT>;
};
template <>
struct attribute_type<ClusterHandle>
{
  static const std::string attr_group;
  // TODO Proxy for handle type;
  template <typename HandleType>
  using IteratorProxy = ClusterBiMap<HandleType>;
};
template <>
struct attribute_type<OptionalClusterHandle>
{
  static const std::string attr_group;
  // TODO Proxy for handle type;
  template <typename HandleType>
  using IteratorProxy = ClusterBiMap<HandleType>;
};

template<typename MapT, typename BaseVecT>
bool AttributeMeshIOBase::addDenseAttributeMap(
    const BaseMesh<BaseVecT>& mesh, const MapT& map, const std::string& name)
{
  AttributeChannel<typename channel_type<typename MapT::ValueType>::type> values(
      map.numValues(), channel_type<typename MapT::ValueType>::w);

  Index i = 0;
  for (auto handle : typename attribute_type<typename MapT::HandleType>::template IteratorProxy<BaseVecT>(mesh))
  {
    values[i++] = map[handle]; //TODO handle deleted map values.
  }
  return addChannel(attribute_type<typename MapT::HandleType>::attr_group, name, values);
}

template <typename MapT>
bool AttributeMeshIOBase::addDenseAttributeMap(const MapT &map, const std::string &name)
{
  AttributeChannel<typename channel_type<typename MapT::ValueType>::type> values(
      map.numValues(), channel_type<typename MapT::ValueType>::w);

  Index i = 0;
  for (auto handle : map)
  {
    values[i++] = map[handle]; //TODO handle deleted map values.
  }
  return addChannel(attribute_type<typename MapT::HandleType>::attr_group, name, values);
}

template <typename MapT>
bool AttributeMeshIOBase::addAttributeMap(const MapT &map, const std::string &name)
{
  AttributeChannel<typename channel_type<typename MapT::ValueType>::type> values(
      map.numValues(), channel_type<typename MapT::ValueType>::w);
  IndexChannel indices(map.numValues(), 1);

  Index i = 0;
  for (auto handle : map)
  {
    values[i] = map[handle];
    indices[i++] = handle.idx();
  }

  return addChannel(attribute_type<typename MapT::HandleType>::attr_group, name, values) && addChannel(attribute_type<typename MapT::HandleType>::attr_group, name + "_idx", indices);
}


template <typename MapT>
boost::optional<MapT> AttributeMeshIOBase::getDenseAttributeMap(const std::string &name)
{
  typename AttributeChannel<typename channel_type<typename MapT::ValueType>::type>::Optional channel_opt;
  if (getChannel(attribute_type<typename MapT::HandleType>::attr_group, name, channel_opt) && channel_opt && channel_opt.get().width() == channel_type<typename MapT::ValueType>::w)
  {
    AttributeChannel<typename channel_type<typename MapT::ValueType>::type> &channel = channel_opt.get();
    MapT map;
    map.reserve(channel.numElements());
    for (size_t i = 0; i < channel.numElements(); i++)
    {
      map.insert(typename MapT::HandleType(i), channel[i]);
    }
    return map;
  }
  return boost::none;
}



template <typename MapT>
boost::optional<MapT> AttributeMeshIOBase::getAttributeMap(const std::string &name)
{
  typename AttributeChannel<typename channel_type<typename MapT::ValueType>::type>::Optional values_opt;
  typename IndexChannel::Optional indices_opt;
  if (getChannel(attribute_type<typename MapT::HandleType>::attr_group, name + "_idx", indices_opt) && getChannel(attribute_type<typename MapT::HandleType>::attr_group, name, values_opt) && indices_opt && values_opt && indices_opt.get().width() == 1 && values_opt.get().width() == channel_type<typename MapT::ValueType>::w && indices_opt.get().numElements() == values_opt.get().numElements())
  {
    auto &indices = indices_opt.get();
    auto &values = values_opt.get();
    MapT map;

    map.reserve(indices.numElements());
    for (size_t i = 0; i < indices.numElements(); i++)
    {
      map.insert(indices[i], values[i]);
    }
    return map;
  }
  return boost::none;
}

} /* namespace lvr2 */

#endif //LAS_VEGAS_MESHIOINTERFACE_TCC
