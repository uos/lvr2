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

#ifndef LAS_VEGAS_MESHIOINTERFACE_H
#define LAS_VEGAS_MESHIOINTERFACE_H

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

class MeshIOInterface
{
 public:

  /**
   * @brief Adds a HalfEdgeMesh to the persistence layer
   * @param hem The HalrfEdgeMesh which has to be stored
   * @return true if the HalfEdgeMesh has been written to the persistence layer, false otherwise
   */
  bool addMesh(const HalfEdgeMesh<BaseVec>& hem);

  /**
   * @brief Reads a HalfEdgeMesh from the persistence layer
   * @return Returns an optional to a HalfEdgeMesh which is valid if the mesh has been read successfully
   */
  boost::optional<HalfEdgeMesh<BaseVec>> getMesh();

  /**
   * @brief addDenseAttributeMap    Stores a dense attribute map to the persistence layer where the attribute group
   *                                is defined by the given map type.
   * @tparam MapT                   The map type of the map which has to be stored
   * @param map                     The map which has to be stored
   * @param name                    The name under which the map should be stored
   * @return                        true if the map has been stored successfully, false otherwise
   */
  template<typename MapT>
  bool addDenseAttributeMap(const MapT& map, const std::string& name);

  /**
   * @brief getDenseAttributeMap    Reads a dense attribute map from the persistence layer where the attribute group
   *                                is defined by the given map type.
   * @tparam MapT                   The map type of the map which has to be loaded
   * @param map                     The map which has to be loaded
   * @param name                    The name under which the map values can be found in the persistence layer
   * @return                        A valid optional to the map if the map has been loaded successfully, a none type
   *                                otherwise
   */
  template <typename MapT>
  boost::optional<MapT> getDenseAttributeMap(const std::string& name);

  /**
   * @brief addAttributeMap         Stores a general attribute map to the persistence layer where the attribute group
   *                                is defined by the given map type. Note, that the map keys are also stored, if you
   *                                want to store a dense attribute map, please see addDenseAttributeMap.
   * @tparam MapT                   The map type of the map which has to be stored
   * @param map                     The map which has to be stored
   * @param name                    The name under which the map should be stored
   * @return                        true if the map has been stored successfully, false otherwise
   */
  template<typename MapT>
  bool addAttributeMap(const MapT& map, const std::string& name);

  /**
   * @brief getAttributeMap         Reads a general attribute map from the persistence layer where the attribute group
   *                                is defined by the given map type. Note, that the map keys have to be available in
   *                                the persistence layer. If you want to load a DenseAttributeMap, please use
   *                                getDenseAttributeMap.
   * @tparam MapT                   The map type of the map which has to be loaded
   * @param map                     The map which has to be loaded
   * @param name                    The name under which the map values can be found in the persistence layer
   * @return                        A valid optional to the map if the map has been loaded successfully, a none type
   *                                otherwise
   */
  template<typename MapT>
  boost::optional<MapT> getAttributeMap(const std::string& name);

 private:

  /**
   * @brief Persistence layer interface, Accesses the vertices of the mesh in the persistence layer.
   * @return An optional float channel, the channel is valid if the mesh vertices have been read successfully
   */
  virtual FloatChannelOptional getVertices() = 0;

  /**
   * @brief Persistence layer interface, Accesses the face indices of the mesh in the persistence layer.
   * @return An optional index channel, the channel is valid if the mesh indices have been read successfully
   */
  virtual IndexChannelOptional getIndices() = 0;

  /**
   * @brief Persistence layer interface, Writes the vertices of the mesh to the persistence layer.
   * @return true if the channel has been written successfully
   */
  virtual bool addVertices(const FloatChannel& channel_ptr) = 0;

  /**
   * @brief Persistence layer interface, Writes the face indices of the mesh to the persistence layer.
   * @return true if the channel has been written successfully
   */
  virtual bool addIndices(const IndexChannel& channel_ptr) = 0;

  /**
   * @brief getChannel  Reads a float attribute channel in the given group with the given name
   * @param group       The associated attribute group
   * @param name        The associated attribute name
   * @param channel     The pointer to the float channel
   * @return            true if the channel has been loaded successfully, false otherwise
   */
  virtual bool getChannel(const std::string group, const std::string name, FloatChannelOptional& channel) = 0;

  /**
   * @brief getChannel  Reads an index attribute channel in the given group with the given name
   * @param group       The associated attribute group
   * @param name        The associated attribute name
   * @param channel     The pointer to the index channel
   * @return            true if the channel has been loaded successfully, false otherwise
   */
  virtual bool getChannel(const std::string group, const std::string name, IndexChannelOptional& channel) = 0;

  /**
   * @brief getChannel  Reads an unsigned char attribute channel in the given group with the given name
   * @param group       The associated attribute group
   * @param name        The associated attribute name
   * @param channel     The pointer to the unsigned char channel
   * @return            true if the channel has been loaded successfully, false otherwise
   */
  virtual bool getChannel(const std::string group, const std::string name, UCharChannelOptional& channel) = 0;


  /**
   * @brief addChannel  Writes a float attribute channel from the given group with the given name
   * @param group       The associated attribute group
   * @param name        The associated attribute name
   * @param channel     The pointer to the float channel which should be written
   * @return            true if the channel has been written successfully, false otherwise
   */
  virtual bool addChannel(const std::string group, const std::string name, const FloatChannel& channel) = 0;

  /**
   * @brief addChannel  Writes an index attribute channel from the given group with the given name
   * @param group       The associated attribute group
   * @param name        The associated attribute name
   * @param channel     The pointer to the index channel which should be written
   * @return            true if the channel has been written successfully, false otherwise
   */
  virtual bool addChannel(const std::string group, const std::string name, const IndexChannel& channel) = 0;

  /**
   * @brief addChannel  Writes an unsigned char attribute channel from the given group with the given name
   * @param group       The associated attribute group
   * @param name        The associated attribute name
   * @param channel     The pointer to the unsigned char channel which should be written
   * @return            true if the channel has been written successfully, false otherwise
   */
  virtual bool addChannel(const std::string group, const std::string name, const UCharChannel& channel) = 0;

};

}

#include "MeshIOInterface.tcc"

#endif //LAS_VEGAS_MESHIOINTERFACE_H
