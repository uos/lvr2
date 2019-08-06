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
#include "lvr2/geometry/Handles.hpp"
#include "lvr2/attrmaps/AttrMaps.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/types/BaseBuffer.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"

#include "lvr2/io/GroupedChannelIO.hpp"
#include "lvr2/io/MeshGeometryIO.hpp"

namespace lvr2{

using BaseVec = BaseVector<float>;

class AttributeMeshIOBase : public MeshGeometryIO, public GroupedChannelIO
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
   * @param mesh                    The mesh which correspond to the map
   * @param map                     The map which has to be stored
   * @param name                    The name under which the map should be stored
   * @return                        true if the map has been stored successfully, false otherwise
   */
  template<typename MapT, typename BaseVecT>
  bool addDenseAttributeMap(const BaseMesh<BaseVecT>& mesh, const MapT& map, const std::string& name);

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

};

}

#include "AttributeMeshIOBase.tcc"

#endif //LAS_VEGAS_MESHIOINTERFACE_H
