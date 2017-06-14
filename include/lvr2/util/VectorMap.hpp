/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/*
 * VectorMap.hpp
 *
 *  @date 08.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_UTIL_VECTORMAP_H_
#define LVR2_UTIL_VECTORMAP_H_

#include <vector>
#include <boost/optional.hpp>

using std::vector;

namespace lvr2
{

/**
 * @brief A map with constant lookup overhead
 *
 * It stores the given values in a vector, where the given key is the key in the vector
 *
 * USE WITH CAUTION: This NEVER deletes values and can get very large!
 *
 * @tparam KeyT Type of keys for this map. Needs to be a subclass of BaseHandle!
 * @tparam ValT Type of values of this map
 */
template<typename KeyT, typename ValT>
class VectorMap
{
private:
    using KeyType = KeyT;
    using ValueType = ValT;

    /// Wrapper for dummy elements
    union Wrapper
    {
        ValueType data;

        Wrapper() {};
        Wrapper(const ValueType& data) : data(data) {};
        Wrapper(const Wrapper& wrapper) : data(wrapper.data) {};
        ~Wrapper() {};

        Wrapper& operator=(const Wrapper& value);
    };

    /// Count of used values in elements vector
    size_t m_usedCount;

    /// Vector for stored values
    vector<Wrapper> m_elements;

    /// Vector for flags, if the same position in the elements vector is deleted
    vector<bool> m_deleted;

    /**
     * @brief Check, if the requested handle is not deleted
     */
    void checkAccess(const KeyType& key) const;

public:
    VectorMap() : m_usedCount(0) {};

    /// Creates a map of size countElements with countElements copies of defaultValue in it
    VectorMap(size_t countElements, const ValueType& defaultValue);

    /// Insert the given element
    void insert(const KeyType& key, const ValueType& value);

    /**
     * @brief Mark the value behind the given key as deleted
     *
     * This does NOT call the DESTRUCTOR of the marked value!
     */
    void erase(const KeyType& key);

    /// Request the value behind the given key
    boost::optional<ValueType&> get(const KeyType& key);

    /// Request the value behind the given key
    boost::optional<const ValueType&> get(const KeyType& key) const;

    /**
     * @brief Request the value behind the given key
     *
     * Important: Do not use this to insert new values! Use insert instead.
     */
    ValueType& operator[](const KeyType& key);

    /// Request the value behind the given key
    const ValueType& operator[](const KeyType& key) const;

    /// Number of not delete-marked values
    size_t sizeUsed() const;

    // TODO: add reserve method to reserve vector memory
};

template <typename BaseVecT, typename ValT>
using EdgeMap = VectorMap<typename BaseMesh<BaseVecT>::EdgeHandle, ValT>;
template <typename BaseVecT, typename ValT>
using FaceMap = VectorMap<typename BaseMesh<BaseVecT>::FaceHandle, ValT>;
template <typename BaseVecT, typename ValT>
using VertexMap = VectorMap<typename BaseMesh<BaseVecT>::VertexHandle, ValT>;

} // namespace lvr2

#include <lvr2/util/VectorMap.tcc>

#endif /* LVR2_UTIL_VECTORMAP_H_ */
