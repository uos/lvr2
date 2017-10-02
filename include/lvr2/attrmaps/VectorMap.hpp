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

#ifndef LVR2_ATTRMAPS_VECTORMAP_H_
#define LVR2_ATTRMAPS_VECTORMAP_H_

#include <vector>
#include <boost/optional.hpp>

#include <lvr2/attrmaps/StableVector.hpp>
#include <lvr2/attrmaps/AttributeMap.hpp>
#include <lvr2/geometry/Handles.hpp>
#include <lvr2/util/Cluster.hpp>

using std::vector;
using boost::optional;

namespace lvr2
{

/**
 * @brief A map with constant lookup overhead using small-ish integer-keys.
 *
 * It stores the given values in a vector, they key is simply the index within
 * the vector. This means that the space requirement is O(largest_key). See
 * StableVector for more information.
 */
template<typename HandleT, typename ValueT>
class VectorMap : public AttributeMap<HandleT, ValueT>
{
public:
    /**
     * @brief Creates an empty map without default element set.
     */
    VectorMap() {}

    /**
     * @brief Creates a map with a given default value.
     *
     * Whenever you request a value for a key and there isn't a value
     * associated with that key, the default value is returned.  Note that if
     * you set a default value (which you do by calling this constructor), you
     * can't remove it. Neither `erase()` nor `clear()` will do it. Calls to
     * `get()` will always return a non-none value and `operator[]` won't ever
     * panic.
     *
     * One additional important detail: if you call `get()` to obtain a
     * mutable reference, the default value is inserted into the map. This is
     * the only sane way to return a mutably reference.
     */
    VectorMap(const ValueT& defaultValue);

    /**
     * @brief Creates a map with a given default value and calls reserve.
     *
     * This works exactly as the `VectorMap(const Value&)` constructor, but
     * also calls `reserve(countElements)` immediately afterwards.
     */
    VectorMap(size_t countElements, const ValueT& defaultValue);

    // =======================================================================
    // Implemented methods from the interface (check interface for docs)
    // =======================================================================
    bool containsKey(HandleT key) const final;
    optional<ValueT> insert(HandleT key, const ValueT& value) final;
    optional<ValueT> erase(HandleT key) final;
    void clear() final;
    optional<ValueT&> get(HandleT key) final;
    optional<const ValueT&> get(HandleT key) const final;
    size_t numValues() const final;

    AttributeMapHandleIteratorPtr<HandleT> begin() const final;
    AttributeMapHandleIteratorPtr<HandleT> end() const final;


    /**
     * @see StableVector::reserve(size_t)
     */
    void reserve(size_t newCap);

private:
    /// The underlying storage
    StableVector<HandleT, ValueT> m_vec;
    optional<ValueT> m_default;
};

template<typename HandleT, typename ValueT>
class VectorMapIterator : public AttributeMapHandleIterator<HandleT>
{
    static_assert(
        std::is_base_of<BaseHandle<Index>, HandleT>::value,
        "HandleT must inherit from BaseHandle!"
    );

public:
    VectorMapIterator(StableVectorIterator<HandleT, ValueT> iter);

    AttributeMapHandleIterator<HandleT>& operator++() final;
    bool operator==(const AttributeMapHandleIterator<HandleT>& other) const final;
    bool operator!=(const AttributeMapHandleIterator<HandleT>& other) const final;
    HandleT operator*() const final;
    std::unique_ptr<AttributeMapHandleIterator<HandleT>> clone() const final;

private:
    StableVectorIterator<HandleT, ValueT> m_iter;
};

} // namespace lvr2

#include <lvr2/attrmaps/VectorMap.tcc>

#endif /* LVR2_ATTRMAPS_VECTORMAP_H_ */
