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

/*
 * ListMap.hpp
 *
 *  @date 27.07.2017
 */

#ifndef LVR2_ATTRMAPS_LISTMAP_H_
#define LVR2_ATTRMAPS_LISTMAP_H_

#include <utility>
#include <vector>

#include "lvr2/attrmaps/AttributeMap.hpp"

using std::vector;
using std::pair;

namespace lvr2
{


/**
 * @brief A simple implementation of AttributeMap for a small number of values.
 *
 * This implementation uses a simple, unordered list of key-value pairs to
 * represent the map. This means that nearly all operations have a complexity
 * of O(number_of_values), which is rather suboptimal. Thus this implementation
 * only makes sense when the number of values is expected to be very small. A
 * modern computer can easily search linearly through, like, 16 things. When
 * we're dealing with a small number of things, often linear search will be
 * faster than something fancy (like hashing or binary search).
 *
 * However, this implementation doesn't use its whole potential right now. The
 * biggest speed gain is possible by using small buffer optimization (SBO).
 * This still needs to be implemented, but should be fairly straight forward.
 */
template<typename HandleT, typename ValueT>
class ListMap : public AttributeMap<HandleT, ValueT>
{
public:
    /**
     * @brief Creates an empty map without default element set.
     */
    ListMap() {}

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
    ListMap(const ValueT& defaultValue);

    /**
     * @brief Creates a map with a given default value and calls reserve.
     *
     * This works exactly as the `ListMap(const Value&)` constructor, but
     * also calls `reserve(countElements)` immediately afterwards.
     */
    ListMap(size_t countElements, const ValueT& defaultValue);

    // =======================================================================
    // Implemented methods from the interface (check interface for docs)
    // =======================================================================
    bool containsKey(HandleT key) const final;
    boost::optional<ValueT> insert(HandleT key, const ValueT& value) final;
    boost::optional<ValueT> erase(HandleT key) final;
    void clear() final;
    boost::optional<ValueT&> get(HandleT key) final;
    boost::optional<const ValueT&> get(HandleT key) const final;
    size_t numValues() const final;

    AttributeMapHandleIteratorPtr<HandleT> begin() const final;
    AttributeMapHandleIteratorPtr<HandleT> end() const final;

    /**
     * @brief Allocates space for at least `newCap` more elements.
     */
    void reserve(size_t newCap);

private:
    vector<pair<HandleT, ValueT>> m_list;
    boost::optional<ValueT> m_default;

    // Internal helper method
    typename vector<pair<HandleT, ValueT>>::const_iterator keyIterator(HandleT key) const;
    typename vector<pair<HandleT, ValueT>>::iterator keyIterator(HandleT key);

    template<typename, typename> friend class ListMap;
};

template<typename HandleT, typename ValueT>
class ListMapIterator : public AttributeMapHandleIterator<HandleT>
{
public:
    ListMapIterator(typename vector<pair<HandleT, ValueT>>::const_iterator iter);

    AttributeMapHandleIterator<HandleT>& operator++() final;
    bool operator==(const AttributeMapHandleIterator<HandleT>& other) const final;
    bool operator!=(const AttributeMapHandleIterator<HandleT>& other) const final;
    HandleT operator*() const final;
    std::unique_ptr<AttributeMapHandleIterator<HandleT>> clone() const final;

private:
    typename vector<pair<HandleT, ValueT>>::const_iterator m_iter;
};

} // namespace lvr2

#include "lvr2/attrmaps/ListMap.tcc"

#endif /* LVR2_ATTRMAPS_LISTMAP_H_ */
