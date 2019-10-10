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
 * HashMap.hpp
 *
 *  @date 27.07.2017
 */

#ifndef LVR2_ATTRMAPS_HASHMAP_H_
#define LVR2_ATTRMAPS_HASHMAP_H_

#include <unordered_map>

#include "lvr2/attrmaps/AttributeMap.hpp"

using std::unordered_map;

namespace lvr2
{

template<typename HandleT, typename ValueT>
class HashMap : public AttributeMap<HandleT, ValueT>
{
public:
    /**
     * @brief Creates an empty map without default element set.
     */
    HashMap() {}

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
    HashMap(const ValueT& defaultValue);

    /**
     * @brief Creates a map with a given default value and calls reserve.
     *
     * This works exactly as the `HashMap(const Value&)` constructor, but
     * also calls `reserve(countElements)` immediately afterwards.
     */
    HashMap(size_t countElements, const ValueT& defaultValue);

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
    unordered_map<HandleT, ValueT> m_map;
    boost::optional<ValueT> m_default;
};

template<typename HandleT, typename ValueT>
class HashMapIterator : public AttributeMapHandleIterator<HandleT>
{
public:
    HashMapIterator(typename unordered_map<HandleT, ValueT>::const_iterator iter);

    AttributeMapHandleIterator<HandleT>& operator++() final;
    bool operator==(const AttributeMapHandleIterator<HandleT>& other) const final;
    bool operator!=(const AttributeMapHandleIterator<HandleT>& other) const final;
    HandleT operator*() const final;
    std::unique_ptr<AttributeMapHandleIterator<HandleT>> clone() const final;

private:
    typename unordered_map<HandleT, ValueT>::const_iterator m_iter;
};

} // namespace lvr2

#include "lvr2/attrmaps/HashMap.tcc"

#endif /* LVR2_ATTRMAPS_HASHMAP_H_ */
