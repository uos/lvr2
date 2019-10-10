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
 * Meap.hpp
 */

#ifndef LVR2_UTIL_MEAP_H_
#define LVR2_UTIL_MEAP_H_

#include <vector>
#include <utility>
#include <unordered_map>

#include <boost/optional.hpp>

#include "lvr2/attrmaps/AttributeMap.hpp"

using std::unordered_map;
using std::pair;


namespace lvr2
{

/**
 * @brief Element in a meap, consisting of a key and a value.
 *
 * This is equivalent to `pair<KeyT, ValueT>`, but with proper names instead of
 * `first` and `second`.
 */
template<typename KeyT, typename ValueT>
class MeapPair
{
public:
    MeapPair(KeyT key, ValueT value)
    :m_key(key)
    ,m_value(value)
    {}

    inline KeyT& key() {
        return m_key;
    }

    inline const KeyT& key() const {
        return m_key;
    }

    inline ValueT& value() {
        return m_value;
    }

    inline const ValueT& value() const {
        return m_value;
    }
private:
    KeyT m_key;
    ValueT m_value;
};

/**
 * @brief A map combined with a binary heap.
 *
 * The elements in the meap are pairs of `KeyT` and `ValueT`. Only the latter
 * is used for sorting the heap. The former can be used to lookup the value, as
 * in a regular map. Combining both, a map and a heap, allows to implement the
 * `updateValue()` method, which would otherwise be impossible with a simple
 * binary heap. In this library, this type is often used with some kind of
 * "cost" as value and a handle as key.
 *
 * This implementation is a min heap: the smallest value sits "at the top" and
 * can be retrieved in O(1) via `popMin()` or `peekMin()`.
 */
template<typename KeyT, typename ValueT>
class Meap
{
public:
    /**
     * @brief Initializes an empty meap.
     */
    Meap() {}

    /**
     * @brief Initializes an empty meap and reserves memory for at least
     *        `capacity` many elements.
     */
    Meap(size_t capacity);


    // =======================================================================
    // These methode work exactly like the ones from `AttributeMap`
    // =======================================================================
    bool containsKey(KeyT key) const;
    boost::optional<ValueT> insert(KeyT key, const ValueT& value);
    boost::optional<ValueT> erase(KeyT key);
    void clear();
    boost::optional<const ValueT&> get(KeyT key) const;
    size_t numValues() const;


    /**
     * @brief Returns a reference to the minimal value with its corresponding
     *        key.
     */
    const MeapPair<KeyT, ValueT>& peekMin() const;

    /**
     * @brief Removes the minimal value with its corresponding key from the
     *        meap and returns it.
     */
    MeapPair<KeyT, ValueT> popMin();

    /**
     * @brief Updates the value of `key` to `newValue` and repairs the heap.
     *
     * The new value might be lower or greater than the old value. The heap
     * is repaired either way. If the new value equals the old one, this method
     * does nothing.
     */
    void updateValue(const KeyT& key, const ValueT& newValue);

    /**
     * @brief Returns `true` iff the meap is empty.
     */
    bool isEmpty() const;

private:
    // This is the main heap which stores the costs as well as all keys.
    std::vector<MeapPair<KeyT, ValueT>> m_heap;

    // This is a map to quickly look up the index within `m_heap` at which a
    // specific key lives.
    unordered_map<KeyT, size_t> m_indices;

    /**
     * @brief Returns the index of the father of the child at index `child`.
     */
    size_t father(size_t child) const;

    /**
     * @brief Returns the index of the left child of the father at index
     *        `father`.
     */
    size_t leftChild(size_t father) const;

    /**
     * @brief Returns the index of the right child of the father at index
     *        `father`.
     */
    size_t rightChild(size_t father) const;

    /**
     * @brief Performs the `bubbleUp` heap operation on the node at `idx`.
     *
     * As long as the father of the node at `idx` still has a greater value
     * than the value of `idx`, both are swapped.
     */
    void bubbleUp(size_t idx);

    /**
     * @brief Performs the `bubbleDown()` heap operation on the node at `idx`.
     */
    void bubbleDown(size_t idx);

    /// Just for debugging purposes. Prints a bunch of information.
    void debugOutput() const;
};

} // namespace lvr2

#include "lvr2/util/Meap.tcc"

#endif /* LVR2_UTIL_MEAP_H_ */
