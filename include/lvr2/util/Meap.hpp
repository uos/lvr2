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
 * Meap.hpp
 */

#ifndef LVR2_UTIL_MEAP_H_
#define LVR2_UTIL_MEAP_H_

#include <vector>
#include <utility>

#include <boost/optional.hpp>

#include <lvr2/attrmaps/AttributeMap.hpp>

using std::vector;
using std::pair;
using boost::optional;

namespace lvr2
{

/**
 * @brief Element in a meap, consisting of a key and a value.
 *
 * This is equivalent to `pair<KeyT, ValueT>`, but with proper names instead of
 * `first` and `second`.
 */
template<typename KeyT, typename ValueT>
struct MeapPair
{
    KeyT key;
    ValueT value;
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
template<typename KeyT, typename ValueT, template<typename, typename> typename MapT>
class Meap
{
    static_assert(
        std::is_base_of<AttributeMap<KeyT, size_t>, MapT<KeyT, size_t>>::value,
        "MapT must implement from AttributeMap!"
    );
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
    optional<ValueT> insert(KeyT key, const ValueT& value);
    optional<ValueT> erase(KeyT key);
    void clear();
    optional<const ValueT&> get(KeyT key) const;
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
    vector<MeapPair<KeyT, ValueT>> m_heap;

    // This is a map to quickly look up the index within `m_heap` at which a
    // specific key lives.
    MapT<KeyT, size_t> m_indices;

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

#include <lvr2/util/Meap.tcc>

#endif /* LVR2_UTIL_MEAP_H_ */