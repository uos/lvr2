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
 * StableVector.hpp
 *
 *  @date 08.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_UTIL_STABLEVECTOR_H_
#define LVR2_UTIL_STABLEVECTOR_H_

#include <vector>

using std::vector;

namespace lvr2
{

/**
 * @brief Iterator over handles in this vector, which skips deleted elements
 *
 * Important: This is NOT a fail fast iterator. If the vector is changed while using an instance of this
 * iterator the behavior is undefined!
 */
template<typename HandleT>
class StableVectorIterator
{
private:
    /// Reference to the deleted marker array this iterator belongs to
    const vector<bool>* m_deleted;

    /// Current position in the vector
    size_t m_pos;
public:
    StableVectorIterator(const vector<bool>* deleted, bool startAtEnd = false)
            : m_deleted(deleted), m_pos(startAtEnd ? deleted->size() : 0) {};

    StableVectorIterator& operator=(const StableVectorIterator& other);
    bool operator==(const StableVectorIterator& other) const;
    bool operator!=(const StableVectorIterator& other) const;

    StableVectorIterator& operator++();

    HandleT operator*() const;
};

/**
 * @brief A vector, which preserves its indices even when an element is deleted
 *
 * This is basically a wrapper for the std::vector, which marks an element as
 * deleted but does not actually delete it.
 *
 * USE WITH CAUTION: This NEVER deletes values (except on its own destruction)
 * and can get very large if used incorrectly! This class is designed for
 * situations where the number deletes are not greatly more than the number
 * of insertions. The memory requirements of this class are O(n_pb) where n_pb
 * is the number of `push_back()` calls.
 *
 * @tparam ElemT Type of elements in the vector
 * @tparam HandleT Type of the index for the vector
 */
template<typename HandleT, typename ElemT>
class StableVector
{
private:
    using ElementType = ElemT;
    using HandleType = HandleT;

    /// Count of used elements in elements vector
    size_t m_usedCount;

    /// Vector for stored elements
    vector<ElementType> m_elements;

    /// Vector for flags, if the same position in the elements vector is deleted
    vector<bool> m_deleted;

    /**
     * @brief Check, if the requested handle is not deleted
     */
    void checkAccess(const HandleType& handle) const;

public:
    StableVector() : m_usedCount(0) {};

    StableVector(size_t countElements, const ElementType& defaultValue);

    /// Add the given element
    HandleT push_back(const ElementType& elem);

    /// The handle which would be returned by calling `push_back` now.
    HandleT nextHandle() const;

    /**
     * @brief Mark the element behind the given handle as deleted
     *
     * This does NOT call the DESTRUCTOR of the marked element!
     */
    void erase(const HandleType& handle);

    /// Request the value behind the given key
    boost::optional<ElemT&> get(const HandleType& key);

    /// Request the value behind the given key
    boost::optional<const ElemT&> get(const HandleType& key) const;

    /// Request the element behind the given handle
    ElemT& operator[](const HandleType& handle);

    /// Request the element behind the given handle
    const ElemT& operator[](const HandleType& handle) const;

    /// Absolute size of the vector (with delete-marked elements)
    size_t size() const;

    /// Number of not delete-marked elements
    size_t sizeUsed() const;

    /**
     * @brief Returns an iterator which starts at the beginning of the vector
     *
     * This iterator auto skips deleted elements and returns handles to the valid elements
     */
    StableVectorIterator<HandleT> begin() const;

    /**
     * @brief Returns an iterator which starts at the end of the vector
     *
     * This iterator auto skips deleted elements and returns handles to the valid elements
     */
    StableVectorIterator<HandleT> end() const;

    // TODO: add reserve method to reserve vector memory
};

} // namespace lvr2

#include <lvr2/util/StableVector.tcc>

#endif /* LVR2_UTIL_STABLEVECTOR_H_ */
