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
 * StableVector.hpp
 *
 *  @date 08.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_ATTRMAPS_STABLEVECTOR_H_
#define LVR2_ATTRMAPS_STABLEVECTOR_H_

#include <vector>
#include <utility>
#include <boost/optional.hpp>
#include <boost/shared_array.hpp>

using std::move;
using std::vector;


#include "lvr2/util/BaseHandle.hpp"
#include "lvr2/geometry/Handles.hpp"


namespace lvr2
{

/**
 * @brief Iterator over handles in this vector, which skips deleted elements
 *
 * Important: This is NOT a fail fast iterator. If the vector is changed while
 * using an instance of this iterator the behavior is undefined!
 */
template<typename HandleT, typename ElemT>
class StableVectorIterator
{
private:
    /// Reference to the deleted marker array this iterator belongs to
    const vector<boost::optional<ElemT>>* m_elements;

    /// Current position in the vector
    size_t m_pos;
public:
    StableVectorIterator(const vector<boost::optional<ElemT>>* deleted, bool startAtEnd = false);

    StableVectorIterator& operator=(const StableVectorIterator& other);
    bool operator==(const StableVectorIterator& other) const;
    bool operator!=(const StableVectorIterator& other) const;

    StableVectorIterator& operator++();

    bool isAtEnd() const;

    HandleT operator*() const;
};

/**
 * @brief A vector which guarantees stable indices and features O(1) deletion.
 *
 * This is basically a wrapper for the std::vector, which marks an element as
 * deleted but does not actually delete it. This means that indices are never
 * invalidated. When inserting an element, you get its index (its so called
 * "handle") back. This handle can later be used to access the element. This
 * remains true regardless of other insertions and deletions happening in
 * between.
 *
 * USE WITH CAUTION: This NEVER frees memory of deleted values (except on its
 * own destruction and can get very large if used incorrectly! If deletions in
 * your use-case are far more numerous than insertions, this data structure is
 * probably not fitting your needs. The memory requirement of this class is
 * O(n_p) where n_p is the number of `push()` calls.
 *
 * @tparam HandleT This handle type contains the actual index. It has to be
 *                 derived from `BaseHandle`!
 * @tparam ElemT Type of elements in the vector.
 */
template<typename HandleT, typename ElemT>
class StableVector
{
    static_assert(
        std::is_base_of<BaseHandle<Index>, HandleT>::value,
        "HandleT must inherit from BaseHandle!"
    );

public:

    using ElementType = ElemT;
    using HandleType = HandleT;

    /**
     * @brief Creates an empty StableVector.
     */
    StableVector() : m_usedCount(0) {};

    /**
     * @brief Creates a StableVector with `countElements` many copies of
     *        `defaultValue`.
     *
     * The elements are stored contiguously in the vectors, thus the valid
     * indices of these elements are 0 to `countElements` - 1.
     */
    StableVector(size_t countElements, const ElementType& defaultValue);

    StableVector(size_t countElements, const boost::shared_array<ElementType>& sharedArray);

    /**
     * @brief Adds the given element to the vector.
     *
     * @return The handle referring to the inserted element.
     */
    HandleType push(const ElementType& elem);

    /**
     * @brief Adds the given element by moving from it.
     *
     * @return The handle referring to the inserted element.
     */
    HandleType push(ElementType&& elem);

    /**
     * @brief Increases the size of the vector to the length of `upTo`.
     *
     * This means that the next call to `push()` after calling `resize(upTo)`
     * will return exactly the `upTo` handle. All elements that are inserted
     * by this method are marked as deleted and thus aren't initialized. They
     * can be set later with `set()`.
     *
     * If `upTo` is already a valid handle, this method will panic!
     */
    void increaseSize(HandleType upTo);

    /**
     * @brief Increases the size of the vector to the length of `upTo` by
     *        inserting copies of `elem`.
     *
     * This means that the next call to `push()` after calling `resize(upTo)`
     * will return exactly the `upTo` handle.
     *
     * If `upTo` is already a valid handle, this method will panic!
     */
    void increaseSize(HandleType upTo, const ElementType& elem);

    /**
     * @brief The handle which would be returned by calling `push` now.
     */
    HandleType nextHandle() const;

    /**
     * @brief Mark the element behind the given handle as deleted.
     *
     * While the element is deleted, the handle stays valid. This means that
     * trying to obtain the element with this handle later, will always result
     * in `none` (if `get()` was used). Additionally, the handle can also be
     * used with the `set()` method.
     */
    void erase(HandleType handle);

    /**
     * @brief Removes all elements from the vector.
     */
    void clear();

    /**
     * @brief Returns the element referred to by `handle`.
     *
     * Returns `none` if the element was deleted or if the handle is out of
     * bounds.
     */
    boost::optional<ElementType&> get(HandleType handle);

    /**
     * @brief Returns the element referred to by `handle`.
     *
     * Returns `none` if the element was deleted or if the handle is out of
     * bounds.
     */
    boost::optional<const ElementType&> get(HandleType handle) const;

    /**
     * @brief Set a value for the existing `handle`.
     *
     * In this method, the `handle` has to be valid: it has to be obtained by
     * a prior `push()` call. If you want to insert a new element, use `push()`
     * instead of this `set()` method!
     */
    void set(HandleType handle, const ElementType& elem);

    /**
     * @brief Set a value for the existing `handle` by moving from `elem`.
     *
     * In this method, the `handle` has to be valid: it has to be obtained by
     * a prior `push()` call. If you want to insert a new element, use `push()`
     * instead of this `set()` method!
     */
    void set(HandleType handle, ElementType&& elem);

    /**
     * @brief Returns the element referred to by `handle`.
     *
     * If `handle` is out of bounds or the element was deleted, this method
     * will throw an exception in debug mode and has UB in release mode. Use
     * `get()` instead to gracefully handle the absence of an element.
     */
    ElementType& operator[](HandleType handle);

    /**
     * @brief Returns the element referred to by `handle`.
     *
     * If `handle` is out of bounds or the element was deleted, this method
     * will throw an exception in debug mode and has UB in release mode. Use
     * `get()` instead to gracefully handle the absence of an element.
     */
    const ElementType& operator[](HandleType handle) const;

    /**
     * @brief Absolute size of the vector (including deleted elements).
     */
    size_t size() const;

    /**
     * @brief Number of non-deleted elements.
     */
    size_t numUsed() const;

    /**
     * @brief Returns an iterator to the first element of this vector.
     *
     * This iterator auto skips deleted elements and returns handles to the
     * valid elements.
     */
    StableVectorIterator<HandleType, ElementType> begin() const;

    /**
     * @brief Returns an iterator to the element after the last element of
     *        this vector.
     */
    StableVectorIterator<HandleType, ElementType> end() const;

    /**
     * @brief Increase the capacity of the vector to a value that's greater or
     *        equal to newCap.
     *
     * If newCap is greater than the current capacity, new storage is
     * allocated, otherwise the method does nothing.
     *
     * @param newCap new capacity of the vector
     */
    void reserve(size_t newCap);

private:
    /// Count of used elements in elements vector
    size_t m_usedCount;

    /// Vector for stored elements
    vector<boost::optional<ElementType>> m_elements;

    /**
     * @brief Assert that the requested handle is not deleted or throw an
     *        exception otherwise.
     */
    void checkAccess(HandleType handle) const;
};

} // namespace lvr2

#include "lvr2/attrmaps/StableVector.tcc"

#endif /* LVR2_ATTRMAPS_STABLEVECTOR_H_ */
