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
#include <utility>

using std::move;
using std::vector;

#include <lvr2/geometry/BaseHandle.hpp>
#include <lvr2/geometry/Handles.hpp>


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
    StableVectorIterator(const vector<bool>* deleted, bool startAtEnd = false);

    StableVectorIterator& operator=(const StableVectorIterator& other);
    bool operator==(const StableVectorIterator& other) const;
    bool operator!=(const StableVectorIterator& other) const;

    StableVectorIterator& operator++();

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

    ~StableVector();

    /**
     * @brief Creates a StableVector with `countElements` many copies of
     *        `defaultValue`.
     *
     * The elements are stored contiguously in the vectors, thus the valid
     * indices of these elements are 0 to `countElements` - 1.
     */
    StableVector(size_t countElements, const ElementType& defaultValue);

    /**
     * @brief Copy constructor deep clones the `other` vector.
     */
    StableVector(const StableVector& other);

    /**
     * @brief Move constructor salvages the `other` vector to copy contents.
     */
    StableVector(StableVector&& other);

    /**
     * @brief Copy assignment operator (does nothing special).
     */
    StableVector<HandleT, ElemT>& operator=(const StableVector& other);
    /**
     * @brief Move assignment operator (does nothing special).
     */
    StableVector<HandleT, ElemT>& operator=(StableVector&& other);

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
    StableVectorIterator<HandleType> begin() const;

    /**
     * @brief Returns an iterator to the element after the last element of
     *        this vector.
     */
    StableVectorIterator<HandleType> end() const;

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
    /// Wrapper for the actual data to avoid calling the constructor or
    /// destructor in certain situations.
    union UnsafeWrapper
    {
        ElementType data;

        // Empty constructor and destructor. The StableVector takes care of
        // initialization and destruction.
        UnsafeWrapper() {}
        ~UnsafeWrapper() {}

        // UnsafeWrapper(const ElementType& data) : data(data) {}
        // UnsafeWrapper(ElementType&& data) : data(data) {}

        void set(const ElementType& value)
        {
            // This funky expression is the placement-new operator. It
            // constructs an object at the given memory location (meaning: it
            // does not allocate). It won't attempt to destruct the object
            // living at the location, which is good because it's probably
            // only garbage!
            cout << "# UnsafeWrapper::set() copy, data @ " << &data << ", this @ " << this << endl;
            new (&data) ElementType(value);
        }
        void set(ElementType&& value)
        {
            // See above for explanation of the placement-new operator. Here we
            // use the move constructor of `ElementType`.
            cout << "# UnsafeWrapper::set() move, data @ " << &data << ", this @ " << this << endl;
            new (&data) ElementType(move(value));
        }

        // Copy and move constructor/assignment operator don't do anything. The
        // StableVector has to take care of copying elements!
        UnsafeWrapper(const UnsafeWrapper& other) : data(other.data) {
            cout << "unsafe copy ctor" << endl;
        }
        UnsafeWrapper(UnsafeWrapper&& other)
            // This beauty declares this function `noexcept` if the move ctor
            // of `ElementType` is `noexcept`. This is actually important as
            // the reallocation of `std::vector` is faster for types with
            // noexcept move ctor.
            noexcept(noexcept(ElementType(std::declval<ElementType>())))
            : data(move(other.data))

        {
            cout << "unsafe move ctor" << endl;
        }

        UnsafeWrapper& operator=(const UnsafeWrapper& other) {
            cout << "unsafe copy assignment" << endl;
        }
        UnsafeWrapper& operator=(UnsafeWrapper&& other) {
            cout << "unsafe move assignment" << endl;
        }
    };

    // static_assert(!(std::is_same<ElementType, string>::value
    //     && !std::is_nothrow_move_constructible<UnsafeWrapper>::value), "ahh");
    static_assert(std::is_nothrow_move_constructible<UnsafeWrapper>::value, "ahh2");

    /// Count of used elements in elements vector
    size_t m_usedCount;

    /// Vector for stored elements
    vector<UnsafeWrapper> m_elements;

    /// Stores whether an element in `m_elements` is deleted or not. Always has
    /// the same length as `m_elements`.
    vector<bool> m_deleted;

    /**
     * @brief Assert that the requested handle is not deleted or throw an
     *        exception otherwise.
     */
    void checkAccess(HandleType handle) const;
};

} // namespace lvr2

#include <lvr2/util/StableVector.tcc>

#endif /* LVR2_UTIL_STABLEVECTOR_H_ */
