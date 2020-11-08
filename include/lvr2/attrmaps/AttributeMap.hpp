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
 * AttributeMap.hpp
 *
 *  @date 26.07.2017
 */

#ifndef LVR2_ATTRMAPS_ATTRIBUTEMAP_H_
#define LVR2_ATTRMAPS_ATTRIBUTEMAP_H_

#include <boost/optional.hpp>
#include <memory>

#include "lvr2/geometry/Handles.hpp"

namespace lvr2
{

// Forward declarations
template<typename> class AttributeMapHandleIteratorPtr;

/**
 * @brief Interface for attribute maps.
 *
 * Attribute maps are associative containers which map from a handle to a
 * value. A simple and obvious implementation of this interface is a hash map.
 *
 * Attribute maps are used a lot in this library and are widely useful. A good
 * example is an algorithm that needs to visit every face by traversing a mesh,
 * but has to make sure to visit every face only once. In that algorithm, the
 * best idea is to use an attribute map which maps from face to bool. This
 * means that we associate a boolean value with each face. This boolean value
 * can be used to store whether or not we already visited that face. Such a
 * map would have the form `AttributeMap<FaceHandle, bool>`.
 *
 * Attribute maps are also used to store non-temporary data, like face-normals,
 * vertex-colors, and much more. It's pretty simple, really: if you want to
 * associate a value of type `T` with a, say, vertex, simply create an
 * `AttributeMap<VertexHandle, T>`.
 *
 * There are different implementations of this interface. The most important
 * ones have a type alias in `AttrMaps.hpp`. Please read the documentation in
 * that file to learn more about different implementations.
 *
 * @tparam HandleT Key type of this map. Has to inherit from `BaseHandle`!
 * @tparam ValueT The type to map to.
 */
template<typename HandleT, typename ValueT>
class AttributeMap
{
    static_assert(
        std::is_base_of<BaseHandle<Index>, HandleT>::value,
        "HandleT must inherit from BaseHandle!"
    );

public:
    /// The type of the handle used as key in this map
    typedef HandleT HandleType;

    /// The type of the value stored in this map
    typedef ValueT ValueType;

    /**
     * @brief Returns true iff the map contains a value associated with the
     *        given key.
     */
    virtual bool containsKey(HandleT key) const = 0;

    /**
     * @brief Inserts the given value at the given key position.
     *
     * @return If there was a value associated with the given key before
     *         inserting the new value, the old value is returned. None
     *         otherwise.
     */
    virtual boost::optional<ValueT> insert(HandleT key, const ValueT& value) = 0;

    /**
     * @brief Removes the value associated with the given key.
     *
     * @return If there was a value associated with the key, it is returned.
     *         None otherwise.
     */
    virtual boost::optional<ValueT> erase(HandleT key) = 0;

    /**
     * @brief Removes all values from the map.
     */
    virtual void clear() = 0;

    /**
     * @brief Returns the value associated with the given key or None
     *        if there is no associated value.
     *
     * Note: this method can not be used to insert a new value. It only allows
     * reading and modifying an already inserted value.
     */
    virtual boost::optional<ValueT&> get(HandleT key) = 0;

    /**
     * @brief Returns the value associated with the given key or None
     *        if there is no associated value.
     *
     * Note: this method can not be used to insert a new value. It only allows
     * reading an already inserted value.
     */
    virtual boost::optional<const ValueT&> get(HandleT key) const = 0;

    /**
     * @brief Returns the number of values in this map.
     */
    virtual size_t numValues() const = 0;

    /**
     * @brief Returns an iterator over all keys of this map. The order of
     *        iteration is unspecified.
     *
     * You can simply iterate over all keys of this map with a range-based
     * for-loop:
     *
     * \code{.cpp}
     *     for (auto handle: attributeMap) { ... }
     * \endcode
     */
    virtual AttributeMapHandleIteratorPtr<HandleT> begin() const = 0;

    /**
     * @brief Returns an iterator to the end of all keys.
     */
    virtual AttributeMapHandleIteratorPtr<HandleT> end() const = 0;

    /**
     * @brief Returns the value associated with the given key or panics
     *        if there is no associated value.
     *
     * Note: since this method panics, if there is no associated value, it
     * cannot be used to insert new values. Use `insert()` if you want to
     * insert new values.
     */
    ValueT& operator[](HandleT key);

    /**
     * @brief Returns the value associated with the given key or panics
     *        if there is no associated value.
     *
     * Note: since this method panics, if there is no associated value, it
     * cannot be used to insert new values. Use `insert()` if you want to
     * insert new values.
     */
    const ValueT& operator[](HandleT key) const;
};


/**
 * @brief Iterator over keys of an attribute map.
 *
 * This is an interface that has to be implemented by the concrete iterators
 * for the implementors of `AttributeMap`.
 */
template<typename HandleT>
class AttributeMapHandleIterator
{
    static_assert(
        std::is_base_of<BaseHandle<Index>, HandleT>::value,
        "HandleT must inherit from BaseHandle!"
    );

public:
    /// Advances the iterator once. Using the dereference operator afterwards
    /// will yield the next handle.
    virtual AttributeMapHandleIterator& operator++() = 0;
    virtual bool operator==(const AttributeMapHandleIterator& other) const = 0;
    virtual bool operator!=(const AttributeMapHandleIterator& other) const = 0;

    /// Returns the current handle.
    virtual HandleT operator*() const = 0;
    virtual std::unique_ptr<AttributeMapHandleIterator> clone() const = 0;

    virtual ~AttributeMapHandleIterator() = default;
};

/**
 * @brief Simple convinience wrapper for unique_ptr<AttributeMapHandleIterator>
 *
 * The unique_ptr is needed to return an abstract class. This `Ptr` class
 * enables the user to easily use this smart pointer as iterator.
 */
template<typename HandleT>
class AttributeMapHandleIteratorPtr
{
    static_assert(
        std::is_base_of<BaseHandle<Index>, HandleT>::value,
        "HandleT must inherit from BaseHandle!"
    );

public:
    AttributeMapHandleIteratorPtr(std::unique_ptr<AttributeMapHandleIterator<HandleT>> iter)
        : m_iter(std::move(iter)) {}
    AttributeMapHandleIteratorPtr(const AttributeMapHandleIteratorPtr& iteratorPtr)
        : m_iter(iteratorPtr.m_iter->clone()) {}

    AttributeMapHandleIteratorPtr& operator++();
    bool operator==(const AttributeMapHandleIteratorPtr& other) const;
    bool operator!=(const AttributeMapHandleIteratorPtr& other) const;
    HandleT operator*() const;

    virtual ~AttributeMapHandleIteratorPtr() = default;

private:
    std::unique_ptr<AttributeMapHandleIterator<HandleT>> m_iter;
};

} // namespace lvr2

#include "lvr2/attrmaps/AttributeMap.tcc"

#endif /* LVR2_ATTRMAPS_ATTRIBUTEMAP_H_ */
