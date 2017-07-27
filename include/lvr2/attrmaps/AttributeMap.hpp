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
 * AttributeMap.hpp
 *
 *  @date 26.07.2017
 */

#ifndef LVR2_ATTRMAPS_ATTRIBUTEMAP_H_
#define LVR2_ATTRMAPS_ATTRIBUTEMAP_H_

#include <boost/optional.hpp>

using boost::optional;

#include <lvr2/util/BaseHandle.hpp>

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
 * There are different implementations of this interface. TODO
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
    virtual bool containsKey(HandleT key) const = 0;
    virtual optional<ValueT> insert(HandleT key, const ValueT& value) = 0;
    virtual optional<ValueT> remove(HandleT key) = 0;
    virtual void clear() = 0;
    virtual optional<ValueT&> get(HandleT key) = 0;
    virtual optional<const ValueT&> get(HandleT key) const = 0;
    virtual size_t numValues() const = 0;

    virtual AttributeMapHandleIteratorPtr<HandleT> begin() const = 0;
    virtual AttributeMapHandleIteratorPtr<HandleT> end() const = 0;

    ValueT& operator[](HandleT key);
    const ValueT& operator[](HandleT key) const;
};

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
};

/// A wrapper for the AttributeMapHandleIterator to save beloved future
/// programmers from dereferencing too much <3
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

    AttributeMapHandleIteratorPtr& operator++();
    bool operator==(const AttributeMapHandleIteratorPtr& other) const;
    bool operator!=(const AttributeMapHandleIteratorPtr& other) const;
    HandleT operator*() const;

private:
    std::unique_ptr<AttributeMapHandleIterator<HandleT>> m_iter;
};

} // namespace lvr2

#include <lvr2/attrmaps/AttributeMap.tcc>

#endif /* LVR2_ATTRMAPS_ATTRIBUTEMAP_H_ */
