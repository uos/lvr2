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
 * @brief A vector, which preserves its indices even when an element is deleted
 *
 * This is basically a wrapper for the std::vector, which marks an element as deleted but does not delete it.
 *
 * USE WITH CAUTION: This NEVER deletes values and can get very large!
 *
 * @tparam ElemT Type of elements in the vector
 * @tparam HandleT Type of the index for the vector
 */
template<typename ElemT, typename HandleT>
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

    /// Add the given element
    void push_back(const ElementType& elem);

    /**
     * @brief Mark the element behind the given handle as deleted
     *
     * This does NOT call the DESTRUCTOR of the marked element!
     */
    void erase(const HandleType& handle);

    /// Request the element behind the given handle
    ElemT& operator[](const HandleType& handle);

    /// Request the element behind the given handle
    const ElemT& operator[](const HandleType& handle) const;

    /// Absolute size of the vector (with delete-marked elements)
    size_t size() const;

    /// Number of not delete-marked elements
    size_t sizeUsed() const;
};

} // namespace lvr2

#include <lvr2/util/StableVector.tcc>

#endif /* LVR2_UTIL_STABLEVECTOR_H_ */
