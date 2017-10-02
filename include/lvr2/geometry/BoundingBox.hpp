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
 * BoundingBox.hpp
 *
 *  @date 22.10.2008
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

#ifndef LVR2_GEOMETRY_BOUNDINGBOX_H_
#define LVR2_GEOMETRY_BOUNDINGBOX_H_

#include <cmath>

#include "Point.hpp"


namespace lvr2
{

/**
 * @brief A dynamic bounding box class.
 */
template<typename BaseVecT>
class BoundingBox
{
public:

    /**
     * @brief Default constructor
     */
    BoundingBox();

    /**
     * @brief Constructs a bounding box with from the given vertices
     *
     * @param v1        Lower left corner of the BoundingBox
     * @param v2        Upper right corner of the BoundingBox
     * @return
     */
    BoundingBox(Point<BaseVecT> v1, Point<BaseVecT> v2);

    /**
     * @brief Expands the bounding box if the given point \ref{v} is
     *        outside the current volume
     *
     * @param v         A 3d point
     */
    inline void expand(Point<BaseVecT> v);

    /**
     * @brief  Calculates the surrounding bounding box of the current
     *         volume and the other given bounding box
     *
     * @param bb        Another bounding box
     */
    inline void expand(const BoundingBox<BaseVecT>& bb);

    /**
     * @brief Returns the radius of the current volume, i.e. the distance
     *        between the centroid and the most distant corner from this
     *        point.
     */
    typename BaseVecT::CoordType getRadius() const;

    /**
     * @brief Returns true if the bounding box has been expanded before or
     *        was initialized with a preset size.
     */
    bool isValid() const;

    /**
     * @brief Returns the center point of the bounding box.
     */
    Point<BaseVecT> getCentroid() const;

    /**
     * @brief Returns the longest side of the bounding box
     */
    typename BaseVecT::CoordType getLongestSide() const;

    /**
     * @brief Returns the x-size of the bounding box
     */
    typename BaseVecT::CoordType getXSize() const;

    /**
     * @brief Returns the y-size of the bounding box
     */
    typename BaseVecT::CoordType getYSize() const;

    /**
     * @brief Returns the z-size of the bounding box
     */
    typename BaseVecT::CoordType getZSize() const;

    /**
     * @brief Returns the upper right coordinates
     */
    Point<BaseVecT> getMax() const;

    /**
     * @brief Returns the lower left coordinates
     */
    Point<BaseVecT> getMin() const;

private:
    /// The lower right point of the bounding box
    Point<BaseVecT> m_min;

    /// The upper right point of the bounding box
    Point<BaseVecT> m_max;

    /// The center point of the bounding box
    Point<BaseVecT> m_centroid;
};

template<typename BaseVecT>
inline std::ostream& operator<<(std::ostream& os, const BoundingBox<BaseVecT>& bb)
{
    os << "Bounding Box[min: " << bb.getMin() << ", max: " <<  bb.getMax();
    os << ", dimension: " << bb.getXSize() << ", " << bb.getYSize() << ", "
       << bb.getZSize() << "]" << endl;
    return os;
}

} // namespace lvr2

#include <lvr2/geometry/BoundingBox.tcc>

#endif /* LVR2_GEOMETRY_BOUNDINGBOX_H_ */