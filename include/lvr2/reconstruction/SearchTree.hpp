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
*  SearchTree.hpp
*
*       Created on: 02.01.2012
*           Author: Florian Otte, Thomas Wiemann
 */

#ifndef LVR2_RECONSTRUCTION_SEARCHTREE_H_
#define LVR2_RECONSTRUCTION_SEARCHTREE_H_

#include <vector>

using std::vector;


namespace lvr2
{

/**
 * @brief Abstract interface for storing and
 *        searching through a set of points.
 *        Query functions for nearest neighbour searches
 *        are defined.
 */
template< typename BaseVecT>
class SearchTree
{
public:

    /**
     * @brief This function performs a k-next-neighbor search on the
              data that was given in the constructor.

     * @param qp          The query point.
     * @param k           The number of neighbours that should be searched.
     * @param indices     A vector that stores the indices for the neighbours
     *                    within the dataset.
     * @param distances   A vector that stores the distances for the neighbours
     *                    that are found.
     */
    virtual void kSearch(
        const Point<BaseVecT>& qp,
        int k,
        vector<size_t>& indices,
        vector<typename BaseVecT::CoordType>& distances
    ) = 0;

    /**
     * @brief Returns all points within the radius `r` of `qp`.

     * @param qp          The query point.
     * @param r           Radius.
     * @param indices     A vector that will be filled with the indices of
     *                    the points that were found.
     */
    virtual void radiusSearch(
        const Point<BaseVecT>& qp,
        typename BaseVecT::CoordType r,
        vector<size_t>& indices
    ) = 0;

    /// Like the other overload, but ignoring the `distances` vector.
    virtual void kSearch(
        const Point<BaseVecT>& qp,
        int k,
        vector<size_t>& indices
    );

    /**
     * @brief Set the number of neighbours used to estimate and interpolate normals.
     */
    virtual void setKn(int kn);


    /**
     * @brief Set the number of neighbours used to estimate and interpolate normals.
     */
    virtual void setKi(int ki);


    /**
     * @brief Set the number of neighbours used for normal estimation
     */
    virtual void setKd(int kd);


    /**
     * @brief Get the number of neighbours used for normal interpolation
     */
    virtual int getKn();


    /**
     * @brief Get the number of tangent planes used for distance determination
     */
    virtual int getKi();


    /**
     * @brief Get the number of neighbours used to estimate and interpolate normals.
     */
    virtual int getKd();


protected:

    // /// Initialize internal buffers and attribute flags
    // virtual void initBuffers(PointBufferPtr buffer);

    /// The number of neighbors used for initial normal estimation
    int                         m_kn;

    /// The number of neighbors used for normal interpolation
    int                         m_ki;

    /// The number of tangent planes used for distance determination
    int                         m_kd;

    // /// A pointer to the original point cloud data
    // floatArr                    m_pointData;

    // /// A pointer to color attributes for the point (or zero, if the point clouds contains no color values)
    // ucharArr                    m_pointColorData;

    // /// Indicator whether point color values are supported by the search tree instance
    // bool                        m_haveColors;

    // /// Number of points managed by this class
    // size_t                      m_numPoints;
};

} // namespace lvr2

#include "SearchTree.tcc"

#endif // LVR2_RECONSTRUCTION_SEARCHTREE_H_
