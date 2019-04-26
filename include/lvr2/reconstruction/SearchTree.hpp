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
*  SearchTree.hpp
*
*       Created on: 02.01.2012
*           Author: Florian Otte, Thomas Wiemann
 */

#ifndef LVR2_RECONSTRUCTION_SEARCHTREE_H_
#define LVR2_RECONSTRUCTION_SEARCHTREE_H_

#include <vector>

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
private:
    using CoordT = typename BaseVecT::CoordType;

public:

    virtual ~SearchTree() = default;

    /**
     * @brief This function performs a k-next-neighbor search on the
              data that was given in the constructor.

     * @param qp          The query point.
     * @param k           The number of neighbours that should be searched.
     * @param indices     A vector that stores the indices for the neighbours
     *                    within the dataset.
     * @param distances   A vector that stores the distances for the neighbours
     *                    that are found.
     * @returns           The number of neighbours found
     */
    virtual int kSearch(
        const BaseVecT& qp,
        int k,
        std::vector<size_t>& indices,
        std::vector<CoordT>& distances
    ) const = 0;

    /**
     * @brief Returns all points within the radius `r` of `qp`.

     * @param qp          The query point.
     * @param r           Radius.
     * @param indices     A vector that will be filled with the indices of
     *                    the points that were found.
     */
    virtual void radiusSearch(
        const BaseVecT& qp,
        CoordT r,
        std::vector<size_t>& indices
    ) const = 0;

    /// Like the other overload, but ignoring the `distances` vector.
    virtual int kSearch(
        const BaseVecT& qp,
        int k,
        std::vector<size_t>& indices
    ) const;

    // /**
    //  * @brief Set the number of neighbours used to estimate and interpolate normals.
    //  */
    // virtual void setKi(int ki);


    // /**
    //  * @brief Set the number of neighbours used for normal estimation
    //  */
    // virtual void setKd(int kd);

    // /**
    //  * @brief Get the number of tangent planes used for distance determination
    //  */
    // virtual int getKi();


    // /**
    //  * @brief Get the number of neighbours used to estimate and interpolate normals.
    //  */
    // virtual int getKd();


protected:
    /// The number of neighbors used for normal interpolation
    int                         m_ki;

    /// The number of tangent planes used for distance determination
    int                         m_kd;
};

template <typename BaseVecT>
using SearchTreePtr = std::shared_ptr<SearchTree<BaseVecT>>;

} // namespace lvr2

#include "SearchTree.tcc"

#endif // LVR2_RECONSTRUCTION_SEARCHTREE_H_
