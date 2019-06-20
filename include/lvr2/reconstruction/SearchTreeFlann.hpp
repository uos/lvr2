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
 * SearchTreeFlann.hpp
 *
 *  Created on: Sep 22, 2015
 *      Author: Thomas Wiemann
 */

#ifndef LVR2_RECONSTRUCTION_SEARCHTREEFLANN_HPP_
#define LVR2_RECONSTRUCTION_SEARCHTREEFLANN_HPP_

#include <vector>
#include <memory>

#include <flann/flann.hpp>

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/reconstruction/SearchTree.hpp"

using std::vector;
using std::unique_ptr;

namespace lvr2
{

/**
 * @brief SearchClass for point data.
 *
 *      This class uses the FLANN ( http://www.cs.ubc.ca/~mariusm/uploads/FLANN )
 *      library to implement a nearest neighbour search for point-data.
 */
template<typename BaseVecT>
class SearchTreeFlann : public SearchTree<BaseVecT>
{
private:
    using CoordT = typename BaseVecT::CoordType;

public:

    // typedef boost::shared_ptr< SearchTreeFlann< VertexT> > Ptr;


    /**
     *  @brief Takes the point-data and initializes the underlying searchtree.
     *
     *  @param buffer  A PointBuffer point that holds the data.
     */
    SearchTreeFlann(PointBufferPtr buffer);

    /// See interface documentation.
    virtual void kSearch(
        const BaseVecT& qp,
        int k,
        vector<size_t>& indices,
        vector<CoordT>& distances
    ) const;

    /// See interface documentation.
    virtual void radiusSearch(
        const BaseVecT& qp,
        CoordT r,
        vector<size_t>& indices
    ) const;

protected:

    /// The FLANN search tree structure.
    unique_ptr<flann::Index<flann::L2_Simple<float>>> m_tree;

    // /// FLANN matrix representation of the points
    // flann::Matrix<float>                                            m_flannPoints;

    // vector<int>                                                     m_ind;
    // vector<float>                                                   m_dst;



};

} // namespace lvr2

#include "lvr2/reconstruction/SearchTreeFlann.tcc"

#endif /* LVR2_RECONSTRUCTION_SEARCHTREEFLANN_HPP_ */
