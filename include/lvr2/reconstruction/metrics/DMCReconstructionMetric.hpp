/**
 * Copyright (c) 2020, University Osnabrück
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
 * MSRReconstructionMetric.hpp
 *
 *  Created on: Dec 11, 2020
 *      Author: Martin ben Ahmed
 */

#ifndef DMCRECONSTRUCTIONMETRIC_HPP
#define DMCRECONSTRUCTIONMETRIC_HPP

#include "../DualOctree.hpp"


namespace lvr2
{

/**
*@brief Interface for DMC Reconstruction Metrics
*/

template <typename BaseVecT, typename BoxT>
class DMCReconstructionMetric
{

public:

    /**
     * @brief Constructor
     */
    DMCReconstructionMetric() = default;

    /**
     * @brief Destructor
     */
    ~DMCReconstructionMetric() = default;

    /**
     * @brief Calculates the distance between a surface and points
     * @param surface 
     * @param points 
     * @param corners
     * @param leaf
     * @param dual
     * @return the msr distances between the given surface and the points
     */
    virtual const double get_distance(PointsetSurfacePtr<BaseVecT> surface, vector<coord<float>*> points, BaseVecT corners[], DualLeaf<BaseVecT, BoxT> *leaf, bool dual) = 0;

};

} // Namespace lvr2

#endif // DMCRECONSTRUCTIONMETRIC_HPP