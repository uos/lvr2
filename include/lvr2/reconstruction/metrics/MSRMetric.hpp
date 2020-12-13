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
 * MSRMetric.hpp
 *
 *  Created on: Dec 10, 2020
 *      Author: Martin ben Ahmed
 */

#ifndef MSRMETRIC_HPP
#define MSRMETRIC_HPP

#include "DMCReconstructionMetric.hpp"
#include "../DualOctree.hpp"

namespace lvr2
{


/**
*@brief original implementation of adapted to DMCReconstructionMetric interface
*/

template <typename BaseVecT, typename BoxT>
class MSRMetric : public DMCReconstructionMetric<BaseVecT, BoxT>
{

public:
    /**
     * @brief Constructor
     */
    MSRMetric();

    /**
     * @brief destructor 
     */
    ~MSRMetric();

    /**
     * @brief Calculates the distance between a surface and points
     * @param surface 
     * @param points 
     * @param corners
     * @param leaf
     * @param dual
     * @return the msr distances between the given surface and the points
     */
    const double get_distance(PointsetSurfacePtr<BaseVecT> surface, vector<coord<float>*> points, BaseVecT corners[], DualLeaf<BaseVecT, BoxT> *leaf, bool dual);

private:
    /**
     * @brief Calculates a rotation matrix for a triangle that rotates it into xy
     *
     * @param matrix Array the matrix should be written in
     * @param v1 First point of the triangle
     * @param v2 Second point of the triangle
     * @param v3 Third point of the triangle
     */
    void getRotationMatrix(float matrix[9], BaseVecT v1, BaseVecT v2, BaseVecT v3);

    /**
     * @brief Performs a matrix multiplication
     *
     * @param matrix Pointer to the matrix
     * @param vector Pointer to th vector
     */
    void matrixDotVector(float* matrix, BaseVecT* vector);

    /**
     * @brief Calculates the distance between a point and a triangle
     *
     * @param p Vertex to calculate distance for
     * @param v1 First point of the triangle
     * @param v2 Second point of the triangle
     * @param v3 Third point of the triangle
     */
    float getDistance(BaseVecT p,
        BaseVecT v1,
        BaseVecT v2,
        BaseVecT v3);
    
    /**
     * @brief Calculates the distance between a point and a line
     *
     * @param p Vertex to calculate distance for
     * @param v1 First point of the line
     * @param v2 Second point of the line
     */
    float getDistance(BaseVecT p,
        BaseVecT v1,
        BaseVecT v2);

    /**
     * @brief Calculates the distance between to points
     *
     * @param p Vertex to calculate distance for
     * @param v1 Point
     */
    float getDistance(BaseVecT v1,
        BaseVecT v2);
    
    /**
     * @brief Calculates whether the given vertex lies left, right or on the given line
     *
     * @param p Vertex to check position for
     * @param v1 First point of the line
     * @param v2 Second point of the line
     */
    float edgeEquation(BaseVecT p,
        BaseVecT v1,
        BaseVecT v2);


};


} // Namespace lvr2

#include "MSRMetric.tcc"

#endif // MSRMETRIC_HPP