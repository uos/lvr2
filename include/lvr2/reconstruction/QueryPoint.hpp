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
 * QueryPoint.h
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef _LVR2_RECONSTRUCTION_QueryPoint_H_
#define _LVR2_RECONSTRUCTION_QueryPoint_H_

namespace lvr2
{

/**
 * @brief A query Vector for marching cubes reconstructions.
 *        It represents a Vector in space together with a
 *        'distance' value that is used by the marching
 *        cubes algorithm
 */
template<typename BaseVecT>
class QueryPoint
{
public:

    /**
     * @brief Default constructor.
     */
    QueryPoint();

    /**
     * @brief Constructor.
     *
     * @param p          The position of the query Vector. The distance
     *                   value is set to 0
     */
    QueryPoint(const BaseVecT& p);

    /**
     * @brief Constructor.
     *
     * @param p         The position of the query Vector.
     * @param f         The distance value for the query Vector.
     */
    QueryPoint(const BaseVecT& p, float f);

    /**
     * @brief Copy constructor.
     * @param o
     * @return
     */
    QueryPoint(const QueryPoint &o);

    /**
     * @brief Destructor.
     */
    virtual ~QueryPoint() {};

    /// The position of the query Vector
    BaseVecT m_position;

    /// The associated distance value
    float           m_distance;

    /// Indicates if the query Vector is valid
    bool            m_invalid;
};

} // namespace lvr2

#include "QueryPoint.tcc"

#endif /* _LVR2_RECONSTRUCTION_QueryPoint_H_ */
