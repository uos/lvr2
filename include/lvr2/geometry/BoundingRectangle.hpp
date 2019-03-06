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

/**
* @author Kristin Schmidt <kschmidt@uni-osnabrueck.de>
* @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
*/


#ifndef LVR2_GEOMETRY_BOUNDINGRECTANGLE_H_
#define LVR2_GEOMETRY_BOUNDINGRECTANGLE_H_

#include "lvr2/geometry/Normal.hpp"

namespace lvr2
{


/**
 * @struct BoundingRectangle
 * @brief A representation of a bounding rectangle
 *
 * A data class for representing a bounding rectangle that is used for
 * texturizing. Each bounding rectangle is composed of a support vector and a
 * normal that define the rectangles position in 3d space, as well as two
 * vectors that describe the 2d coordinate system for the rectangle. Min dist A
 * and B describe the min distance from the support vector in the 2d coordinate
 * system. The same applies for max dist A and B.
 *
 *
 *                         vec1
 *                  ─────────────────>
 *
 *         minDistA
 *     |----------------|           maxDistA
 *                      |-------------------------------|
 *
 *     ┌────────────────────────────────────────────────┐  ┬
 *     │                                                │  ¦
 *     │                                                │  ¦             │
 *     │                                                │  ¦ minDistB    │
 *     │                                                │  ¦             │ vec2
 *     │                  supportVector                 │  ¦             │
 *     │                ×                               │  ┴ ┬           │
 *     │                                                │    ¦           V
 *     │                                                │    ¦ maxDistB
 *     │                                                │    ¦
 *     └────────────────────────────────────────────────┘    ┴
 *
 *
 */
template<typename CoordType>
struct BoundingRectangle
{
    /// A 3D point, that is the support vector of the rectangle
    BaseVector<CoordType> m_supportVector;
    /// The first direction vector
    BaseVector<CoordType> m_vec1;
    /// The second direction vector (that should be orthogonal to the first direction)
    BaseVector<CoordType> m_vec2;
    /// The normal of the rectangle
    Normal<CoordType> m_normal;
    /// Distance of the beginning border from the support vector in the first direction (if negative: reverse direction)
    CoordType m_minDistA;
    /// Distance of the end border from the support vector in the first direction (if negative: reverse direction)
    CoordType m_maxDistA;
    /// Distance of the beginning border from the support vector in the second direction (if negative: reverse
    /// direction)
    CoordType m_minDistB;
    /// Distance of the end border from the support vector in the second direction (if negative: reverse direction)
    CoordType m_maxDistB;

    /**
     * @brief Constructor
     */
    BoundingRectangle(
        BaseVector<CoordType> supportVector,
        BaseVector<CoordType> vec1,
        BaseVector<CoordType> vec2,
        Normal<CoordType> normal,
        CoordType minDistA,
        CoordType maxDistA,
        CoordType minDistB,
        CoordType maxDistB
    ) :
        m_supportVector(supportVector),
        m_vec1(vec1),
        m_vec2(vec2),
        m_normal(normal),
        m_minDistA(minDistA),
        m_maxDistA(maxDistA),
        m_minDistB(minDistB),
        m_maxDistB(maxDistB)
    {
    }

};

} // namespace lvr2



#endif /* LVR2_GEOMETRY_BOUNDINGRECTANGLE_H_ */
