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
*
* @author Kristin Schmidt <kschmidt@uni-osnabrueck.de>
* @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
*/


#ifndef LVR2_GEOMETRY_BOUNDINGRECTANGLE_H_
#define LVR2_GEOMETRY_BOUNDINGRECTANGLE_H_


namespace lvr2
{


/**
 * A data class for representing a bounding rectangle that is used for
 * texturizing. Each bounding rectangle is composed of a support vector and a
 * normal that define the rectangles position in 3d space, as well as two
 * vectors that describe the 2d coordinate system for the rectangle. Min dist A
 * and B describe the min distance from the support vector in the 2d coordinate
 * system. The same applies for max dist A and B.
 */
template<typename BaseVecT>
struct BoundingRectangle
{
    Vector<BaseVecT> m_supportVector;
    Vector<BaseVecT> m_vec1;
    Vector<BaseVecT> m_vec2;
    Normal<BaseVecT> m_normal;
    float m_minDistA;
    float m_maxDistA;
    float m_minDistB;
    float m_maxDistB;

    BoundingRectangle(
        Vector<BaseVecT> supportVector,
        Vector<BaseVecT> vec1,
        Vector<BaseVecT> vec2,
        Normal<BaseVecT> normal,
        float minDistA,
        float maxDistA,
        float minDistB,
        float maxDistB
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
