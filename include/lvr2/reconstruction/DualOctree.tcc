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
 * DualOctree.tcc
 *
 *  Created on: 18.01.2019
 *      Author: Benedikt Schumacher
 */

#include "lvr2/geometry/BaseMesh.hpp"
#include <vector>
using std::vector;
namespace lvr2
{

template<typename BaseVecT, typename BoxT>
DualLeaf<BaseVecT, BoxT>::DualLeaf(BaseVecT vertices[8])
{
    for (unsigned char i = 0; i < 8; i++)
    {
        m_vertices[i] = vertices[i];
    }

    for (unsigned char i = 0; i < 12; i++)
    {
        m_intersections[i] = numeric_limits<uint>::max();
    }
}

template<typename BaseVecT, typename BoxT>
float DualLeaf<BaseVecT, BoxT>::calcIntersection(float x1, float x2, float d1, float d2)
{
    if((d1 < 0 && d2 >= 0) || (d2 < 0 && d1 >= 0))
    {
        return  x2 - d2 * (x1 - x2) / (d1 - d2);
    }
    else
    {
        return  (x2 + x1) / 2.0;
    }
}

template<typename BaseVecT, typename BoxT>
int DualLeaf<BaseVecT, BoxT>::getIndex(float distances[])
{
    int index = 0;
    for(unsigned char i = 0; i < 8; i++)
    {
        if(distances[i] > 0) index |= (1 << i);
    }
    return index;
}

template<typename BaseVecT, typename BoxT>
BaseVecT DualLeaf<BaseVecT, BoxT>::getIntersectionPoint(float intersection, BaseVecT corner_one, BaseVecT corner_two)
{
    BaseVecT intersectionPoint = {intersection, 0, 0};
    BaseVecT rv = {corner_two[0] - corner_one[0], corner_two[1] - corner_one[1], corner_two[2] - corner_one[2]};
    rv.normalize();
    float r = (intersection - corner_one[0]) / rv[0];
    intersectionPoint[1] = corner_one[1] + r * rv[1];
    intersectionPoint[2] = corner_one[2] + r * rv[2];
    return intersectionPoint;
}

template<typename BaseVecT, typename BoxT>
BaseVecT DualLeaf<BaseVecT, BoxT>::getIntersectionPoint(BaseVecT corner_one, float intersection, BaseVecT corner_two)
{
    BaseVecT intersectionPoint = {0, intersection, 0};
    BaseVecT rv = {corner_two[0] - corner_one[0], corner_two[1] - corner_one[1], corner_two[2] - corner_one[2]};
    rv.normalize();
    float r = (intersection - corner_one[1]) / rv[1];
    intersectionPoint[0] = corner_one[0] + r * rv[0];
    intersectionPoint[2] = corner_one[2] + r * rv[2];
    return intersectionPoint;
}

template<typename BaseVecT, typename BoxT>
BaseVecT DualLeaf<BaseVecT, BoxT>::getIntersectionPoint(BaseVecT corner_one, BaseVecT corner_two, float intersection)
{
    BaseVecT intersectionPoint = {0, 0, intersection};
    BaseVecT rv = {corner_two[0] - corner_one[0], corner_two[1] - corner_one[1], corner_two[2] - corner_one[2]};
    rv.normalize();
    float r = (intersection - corner_one[2]) / rv[2];
    intersectionPoint[0] = corner_one[0] + r * rv[0];
    intersectionPoint[1] = corner_one[1] + r * rv[1];
    return intersectionPoint;
}

template<typename BaseVecT, typename BoxT>
void DualLeaf<BaseVecT, BoxT>::getIntersections(
        BaseVecT corners[],
        float distance[],
        BaseVecT positions[])
{
    float intersection;

    intersection = calcIntersection(corners[0][0], corners[1][0], distance[0], distance[1]);
    if(corners[0][1] == corners[1][1] && corners[0][2] == corners[1][2])
    {
        positions[0] = BaseVecT(intersection, corners[0][1], corners[0][2]);
    }
    else
    {
        positions[0] = getIntersectionPoint(intersection, corners[0], corners[1]);
    }

    intersection = calcIntersection(corners[1][1], corners[2][1], distance[1], distance[2]);
    if(corners[1][0] == corners[2][0] && corners[1][2] == corners[2][2])
    {
        positions[1] = BaseVecT(corners[1][0], intersection, corners[1][2]);
    }
    else
    {
        positions[1] = getIntersectionPoint(corners[1], intersection, corners[2]);
    }

    intersection = calcIntersection(corners[3][0], corners[2][0], distance[3], distance[2]);
    if(corners[3][1] == corners[2][1] && corners[3][2] == corners[2][2])
    {
        positions[2] = BaseVecT(intersection, corners[2][1], corners[2][2]);
    }
    else
    {
        positions[2] = getIntersectionPoint(intersection, corners[3], corners[2]);
    }

    intersection = calcIntersection(corners[0][1], corners[3][1], distance[0], distance[3]);
    if(corners[0][0] == corners[3][0] && corners[0][2] == corners[3][2])
    {
        positions[3] = BaseVecT(corners[3][0], intersection, corners[3][2]);
    }
    else
    {
        positions[3] = getIntersectionPoint(corners[0], intersection, corners[3]);
    }

    intersection = calcIntersection(corners[4][0], corners[5][0], distance[4], distance[5]);
    if(corners[4][1] == corners[5][1] && corners[4][2] == corners[5][2])
    {
        positions[4] = BaseVecT(intersection, corners[4][1], corners[4][2]);
    }
    else
    {
        positions[4] = getIntersectionPoint(intersection, corners[4], corners[5]);
    }

    intersection = calcIntersection(corners[5][1], corners[6][1], distance[5], distance[6]);
    if(corners[5][0] == corners[6][0] && corners[5][2] == corners[6][2])
    {
        positions[5] = BaseVecT(corners[5][0], intersection, corners[5][2]);
    }
    else
    {
        positions[5] = getIntersectionPoint(corners[5], intersection, corners[6]);
    }

    intersection = calcIntersection(corners[7][0], corners[6][0], distance[7], distance[6]);
    if(corners[7][1] == corners[6][1] && corners[7][2] == corners[6][2])
    {
        positions[6] = BaseVecT(intersection, corners[6][1], corners[6][2]);
    }
    else
    {
        positions[6] = getIntersectionPoint(intersection, corners[7], corners[6]);
    }

    intersection = calcIntersection(corners[4][1], corners[7][1], distance[4], distance[7]);
    if(corners[4][0] == corners[7][0] && corners[4][2] == corners[7][2])
    {
        positions[7] = BaseVecT(corners[7][0], intersection, corners[7][2]);
    }
    else
    {
        positions[7] = getIntersectionPoint(corners[4], intersection, corners[7]);
    }

    intersection = calcIntersection(corners[0][2], corners[4][2], distance[0], distance[4]);
    if (corners[0][0] == corners[4][0] && corners[0][1] == corners[4][1])
    {
        positions[8] = BaseVecT(corners[0][0], corners[0][1], intersection);
    }
    else
    {
        positions[8] = getIntersectionPoint(corners[0], corners[4], intersection);
    }

    intersection = calcIntersection(corners[1][2], corners[5][2], distance[1], distance[5]);
    if (corners[1][0] == corners[5][0] && corners[1][1] == corners[5][1])
    {
        positions[9] = BaseVecT(corners[1][0], corners[1][1], intersection);
    }
    else
    {
        positions[9] = getIntersectionPoint(corners[1], corners[5], intersection);
    }

    intersection = calcIntersection(corners[3][2], corners[7][2], distance[3], distance[7]);
    if (corners[3][0] == corners[7][0] && corners[3][1] == corners[7][1])
    {
        positions[10] = BaseVecT(corners[3][0], corners[3][1], intersection);
    }
    else
    {
        positions[10] = getIntersectionPoint(corners[3], corners[7], intersection);
    }

    intersection = calcIntersection(corners[2][2], corners[6][2], distance[2], distance[6]);
    if (corners[2][0] == corners[6][0] && corners[2][1] == corners[6][1])
    {
        positions[11] = BaseVecT(corners[2][0], corners[2][1], intersection);
    }
    else
    {
        positions[11] = getIntersectionPoint(corners[2], corners[6], intersection);
    }
}

template<typename BaseVecT, typename BoxT>
void DualLeaf<BaseVecT, BoxT>::getVertices(
        BaseVecT corners[])
{
    for (unsigned char j = 0; j < 8; j++)
    {
        corners[j] = m_vertices[j];
    }
}

template<typename BaseVecT, typename BoxT>
uint DualLeaf<BaseVecT, BoxT>::getIntersection(char i)
{
    return m_intersections[i];
}

template<typename BaseVecT, typename BoxT>
BaseVecT& DualLeaf<BaseVecT, BoxT>::getMiddle()
{
    // return m_middle;
    return 0;
}

template<typename BaseVecT, typename BoxT>
void DualLeaf<BaseVecT, BoxT>::setIntersection(char i, uint value)
{
    m_intersections[i] = value;
}

}
