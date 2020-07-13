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
 * DMCReconstruction.tcc
 *
 *  Created on: 18.01.2019
 *      Author: Benedikt Schumacher
 */

#include "lvr2/geometry/BaseMesh.hpp"
#include <vector>
#include <random>
using std::vector;
namespace lvr2
{

template<typename BaseVecT, typename BoxT>
DMCReconstruction<BaseVecT, BoxT>::DMCReconstruction(
        PointsetSurfacePtr<BaseVecT> surface,
        BoundingBox<BaseVecT> bb,
        bool dual,
        int maxLevel,
        float maxError) : PointsetMeshGenerator<BaseVecT>(surface)
{
    m_dual = dual;
    m_maxLevel = maxLevel;
    m_maxError = maxError;

    bb_min = bb.getMin();
    bb_max = bb.getMax();

    // increase size of BoundingBox
    // float step = 0.5;
    float step = 0.7;
    // float step = 0.0;
    for(unsigned char a = 0; a < 3; a++)
    {
        bb_min[a] -= (fabs(bb_min[a]) * step);
        bb_max[a] += (fabs(bb_max[a]) * step);
        bb_size[a] = bb_max[a] - bb_min[a];
    }

    // Get all points
    floatArr points_floatArr = this->m_surface->pointBuffer()->getPointArray();
    coord3fArr points = *((coord3fArr*) &points_floatArr);
    vector<coord<float>*> containedPoints;
    for(size_t i = 0; i < this->m_surface->pointBuffer()->numPoints(); i++)
    {
        containedPoints.push_back(&points[i]);
    }

    m_pointHandler = std::unique_ptr<DMCPointHandle<BaseVecT>>(new DMCVecPointHandle<BaseVecT>(containedPoints));

    // Instanciate Octree
    octree = new C_Octree<BaseVecT, BoxT, my_dummy>();
    octree->initialize(m_maxLevel);

    m_leaves = 0;
}

template<typename BaseVecT, typename BoxT>
DMCReconstruction<BaseVecT, BoxT>::~DMCReconstruction()
{
    delete m_progressBar;
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::buildTree(
        C_Octree<BaseVecT, BoxT, my_dummy> &parent,
        int levels,
        bool dual)
{
    m_leaves = 0;
    int cells = 1;
    int max_cells = (1 << m_maxLevel);
    float* max_bb_width = std::max_element(bb_size, bb_size+3);

    for(int cur_Level = levels; cur_Level > 0; --cur_Level)
    {
        // calculating stepwidth at current level for transformation into real world coordinates
        if(cur_Level < levels)
        {
            cells *= 2;
            max_cells /= 2;
        }
        float stepWidth = *max_bb_width / cells;

        CellHandle ch_end = parent.end();

        int cellCounter = 0;

        // visiting all cells of the octree
        for (CellHandle ch = parent.root(); ch != ch_end; ++ch)
        {
            //  visiting cells that matches the current level
            if ((parent.level(ch) == cur_Level))
            {
                cellCounter++;

                // get the points of the current (dual) cell(s)
                vector< vector<coord<float>*> > cellPoints;
                std::vector<CellHandle> cellHandles;
                std::vector<uint> markers;
                if(dual && cur_Level < levels - 2)
                {
                    int cells_tmp = 2;
                    for(int i = 1; i < m_maxLevel; i++)
                    {
                        cells_tmp *= 2;
                    }

                    for(int position = 0; position < 8; position++)
                    {
                        std::vector<CellHandle> cellHandles_tmp;
                        std::vector<uint> markers_tmp;
                        std::tie(cellHandles_tmp, markers_tmp) = parent.all_corner_neighbors(ch, position);

                        cellHandles.insert(cellHandles.end(), cellHandles_tmp.begin(), cellHandles_tmp.end());
                        markers.insert(markers.end(), markers_tmp.begin(), markers_tmp.end());

                        for(int ch_idx = 0; ch_idx < 8; ch_idx++)
                        {
                            vector<coord<float>*> p;
                            cellPoints.push_back(p);
                            vector<coord<float>*> tmp = m_pointHandler->getContainedPoints(cellHandles_tmp[ch_idx].idx());
                            for (vector<coord<float>*>::iterator it = tmp.begin(); it != tmp.end(); it++)
                            {
                                BaseVecT center = parent.cell_center(cellHandles_tmp[ch_idx]);
                                center = center * (*max_bb_width / cells_tmp);
                                center = center + bb_min;
                                if( parent.getChildIndex(center, *it) == (7 - ch_idx) )
                                {
                                    cellPoints[position].push_back(*it);
                                }
                            }
                        }
                    }
                }
                else {
                    vector<coord<float>*> p;
                    p = m_pointHandler->getContainedPoints(ch.idx());
                    cellPoints.push_back(p);
                }

                // check each (dual) cell
                vector<int> splitting_pos;
                float highest_error = 0;
                bool markToSplit = false;
                int idx = 0;

                // iterate over one primal cell or over 8 dual cells until error ist to high
                while (idx < cellPoints.size() && (!markToSplit || dual))
                {
                    // get cell points
                    vector<coord<float>*> points = cellPoints[idx];

                    // when the cell holds points check whether tey fit well to a trinangle
                    if(points.size() > 12)
                    {
                        // get corner vertices of the cell
                        BaseVecT corners[8];

                        int cells_tmp = 2;
                        for(int i = 1; i < m_maxLevel; i++)
                        {
                            cells_tmp *= 2;
                        }

                        // calculation for dual cells
                        if(dual && cur_Level < levels - 2)
                        {
                            for(int i = 0; i < 8; i++)
                            {
                                BaseVecT tmp;
                                detectVertexForDualCell(parent, cellHandles[idx * 8 + i], cells_tmp, *max_bb_width, i, markers[idx * 8 + i], tmp);
                                corners[i] = BaseVecT(tmp[0], tmp[1], tmp[2]);
                            }

                            // swap position of the corners
                            BaseVecT tmp = corners[2];
                            corners[2] = corners[3];
                            corners[3] = tmp;
                            tmp = corners[6];
                            corners[6] = corners[7];
                            corners[7] = tmp;

                            /*for(int i = 0; i < 8; i++)
                            {
                                std::cout << corners[i][0] << "; " << corners[i][1] << "; " << corners[i][2] << std::endl;
                            }
                            std::cout << "-----------" << std::endl;
                            int test = 0;
                            while(test < points.size())
                            {
                                std::cout << (*points[test])[0] << "; " << (*points[test])[1] << "; " << (*points[test])[2] << std::endl;
                                test += 15;
                            }
                            std::cout << "+++++++++++" << std::endl;*/

                        }
                        // calculation for primal cells
                        else
                        {
                            // calculating the real world positions of the corners
                            Location loc = parent.location(ch);
                            int binary_cell_size = 1 << loc.level();
                            corners[0] = BaseVecT(loc.loc_x(),                    loc.loc_y(),                    loc.loc_z());
                            corners[1] = BaseVecT(loc.loc_x() + binary_cell_size, loc.loc_y(),                    loc.loc_z());
                            corners[2] = BaseVecT(loc.loc_x() + binary_cell_size, loc.loc_y() + binary_cell_size, loc.loc_z());
                            corners[3] = BaseVecT(loc.loc_x(),                    loc.loc_y() + binary_cell_size, loc.loc_z());
                            corners[4] = BaseVecT(loc.loc_x(),                    loc.loc_y(),                    loc.loc_z() + binary_cell_size);
                            corners[5] = BaseVecT(loc.loc_x() + binary_cell_size, loc.loc_y(),                    loc.loc_z() + binary_cell_size);
                            corners[6] = BaseVecT(loc.loc_x() + binary_cell_size, loc.loc_y() + binary_cell_size, loc.loc_z() + binary_cell_size);
                            corners[7] = BaseVecT(loc.loc_x(),                    loc.loc_y() + binary_cell_size, loc.loc_z() + binary_cell_size);

                            for(unsigned char a = 0; a < 8; a++)
                            {
                                corners[a] = corners[a] * (*max_bb_width / cells_tmp);
                                corners[a] = corners[a] + bb_min;
                            }
                        }

                        // this is not necessarily a dual leaf
                        DualLeaf<BaseVecT, BoxT> *leaf = new DualLeaf<BaseVecT, BoxT>(corners);

                        // calculate distances
                        float distances[8];
                        BaseVecT vertex_positions[12];
                        float projectedDistance;
                        float euklideanDistance;
                        for (unsigned char i = 0; i < 8; i++)
                        {
                            float projectedDistance;
                            float euklideanDistance;
                            std::tie(projectedDistance, euklideanDistance) = this->m_surface->distance(corners[i]);
                            distances[i] = projectedDistance;
                        }
                        leaf->getIntersections(corners, distances, vertex_positions);

                        /*for(int z = 0; z < 8; z++)
                        {
                            std::cout << distances[z] << std::endl;
                        }
                        std::cout << "-------" << std::endl;*/

                        // check for valid length of the distances
                        bool distancesValid = true;
                        bool d_all_null = false;

                        // calculate max tolerated distance
                        float length = 0;
                        if(!dual)
                        {
                            length = corners[1][0] - corners[0][0];
                            length *= 1.7;
                        }
                        else
                        {
                            for(uint s = 0; s < 12; s++)
                            {
                                BaseVecT vec_tmp = corners[edgeDistanceTable[s][0]] - corners[edgeDistanceTable[s][1]];
                                float float_tmp = sqrt(vec_tmp[0] * vec_tmp[0] + vec_tmp[1] * vec_tmp[1] + vec_tmp[2] * vec_tmp[2]);
                                if(float_tmp > length)
                                {
                                    length = float_tmp;
                                }
                            }
                            // length *= 1.7;
                        }

                        for(unsigned char a = 0; a < 8; a++)
                        {
                            if(abs(distances[a]) > length)
                            {
                                distancesValid = false;
                                /*if(dual && cur_Level < levels - 2)
                                {
                                    markToSplit = false;
                                }*/
                            }
                            else if(distances[a] > 0)
                            {
                                d_all_null = false;
                            }
                        }
                        if(distancesValid)
                        {
                            bool pointsFittingWell = true;

                            vector< vector<BaseVecT> > triangles;
                            int index = leaf->getIndex(distances);
                            /*if(index == 0 || index == 255)
                            {
                                for(int a = 0; a < 8; a++)
                                {
                                    std::cout << corners[a][0] << "; " << corners[a][1] << "; " << corners[a][2] << std::endl;
                                }
                                std::cout << "++++++++++++" << std::endl;
                                for(unsigned char a = 0; a < 8; a++)
                                {
                                    std::cout << distances[a] << std::endl;
                                }
                                std::cout << "------------" << std::endl;
                            }*/
                            if(!d_all_null)
                            {
                                uint edge_index = 0;

                                for(unsigned char a = 0; MCTable[index][a] != -1; a+= 3)
                                {
                                    vector<BaseVecT> triangle_vertices;
                                    for(unsigned char b = 0; b < 3; b++)
                                    {
                                        edge_index = MCTable[index][a + b];
                                        triangle_vertices.push_back(vertex_positions[edge_index]);
                                    }
                                    triangles.push_back(triangle_vertices);
                                }

                                // check, whether the points are fitting well
                                vector<float*> matrices = vector<float*>();

                                // calculate rotation matrix of every triangle
                                for ( uint a = 0; a < triangles.size(); a++ )
                                {
                                    float matrix[9] = { 0 };
                                    BaseVecT v1 = triangles[a][0];
                                    BaseVecT v2 = triangles[a][1];
                                    BaseVecT v3 = triangles[a][2];
                                    getRotationMatrix(matrix, v1, v2, v3);

                                    matrices.push_back(matrix);
                                }

                                vector<float> error(triangles.size(), 0);
                                vector<int> counter(triangles.size(), 0);

                                // for every point check to which trinagle it is the nearest
                                if(triangles.size() > 0)
                                {
                                    for ( uint a = 0; a < points.size(); a++ )
                                    {
                                        signed char min_dist_pos = -1;
                                        float min_dist = -1;

                                        // check which triangle is nearest
                                        for ( uint b = 0; b < triangles.size(); b++ )
                                        {
                                            BaseVecT tmp = {(*points[a])[0] - (triangles[b][0])[0],
                                                            (*points[a])[1] - (triangles[b][0])[1],
                                                            (*points[a])[2] - (triangles[b][0])[2]};

                                            // use rotation matrix for triangle and point
                                            BaseVecT t1 = triangles[b][0] - triangles[b][0];
                                            BaseVecT t2 = triangles[b][1] - triangles[b][0];
                                            BaseVecT t3 = triangles[b][2] - triangles[b][0];
                                            matrixDotVector(matrices[b], &t1);
                                            matrixDotVector(matrices[b], &t2);
                                            matrixDotVector(matrices[b], &t3);
                                            matrixDotVector(matrices[b], &tmp);

                                            // calculate distance from point to triangle
                                            float d = getDistance(tmp, t1, t2, t3);

                                            if( min_dist == -1 )
                                            {
                                                min_dist = d;
                                                min_dist_pos = b;
                                            }
                                            else if( d < min_dist )
                                            {
                                                min_dist = d;
                                                min_dist_pos = b;
                                            }
                                        }

                                        error[min_dist_pos] += (min_dist * min_dist);
                                        counter[min_dist_pos] += 1;
                                    }

                                    uint a = 0;
                                    while(a < error.size() && pointsFittingWell)
                                    {
                                        error[a] /= counter[a];
                                        error[a] = sqrt(error[a]);

                                        if(error[a] > m_maxError)
                                        {
                                            splitting_pos.push_back(idx);
                                            pointsFittingWell = false;
                                        }
                                        a++;
                                    }
                                }
                            }
                            if(MCTable[index][0] == -1 || !pointsFittingWell)
                            // if((!dual && MCTable[index][0] == -1) || cur_Level >= levels - 2 || !pointsFittingWell)
                            {
                                markToSplit = true;
                            }
                        }
                        delete(leaf);
                    }
                    idx++;
                }
                if(markToSplit)
                {
                    vector<coord<float>*> points;
                    CellHandle cellHandle;
                    if(dual && cur_Level < levels - 2)
                    {
                        // get the correct cellHandle depending on the split_positions
                        std::vector<CellHandle> cellHandles;
                        std::vector<uint> markers;
                        std::vector<CellHandle> cellHandles_tmp;
                        std::vector<uint> markers_tmp;

                        sort(splitting_pos.begin(), splitting_pos.end());
                        splitting_pos.erase(unique(splitting_pos.begin(), splitting_pos.end()), splitting_pos.end());

                        for(int j = 0; j < splitting_pos.size(); j++)
                        {
                            std::tie(cellHandles_tmp, markers_tmp) = parent.all_corner_neighbors(ch, splitting_pos[j]);
                            cellHandles.insert(cellHandles.end(), cellHandles_tmp.begin(), cellHandles_tmp.end());
                            markers.insert(markers.end(), markers_tmp.begin(), markers_tmp.end());
                        }

                        // (idx - 1) is the position of the dual cell that violates the splitting condition
                        // std::tie(cellHandles, markers) = parent.all_corner_neighbors(ch, idx - 1);

                        // avoid splitting the same cell twice
                        sort(cellHandles.begin(), cellHandles.end());
                        cellHandles.erase(unique(cellHandles.begin(), cellHandles.end()), cellHandles.end());

                        for(int w = 0; w < cellHandles.size(); w++)
                        {
                            cellHandle = cellHandles[w];
                            points = m_pointHandler->getContainedPoints(cellHandle.idx());

                            BaseVecT cellCenter = parent.cell_center(cellHandle);
                            cellCenter /= max_cells;
                            cellCenter *= stepWidth;
                            cellCenter += bb_min;

                            // sort the points to the corresponding children
                            vector<coord<float>*> childrenPoints[8];
                            for (vector<coord<float>*>::iterator it = points.begin(); it != points.end(); it++)
                            {
                                childrenPoints[parent.getChildIndex(cellCenter, *it)].push_back(*it);
                            }
                            m_pointHandler->split(cellHandle.idx(), childrenPoints, m_dual);

                            // split the cell
                            parent.split( cellHandle );
                            m_leaves += 7;
                        }
                    }
                    else
                    {
                        cellHandle = ch;
                        // points = cellPoints[idx - 1];
                        points = m_pointHandler->getContainedPoints(cellHandle.idx());

                        vector<coord<float>*> childrenPoints[8];
                        BaseVecT cellCenter = parent.cell_center(cellHandle);
                        cellCenter /= max_cells;
                        cellCenter *= stepWidth;
                        cellCenter += bb_min;

                        // sort the points to the corresponding children
                        for (vector<coord<float>*>::iterator it = points.begin(); it != points.end(); it++)
                        {
                            childrenPoints[parent.getChildIndex(cellCenter, *it)].push_back(*it);
                        }
                        m_pointHandler->split(cellHandle.idx(), childrenPoints, m_dual);

                        // split the cell
                        parent.split( cellHandle );
                        m_leaves += 7;
                    }
                }
            // end of visiting cell at current level
            }
        // end of visiting all cells of the octree
        }
        std::cout << cellCounter << " cells at level " << cur_Level << std::endl;
    // end of visiting the current level
    }
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::getRotationMatrix(float matrix[9], BaseVecT v1, BaseVecT v2, BaseVecT v3)
{
    // translate to origin
    BaseVecT vec1 = v1 - v1;
    BaseVecT vec2 = v2 - v1;
    BaseVecT vec3 = v3 - v1;

    // calculate rotaion matrix
    BaseVecT tmp = vec1 - vec2;
    tmp.normalize();
    for(uint a = 0; a < 3; a++)
    {
        matrix[a] = tmp[a];
    }

    tmp[0] = ((vec3[2] - vec1[2]) * matrix[1]) - ((vec3[1] - vec1[1]) * matrix[2]);
    tmp[1] = ((vec3[0] - vec1[0]) * matrix[2]) - ((vec3[2] - vec1[2]) * matrix[0]);
    tmp[2] = ((vec3[1] - vec1[1]) * matrix[0]) - ((vec3[0] - vec1[0]) * matrix[1]);
    tmp.normalize();
    for(uint a = 0; a < 3; a++)
    {
        matrix[a + 6] = tmp[a];
    }

    tmp[0] = matrix[8] * matrix[1] - matrix[7] * matrix[2];
    tmp[1] = matrix[6] * matrix[2] - matrix[8] * matrix[0];
    tmp[2] = matrix[7] * matrix[0] - matrix[6] * matrix[1];
    tmp.normalize();
    for(uint a = 0; a < 3; a++)
    {
        matrix[a + 3] = tmp[a];
    }
}

template<typename BaseVecT, typename BoxT>
float DMCReconstruction<BaseVecT, BoxT>::getDistance(BaseVecT p, BaseVecT v1, BaseVecT v2, BaseVecT v3)
{
    bool flipped = false;
    float dist = 0;

    // check whether the direction of the vertices is correct for further operations
    if( !(edgeEquation(v3, v1, v2) < 0) )
    {
        BaseVecT tmp = v2;
        v2 = v3;
        v3 = tmp;
        flipped = true;
    }

    float v1_v2 = edgeEquation(p, v1, v2);
    float v2_v3 = edgeEquation(p, v2, v3);
    float v3_v1 = edgeEquation(p, v3, v1);

    if ( v1_v2 == 0 || v2_v3 == 0 || v3_v1 == 0 )
    {
        // p lies on an edge
        dist = p[2];
    }
    else if ( v1_v2 < 0 && v2_v3 < 0 && v3_v1 < 0 )
    {
        // p lies in the triangle
        dist = p[2];
    }
    else if ( v1_v2 < 0 && v2_v3 < 0 && v3_v1 > 0 )
    {
        // p is nearest to v3_v1
        dist = getDistance(p, v3, v1);
    }
    else if ( v1_v2 < 0 && v2_v3 > 0 && v3_v1 < 0 )
    {
        // p is nearest to v2_v3
        dist = getDistance(p, v2, v3);
    }
    else if ( v1_v2 < 0 && v2_v3 > 0 && v3_v1 > 0 )
    {
        // p is nearest to v3
        dist = getDistance(v3, p);
    }
    else if ( v1_v2 > 0 && v2_v3 < 0 && v3_v1 < 0 )
    {
        // p is nearest to v1_v2
        dist = getDistance(p, v1, v2);
    }
    else if ( v1_v2 > 0 && v2_v3 < 0 && v3_v1 > 0 )
    {
        // p is nearest to v1
        dist = getDistance(v1, p);
    }
    else if ( v1_v2 > 0 && v2_v3 > 0 && v3_v1 < 0 )
    {
        // p is nearest to v2
        dist = getDistance(v2, p);
    }
    else if ( v1_v2 > 0 && v2_v3 > 0 && v3_v1 > 0 )
    {
        // impossible to reach
    }

    if ( flipped )
    {
        BaseVecT tmp = v2;
        v2 = v3;
        v3 = tmp;
    }
    return dist;
}

template<typename BaseVecT, typename BoxT>
float DMCReconstruction<BaseVecT, BoxT>::getDistance(BaseVecT p, BaseVecT v1, BaseVecT v2)
{
    BaseVecT normal = v2 - v1;
    normal[2] = normal[0];
    normal[0] = normal[1];
    normal[1] = -1 * normal[2];
    normal[2] = 0.0;

    float v1_v12 = edgeEquation(p, v1, v1 + normal);
    float v2_v22 = edgeEquation(p, v2, v2 + normal);

    if ( v1_v12 < 0 && v2_v22 > 0 )
    {
        BaseVecT d = ( v2 - v1 ) / getDistance(v2, v1);
        BaseVecT v = p - v1;
        float t = v.dot(d);
        BaseVecT projection = v1 + ( d * t );
        return getDistance(projection, p);
    }
    else if ( v1_v12 > 0 && v2_v22 > 0 )
    {
        // v1 is nearest point
        return getDistance(v1, p);
    }
    else if ( v1_v12 < 0 && v2_v22 < 0 )
    {
        // v2 is nearest point
        return getDistance(v2, p);
    }

    return 0;
}

template<typename BaseVecT, typename BoxT>
float DMCReconstruction<BaseVecT, BoxT>::getDistance(BaseVecT v1, BaseVecT v2)
{
    return sqrt( ( v1[0] - v2[0] ) * (v1[0] - v2[0] ) +
                 ( v1[1] - v2[1] ) * (v1[1] - v2[1] ) +
                 ( v1[2] - v2[2] ) * (v1[2] - v2[2] ) );
}

template<typename BaseVecT, typename BoxT>
float DMCReconstruction<BaseVecT, BoxT>::edgeEquation(BaseVecT p, BaseVecT v1, BaseVecT v2)
{
    float dx = v2[0] - v1[0];
    float dy = v2[1] - v1[1];
    return (p[0] - v1[0]) * dy - (p[1] - v1[1]) * dx;
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::matrixDotVector(float* matrix, BaseVecT* vector)
{
    BaseVecT v = BaseVecT((*vector)[0], (*vector)[1], (*vector)[2]);
    for(unsigned char a = 0; a < 3; a++)
    {
        (*vector)[a] = v[0] * (matrix[a * 3 + 0]) + v[1] * (matrix[a * 3 + 1]) + v[2] * (matrix[a * 3 + 2]);
    }
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::getMesh(BaseMesh<BaseVecT> &mesh)
{
    // start building adaptive octree
    string comment = timestamp.getElapsedTime() + "Creating Octree...";
    cout << comment << endl;
    buildTree(*octree, m_maxLevel, m_dual);

    comment = timestamp.getElapsedTime() + "Cleaning up RAM...";
    cout << comment << endl;
    m_pointHandler->clear();

    comment = timestamp.getElapsedTime() + "Creating Mesh ";
    m_progressBar = new ProgressBar(m_leaves, comment);
    traverseTree(mesh, *octree);
    delete(octree);
    cout << endl;
}

template<typename BaseVecT, typename BoxT>
DualLeaf<BaseVecT, BoxT>* DMCReconstruction<BaseVecT, BoxT>::getDualLeaf(
        CellHandle &cell,
        int cells,
        C_Octree<BaseVecT, BoxT, my_dummy> &octree,
        char pos)
{
    // start building dual Leaf
    BaseVecT corners[8];
    float* max_bb_width = std::max_element(bb_size, bb_size+3);

    bool outside = false;

    // get all neighbors
    std::vector<CellHandle> cellHandles;
    std::vector<uint> markers;
    std::tie(cellHandles, markers) = octree.all_corner_neighbors(cell, pos);

    if(cellHandles.size() != 8)
    {
        std::cout << "ERROR - CellHandles Size is " << cellHandles.size() << " but must be 8!"<< std::endl;
    }

    // find vertex of each cell
    for(uint i = 0; i < 8; i++)
    {
        BaseVecT tmp;
        detectVertexForDualCell(octree, cellHandles[i], cells, *max_bb_width, i, markers[i], tmp);
        corners[i] = tmp;
    }

    // resort the vertices to needed coordinate system
    //        6------7            7------6
    //       /|     /|           /|     /|
    //      2------3 |          3------2 |
    // FROM | 4----|-5  ==TO==> | 4----|-5
    //      |/     |/           |/     |/
    //      0------1            0------1
    //
    BaseVecT tmp = corners[2];
    corners[2] = corners[3];
    corners[3] = tmp;
    tmp = corners[6];
    corners[6] = corners[7];
    corners[7] = tmp;

    // generate dual leaf
    return new DualLeaf<BaseVecT, BoxT>(corners);

}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::traverseTree(
        BaseMesh<BaseVecT> &mesh,
        C_Octree<BaseVecT, BoxT, my_dummy> &octree)
{
    CellHandle ch_end = octree.end();
    int cells = 2;
    for(int i = 1; i < m_maxLevel; i++)
    {
        cells *= 2;
    }

    for (CellHandle ch = octree.root(); ch != ch_end; ++ch)
    {
        if (octree.is_leaf(ch))
        {
            for(unsigned char c = 0; c < 8; c++)
            {
                DualLeaf<BaseVecT, BoxT> *dualLeaf = getDualLeaf(ch, cells, octree, c);

                getSurface(mesh, dualLeaf, cells, (short)octree.level(ch));

                // free memory
                delete dualLeaf;
            }
            ++(*m_progressBar);
        }
    }
    return;
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::detectVertexForDualCell(
        C_Octree<BaseVecT, BoxT, my_dummy> &octree,
        CellHandle ch,
        int cells,
        float max_bb_width,
        uint pos,
        int onEdge,
        BaseVecT &feature)
{
    feature = octree.cell_center(ch);

    BaseVecT cellCorner = octree.cell_corner(ch, pos);

    if(onEdge & 1 && cellCorner[0] == 0.0)
    {
        feature[0] = 0.0;
    }
    else if(onEdge & 1 && cellCorner[0] == cells)
    {
        feature[0] = cellCorner[0];
    }

    if(onEdge & 2 && cellCorner[1] == 0.0)
    {
        feature[1] = 0.0;
    }
    else if(onEdge & 2 && cellCorner[1] == cells)
    {
        feature[1] = cellCorner[1];
    }

    if(onEdge & 4 && cellCorner[2] == 0.0)
    {
        feature[2] = 0.0;
    }
    else if(onEdge & 4 && cellCorner[2] == cells)
    {
        feature[2] = cellCorner[2];
    }

    for(unsigned char i = 0; i < 3; i++)
    {
        feature[i] = feature[i] * (max_bb_width / cells);
        feature[i] = feature[i] + (bb_min[i]);
    }
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::getSurface(
        BaseMesh<BaseVecT> &mesh,
        DualLeaf<BaseVecT, BoxT> *leaf,
        int cells,
        short level)
{
    BaseVecT edges[8];
    float distances[8];
    BaseVecT vertex_positions[12];
    float projectedDistance;
    float euklideanDistance;
    HalfEdgeMesh<BaseVecT> *meshCast;

    leaf->getVertices(edges);

    /*for(uint a = 0; a < 8; a++)
    {
        dualVertices.push_back(edges[a]);
    }*/

    for (unsigned char i = 0; i < 8; i++)
    {
        // get distances from kd tree
        float projectedDistance;
        float euklideanDistance;
        std::tie(projectedDistance, euklideanDistance) = this->m_surface->distance(edges[i]);
        distances[i] = projectedDistance;
    }

    // mark edges with intersections
    leaf->getIntersections(edges, distances, vertex_positions);

    // check whether the distances are close enough or not
    float length = 0;
    if(!m_dual)
    {
        length = edges[1][0] - edges[0][0];
        length *= 1.7;
    }
    else
    {
        for(uint s = 0; s < 12; s++)
        {
            BaseVecT vec_tmp = edges[edgeDistanceTable[s][0]] - edges[edgeDistanceTable[s][1]];
            float float_tmp = sqrt(vec_tmp[0] * vec_tmp[0] + vec_tmp[1] * vec_tmp[1] + vec_tmp[2] * vec_tmp[2]);
            if(float_tmp > length)
            {
                length = float_tmp;
            }
        }
        // length *= 3.4;
    }

    /*for(unsigned char a = 0; a < 8; a++)
    {
        if(distances[a] > length)
        {
            return;
        }
    }*/

    // index is for mc-table
    int index = leaf->getIndex(distances);
    uint edge_index = 0;

    for(unsigned char a = 0; MCTable[index][a] != -1; a+= 3)
    {
        vector<VertexHandle> triangle_vertices;
        for(unsigned char b = 0; b < 3; b++)
        {
            edge_index = MCTable[index][a + b];
            BaseVecT v = vertex_positions[edge_index];
            // VertexHandle vh = mesh.addVertex(v, level);
            VertexHandle vh = mesh.addVertex(v);
            triangle_vertices.push_back(vh);
        }
        mesh.addFace(triangle_vertices[0], triangle_vertices[1], triangle_vertices[2]);
    }
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::getMesh(
    BaseMesh<BaseVecT>& mesh,
    BoundingBox<BaseVecT>& bb,
    vector<unsigned int>& duplicates,
    float comparePrecision
)
{
//    // Status message for mesh generation
//    string comment = timestamp.getElapsedTime() + "Creating Mesh ";
//    ProgressBar progress(m_grid->getNumberOfCells(), comment);

//    // Some pointers
//    BoxT* b;
//    unsigned int global_index = mesh.numVertices();

//    // Iterate through cells and calculate local approximations
//    typename HashGrid<BaseVecT, BoxT>::box_map_it it;
//    for(it = m_grid->firstCell(); it != m_grid->lastCell(); it++)
//    {
//        b = it->second;
//        b->getSurface(mesh, m_grid->getQueryPoints(), global_index, bb, duplicates, comparePrecision);
//        if(!timestamp.isQuiet())
//            ++progress;
//    }

//    if(!timestamp.isQuiet())
//        cout << endl;
}

} // namespace lvr2
