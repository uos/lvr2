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
 * MSRMetric.tcc
 *
 *  Created on: Dec 10, 2020
 *      Author: Martin ben Ahmed
 */



 namespace lvr2
 {
    template <typename BaseVecT, typename BoxT>
    MSRMetric<BaseVecT, BoxT>::MSRMetric()
    {
        // do nothing yet
    }

    template <typename BaseVecT, typename BoxT>
    MSRMetric<BaseVecT, BoxT>::~MSRMetric()
    {
        // do nothing yet
    }

    template <typename BaseVecT, typename BoxT>
    const double MSRMetric<BaseVecT, BoxT>::get_distance(PointsetSurfacePtr<BaseVecT> surface, vector<coord<float>*> points,  BaseVecT corners[], DualLeaf<BaseVecT, BoxT> *leaf, bool dual)
    {
        double error_sum;

        // pasted and adapted from DMCReconstruction.tcc

        float distances[8];
        BaseVecT vertex_positions[12];
        float projectedDistance;
        float euklideanDistance;

        for (unsigned char i = 0; i < 8; i++)
        {
            std::tie(projectedDistance, euklideanDistance) = surface->distance((corners[i]));
            distances[i] = projectedDistance;
        }

        leaf->getIntersections(corners, distances, vertex_positions);


        bool distancesValid = true;
        

        // calculate max tolerated distance
        float length = 0;
        if(!dual)
        {
            length = corners[1][0] - corners[0][0];
            length *= 1.7; // <--- why is that?
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
        }

        for(unsigned char a = 0; a < 8; a++)
        {
            if(abs(distances[a]) > length)
            {
                distancesValid = false;
            }
        }

        if(distancesValid)
        {

            vector< vector<BaseVecT> > triangles;
            int index = leaf->getIndex(distances);
            
            

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

                for(uint a = 0; a < error.size(); a++)
                {
                    error[a] /= counter[a];
                    error[a] = sqrt(error[a]);
                    error_sum += error[a];
                }
            }
        }
        return error_sum;
    }

    template<typename BaseVecT, typename BoxT>
    void MSRMetric<BaseVecT, BoxT>::getRotationMatrix(float matrix[9], BaseVecT v1, BaseVecT v2, BaseVecT v3)
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
    void MSRMetric<BaseVecT, BoxT>::matrixDotVector(float* matrix, BaseVecT* vector)
    {
        BaseVecT v = BaseVecT((*vector)[0], (*vector)[1], (*vector)[2]);
        for(unsigned char a = 0; a < 3; a++)
        {
            (*vector)[a] = v[0] * (matrix[a * 3 + 0]) + v[1] * (matrix[a * 3 + 1]) + v[2] * (matrix[a * 3 + 2]);
        }
    }

    template<typename BaseVecT, typename BoxT>
    float MSRMetric<BaseVecT, BoxT>::getDistance(BaseVecT p, BaseVecT v1, BaseVecT v2, BaseVecT v3)
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
    float MSRMetric<BaseVecT, BoxT>::getDistance(BaseVecT p, BaseVecT v1, BaseVecT v2)
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
    float MSRMetric<BaseVecT, BoxT>::getDistance(BaseVecT v1, BaseVecT v2)
    {
        return sqrt( ( v1[0] - v2[0] ) * (v1[0] - v2[0] ) +
                    ( v1[1] - v2[1] ) * (v1[1] - v2[1] ) +
                    ( v1[2] - v2[2] ) * (v1[2] - v2[2] ) );
    }

    template<typename BaseVecT, typename BoxT>
    float MSRMetric<BaseVecT, BoxT>::edgeEquation(BaseVecT p, BaseVecT v1, BaseVecT v2)
    {
        float dx = v2[0] - v1[0];
        float dy = v2[1] - v1[1];
        return (p[0] - v1[0]) * dy - (p[1] - v1[1]) * dx;
    }
}