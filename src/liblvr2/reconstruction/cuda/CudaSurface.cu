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
 * CudaSurface.cu
 *
 * @author Alexander Mock
 */

#include "lvr2/reconstruction/cuda/CudaSurface.hpp"
#include "lvr2/config/lvropenmp.hpp"

namespace lvr2
{

/// Define Kernels

__global__ void KNNKernel1(const LBPointArray<float> D_V,
        const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits ,
        LBPointArray<float> D_Normals, int k, int method,
        float flip_x, float flip_y, float flip_z);

__global__ void KNNKernel2(const LBPointArray<float> D_V,
        const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits ,
        LBPointArray<float> D_Normals, int k, int method,
        float flip_x, float flip_y, float flip_z);

// IN WORK
__global__ void InterpolationKernel(const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits,
        LBPointArray<float> D_Normals, int ki );

// IN WORK
__global__ void GridDistanceKernel(const LBPointArray<float> D_V,const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits,
         const LBPointArray<float> D_Normals, QueryPointC* D_Query_Points, const unsigned int qp_size, int k);


// Get a matrix element
__device__ unsigned int GetKdTreePosition(const LBPointArray<float>& D_kd_tree_values, const LBPointArray<unsigned char>& D_kd_tree_splits, float x, float y, float z)
{
    unsigned int pos = 0;
    unsigned int current_dim = 0;

    while(pos < D_kd_tree_splits.width)
    {
        current_dim = static_cast<unsigned int>(D_kd_tree_splits.elements[pos]);
        if(current_dim == 0)
        {
            if(x <= D_kd_tree_values.elements[pos] )
            {
                pos = pos*2+1;
            } else {
                pos = pos*2+2;
            }
        } else if(current_dim == 1) {
            if(y <= D_kd_tree_values.elements[pos] ){
                pos = pos*2+1;
            }else{
                pos = pos*2+2;
            }
        } else {
            if(z <= D_kd_tree_values.elements[pos] ){
                pos = pos*2+1;
            }else{
                pos = pos*2+2;
            }
        }
    }

    return pos;
}

__device__ float SearchQueryPoint(const LBPointArray<float>& D_kd_tree_values, const LBPointArray<unsigned char>& D_kd_tree_splits, float x, float y, float z)
{
    return D_kd_tree_values.elements[GetKdTreePosition(D_kd_tree_values, D_kd_tree_splits, x, y, z)];
}


__global__ void KNNKernel1(const LBPointArray<float> D_V,
        const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits ,
        LBPointArray<float> D_Normals, int k, int method,
        float flip_x, float flip_y, float flip_z)
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < D_V.width){

        // instant leaf!
        unsigned int pos = tid + D_kd_tree_splits.width;

        unsigned int vertex_index = static_cast<unsigned int>(D_kd_tree_values.elements[pos]+ 0.5);

        float vertex_x = D_V.elements[ vertex_index * 3 + 0 ];
        float vertex_y = D_V.elements[ vertex_index * 3 + 1 ];
        float vertex_z = D_V.elements[ vertex_index * 3 + 2 ];

        // start and end defined

        float result_x = 0.0;
        float result_y = 0.0;
        float result_z = 0.0;

        unsigned int subtree_pos = pos;
        unsigned int i;
        for(i=1; i<(k+1) && subtree_pos>0; i*=2) {
                subtree_pos = static_cast<unsigned int>((subtree_pos  - 1) / 2);
        }



        // PCA STUFF INIT

        //x
        float xx = 0.0;
        float xy = 0.0;
        float xz = 0.0;

        //y
        float yy = 0.0;
        float yz = 0.0;

        //z
        float zz = 0.0;



        unsigned int iterator = subtree_pos;
        unsigned int max_nodes = 1;
        bool leaf_reached = false;
        unsigned int i_nn = 0;

        // like width search
        // go kd-tree up until max_nodes(leaf_nodes of subtree) bigger than needed nodes k
        // iterator = iterator * 2 + 1 -> go to
        const unsigned int num_values = D_kd_tree_values.width;
        for( ; iterator < num_values; iterator = iterator * 2 + 1, max_nodes *= 2)
        {
        // collect nodes from current height
            for( unsigned int i=0; i < max_nodes && iterator + i < num_values; i++)
            {
                unsigned int current_pos = iterator + i;
                unsigned int leaf_value  = static_cast<unsigned int>(D_kd_tree_values.elements[ current_pos ] + 0.5 );

                if( leaf_reached && i_nn <= k )
                {
                    if( leaf_value != vertex_index && leaf_value < D_V.width )
                    {
                        unsigned int curr_nn_index = leaf_value * 3;

                        float rx = D_V.elements[ curr_nn_index ] - vertex_x;
                        float ry = D_V.elements[ curr_nn_index + 1 ] - vertex_y;
                        float rz = D_V.elements[ curr_nn_index + 2 ] - vertex_z;

                        if(rx != 0.0 || ry != 0.0 || rz != 0.0)
                        {
                            // ilikebigbits.com/blog/2015/3/2/plane-from-points
                            xx += rx * rx;
                            xy += rx * ry;
                            xz += rx * rz;
                            yy += ry * ry;
                            yz += ry * rz;
                            zz += rz * rz;
                            i_nn ++;
                        }
                    }
                } else if( current_pos * 2 + 1 >= num_values ) {

                    unsigned int curr_nn_index = leaf_value * 3;
                    //first leaf reached
                    leaf_reached = true;
                    if( leaf_value != vertex_index && i_nn <= k*3 )
                    {
                        float rx    = D_V.elements[ curr_nn_index ] - vertex_x;
                        float ry     = D_V.elements[ curr_nn_index + 1 ] - vertex_y;
                        float rz     = D_V.elements[ curr_nn_index + 2] - vertex_z;

                        // instant PCA!
                        xx += rx * rx;
                        xy += rx * ry;
                        xz += rx * rz;
                        yy += ry * ry;
                        yz += ry * rz;
                        zz += rz * rz;

                        i_nn ++;
                    }
                }
            }
        }

        //determinante?
        float det_x = yy * zz - yz * yz;
        float det_y = xx * zz - xz * xz;
        float det_z = xx * yy - xy * xy;

        float dir_x;
        float dir_y;
        float dir_z;
        // det X biggest
        if( det_x >= det_y && det_x >= det_z){

            if(det_x <= 0.0){
                //not a plane
            }

            dir_x = 1.0;
            dir_y = (xz * yz - xy * zz) / det_x;
            dir_z = (xy * yz - xz * yy) / det_x;
        } //det Y biggest
        else if( det_y >= det_x && det_y >= det_z){

            if(det_y <= 0.0){
                // not a plane
            }

            dir_x = (yz * xz - xy * zz) / det_y;
            dir_y = 1.0;
            dir_z = (xy * xz - yz * xx) / det_y;
        } // det Z biggest
        else{
            if(det_z <= 0.0){
                // not a plane
            }

            dir_x = (yz * xy - xz * yy ) / det_z;
            dir_y = (xz * xy - yz * xx ) / det_z;
            dir_z = 1.0;
        }

        float invnorm = 1/sqrtf( dir_x * dir_x + dir_y * dir_y + dir_z * dir_z );

        result_x = dir_x * invnorm;
        result_y = dir_y * invnorm;
        result_z = dir_z * invnorm;



        // FLIP NORMALS
        float x_dir = flip_x - vertex_x;
        float y_dir = flip_y - vertex_y;
        float z_dir = flip_z - vertex_z;

        float scalar = x_dir * result_x + y_dir * result_y + z_dir * result_z;

        // gegebenfalls < durch > ersetzen
        if(scalar < 0)
        {
            result_x = -result_x;
            result_y = -result_y;
            result_z = -result_z;
        }

        D_Normals.elements[vertex_index * D_Normals.dim ] = result_x;
        D_Normals.elements[vertex_index * D_Normals.dim + 1 ] = result_y;
        D_Normals.elements[vertex_index * D_Normals.dim + 2 ] = result_z;

    }

}

__global__ void KNNKernel2(const LBPointArray<float> D_V,
        const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits ,
        LBPointArray<float> D_Normals, int k, int method,
        float flip_x, float flip_y, float flip_z)
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // instant leaf!
    //unsigned int pos = tid + D_kd_tree_splits.width;

    if(tid < D_V.width )
    {
        unsigned int pos = GetKdTreePosition(D_kd_tree_values, D_kd_tree_splits, D_V.elements[tid * D_V.dim], D_V.elements[tid * D_V.dim + 1], D_V.elements[tid * D_V.dim +2] );

        // instant leaf!
        unsigned int vertex_index = static_cast<unsigned int>(D_kd_tree_values.elements[pos]+ 0.5);



        if(vertex_index < D_V.width)
        {

            float vertex_x = D_V.elements[ vertex_index * 3 + 0 ];
            float vertex_y = D_V.elements[ vertex_index * 3 + 1 ];
            float vertex_z = D_V.elements[ vertex_index * 3 + 2 ];

            unsigned int nearest_index;

            int start = pos-(k/2);
            int end = pos+((k+1)/2);

            int correct = 0;

            if(start < D_kd_tree_splits.width)
            {
                correct = D_kd_tree_splits.width - start;
            }else if(end > D_kd_tree_values.width)
            {
                correct = D_kd_tree_values.width - end;
            }

            start += correct;
            end += correct;

            // start and end defined

            float result_x = 0.0;
            float result_y = 0.0;
            float result_z = 0.0;

            // PCA STUFF INIT

            //x
            float xx = 0.0;
            float xy = 0.0;
            float xz = 0.0;

            //y
            float yy = 0.0;
            float yz = 0.0;

            //z
            float zz = 0.0;

            for(unsigned int i = start; i < end && i<D_kd_tree_values.width; i++ )
            {
                if(i != pos)
                {
                    nearest_index = static_cast<unsigned int>(D_kd_tree_values.elements[i]+ 0.5);

                    if(nearest_index < D_V.width)
                    {
                        //vector from query point to nearest neighbor
                        float rx = D_V.elements[ nearest_index * 3 + 0 ] - vertex_x;
                        float ry = D_V.elements[ nearest_index * 3 + 1 ] - vertex_y;
                        float rz = D_V.elements[ nearest_index * 3 + 2 ] - vertex_z;

                        // instant PCA!
                        xx += rx * rx;
                        xy += rx * ry;
                        xz += rx * rz;
                        yy += ry * ry;
                        yz += ry * rz;
                        zz += rz * rz;

                    }

                }
            }


            //determinante?
            float det_x = yy * zz - yz * yz;
            float det_y = xx * zz - xz * xz;
            float det_z = xx * yy - xy * xy;

            float dir_x;
            float dir_y;
            float dir_z;
            // det X biggest
            if( det_x >= det_y && det_x >= det_z){

                if(det_x <= 0.0){
                    //not a plane
                }

                dir_x = 1.0;
                dir_y = (xz * yz - xy * zz) / det_x;
                dir_z = (xy * yz - xz * yy) / det_x;
            } //det Y biggest
            else if( det_y >= det_x && det_y >= det_z){

                if(det_y <= 0.0){
                    // not a plane
                }

                dir_x = (yz * xz - xy * zz) / det_y;
                dir_y = 1.0;
                dir_z = (xy * xz - yz * xx) / det_y;
            } // det Z biggest
            else{
                if(det_z <= 0.0){
                    // not a plane
                }

                dir_x = (yz * xy - xz * yy ) / det_z;
                dir_y = (xz * xy - yz * xx ) / det_z;
                dir_z = 1.0;
            }

            float invnorm = 1/sqrtf( dir_x * dir_x + dir_y * dir_y + dir_z * dir_z );

            result_x = dir_x * invnorm;
            result_y = dir_y * invnorm;
            result_z = dir_z * invnorm;



            // FLIP NORMALS
            float x_dir = flip_x - vertex_x;
            float y_dir = flip_y - vertex_y;
            float z_dir = flip_z - vertex_z;

            float scalar = x_dir * result_x + y_dir * result_y + z_dir * result_z;

            // gegebenfalls < durch > ersetzen
            if(scalar < 0)
            {
                result_x = -result_x;
                result_y = -result_y;
                result_z = -result_z;
            }

            D_Normals.elements[tid * D_Normals.dim ] = result_x;
            D_Normals.elements[tid * D_Normals.dim + 1 ] = result_y;
            D_Normals.elements[tid * D_Normals.dim + 2 ] = result_z;


        }
    }


}


// INTERPOLATION

__device__ float getGaussianFactor(const unsigned int& index, const unsigned int& middle_i, const int& ki, const float& norm)
{
    float val = static_cast<float>(index);
    float middle = static_cast<float>(middle_i);
    float ki_2 = static_cast<float>(ki)/2.0;

    if(val > middle)
    {
        val = val - middle;
    }else{
        val = middle - val;
    }

    if(val > ki_2)
    {
        return 0.0;
    }else{
        float border_val = 0.2;
        float gaussian = 1.0 - powf(val/ki_2, 2.0) * (1.0-border_val) ;
        return gaussian * norm;
    }

}

__global__ void InterpolationKernel(const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits,
        LBPointArray<float> D_Normals, int ki )
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < D_Normals.width)
    {
        // Interpolate to the Left
        int c = 0;
        unsigned int offset = D_kd_tree_splits.width;

        unsigned int query_index = static_cast<unsigned int>(D_kd_tree_values.elements[offset + tid]+ 0.5);
        unsigned int nearest_index;

        float gaussian = 5.0;

        if(query_index < D_Normals.width)
        {
            float n_x = D_Normals.elements[query_index * D_Normals.dim + 0];
            float n_y = D_Normals.elements[query_index * D_Normals.dim + 1];
            float n_z = D_Normals.elements[query_index * D_Normals.dim + 2];

            if(tid > 1)
            {
                for(unsigned int i = tid-1; i > 0 && c < ki/2; i--,c++ )
                {
                    nearest_index = static_cast<unsigned int>(D_kd_tree_values.elements[i + offset]+ 0.5);

                    if(nearest_index < D_Normals.width)
                    {

                        gaussian = getGaussianFactor(i, tid, ki, 5.0);

                        n_x += gaussian * D_Normals.elements[nearest_index * D_Normals.dim + 0];
                        n_y += gaussian * D_Normals.elements[nearest_index * D_Normals.dim + 1];
                        n_z += gaussian * D_Normals.elements[nearest_index * D_Normals.dim + 2];
                    }

                }
            }

            if(tid < D_Normals.width-1)
            {
                for(unsigned int i = tid+1; i < D_Normals.width && c < ki; i++,c++ )
                {
                    nearest_index = static_cast<unsigned int>(D_kd_tree_values.elements[i + offset]+ 0.5);

                    if(nearest_index < D_Normals.width)
                    {
                        gaussian = getGaussianFactor(i, tid, ki, 5.0);

                        n_x += gaussian * D_Normals.elements[nearest_index * D_Normals.dim + 0];
                        n_y += gaussian * D_Normals.elements[nearest_index * D_Normals.dim + 1];
                        n_z += gaussian * D_Normals.elements[nearest_index * D_Normals.dim + 2];
                    }
                }
            }

            float norm = sqrtf(powf(n_x,2) + powf(n_y,2) + powf(n_z,2));
            n_x = n_x/norm;
            n_y = n_y/norm;
            n_z = n_z/norm;
            D_Normals.elements[query_index * D_Normals.dim + 0] = n_x;
            D_Normals.elements[query_index * D_Normals.dim + 1] = n_y;
            D_Normals.elements[query_index * D_Normals.dim + 2] = n_z;

        }

    }

}

// DISTANCE VALUES

__device__ unsigned int parent(unsigned int pos)
{
    if(pos == 0)
    {
        return 0;
    }
    return static_cast<int>( (pos-1)/2 );
}

__device__ unsigned int leftChild(unsigned int pos)
{
    return pos*2+1;
}

__device__ unsigned int rightChild(unsigned int pos)
{
    return pos*2+2;
}

__device__ float euklideanDistance(float x, float y, float z)
{
    return x*x + y*y + z*z;
}

__device__ bool isLeftChild(int pos)
{
    if( leftChild(parent(pos)) == pos )
    {
        return true;
    }else{
        return false;
    }
}

__device__ void getApproximatedKdTreePositionFromSubtree(const LBPointArray<float>& D_kd_tree_values, const LBPointArray<unsigned char>& D_kd_tree_splits,
    int current_dim, const LBPointArray<float>& D_V, float x, float y, float z, unsigned int &pos, unsigned int tid)
{
    while(pos < D_kd_tree_splits.width)
    {
        current_dim = static_cast<unsigned int>(D_kd_tree_splits.elements[pos]);
        if(current_dim == 0)
        {
            if(x <= D_kd_tree_values.elements[pos] )
            {
                pos = pos*2+1;
            } else {
                pos = pos*2+2;
            }
        } else if(current_dim == 1) {

            if(y <= D_kd_tree_values.elements[pos] ){
                pos = pos*2+1;
            }else{
                pos = pos*2+2;
            }
        } else {
            if(z <= D_kd_tree_values.elements[pos] ){
                pos = pos*2+1;
            }else{
                pos = pos*2+2;
            }
        }
    }
}

__device__ void getApproximatedKdTreePosition(const LBPointArray<float>& D_kd_tree_values, const LBPointArray<unsigned char>& D_kd_tree_splits, const LBPointArray<float>& D_V, float x, float y, float z, unsigned int &pos, unsigned int tid)
{
    pos = 0;
    unsigned int current_dim = 0;

    while(pos < D_kd_tree_splits.width)
    {
        current_dim = static_cast<unsigned int>(D_kd_tree_splits.elements[pos]);

        if(current_dim == 0)
        {
            if(x <= D_kd_tree_values.elements[pos] )
            {
                pos = pos*2+1;
            } else {
                pos = pos*2+2;
            }
        } else if(current_dim == 1) {

            if(y <= D_kd_tree_values.elements[pos] ){
                pos = pos*2+1;
            }else{
                pos = pos*2+2;
            }
        } else {
            if(z <= D_kd_tree_values.elements[pos] ){
                pos = pos*2+1;
            }else{
                pos = pos*2+2;
            }
        }

    }
}

__device__ void getNNFromIndex(const LBPointArray<float>& D_kd_tree_values, const LBPointArray<unsigned char>& D_kd_tree_splits,
             const unsigned int& kd_pos, unsigned int *nn, int k)
{

    unsigned int start = kd_pos - (k/2);
    unsigned int i_nn = 0;

    for(unsigned int i = start; i_nn < k; i++, i_nn++)
    {
        if(i >= D_kd_tree_values.width )
        {
            nn[i_nn] = static_cast<unsigned int>(D_kd_tree_values.elements[D_kd_tree_values.width-1]+0.5);
        }else if(i < D_kd_tree_splits.width)
        {
            nn[i_nn] = static_cast<unsigned int>(D_kd_tree_values.elements[D_kd_tree_splits.width]+0.5);
        }else{
            nn[i_nn] = static_cast<unsigned int>(D_kd_tree_values.elements[i]+0.5);
        }
    }

}

__device__ void calculateDistance(const float& x, const float& y, const float& z,
                                    const float& n_x, const float& n_y, const float& n_z,
                                    const float& qp_x, const float& qp_y, const float& qp_z,
                                    float& euklidean_distance, float& projected_distance, unsigned int tid)
{
    float vec_x = (qp_x - x);
    float vec_y = (qp_y - y);
    float vec_z = (qp_z - z);

    if(tid == 5)
    {
        printf("qp - nn: %f %f %f\n", vec_x, vec_y, vec_z);
    }

    projected_distance = vec_x * n_x + vec_y * n_y + vec_z * n_z;

    euklidean_distance = sqrt( vec_x * vec_x + vec_y * vec_y + vec_z * vec_z );
}

//distance function without transformation
__global__ void GridDistanceKernel(const LBPointArray<float> D_V, const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits,
     const LBPointArray<float> D_Normals, QueryPointC* D_Query_Points, const unsigned int qp_size, int k, float voxel_size)
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < qp_size)
    {
        k = 1;
        if(tid == 5)
        {
            printf("TEEST\n");
        }

        unsigned int kd_pos;
        // nearest neighbors;
        unsigned int* nn = (unsigned int*)malloc(sizeof(unsigned int)*3*k );

        float qp_x = D_Query_Points[tid].m_position.x;
        float qp_y = D_Query_Points[tid].m_position.y;
        float qp_z = D_Query_Points[tid].m_position.z;

        getApproximatedKdTreePosition(D_kd_tree_values, D_kd_tree_splits, D_V,
                qp_x,
                 qp_y,
                qp_z,
                 kd_pos , tid );

        getNNFromIndex(D_kd_tree_values,D_kd_tree_splits, kd_pos, nn, k);

        if(tid == 5)
        {
            printf("Query Point: %f %f %f\n",qp_x, qp_y, qp_z);
            for(int i=0; i<k; i++)
            {
                printf("Nearest Point %d: %f %f %f\n",i, D_V.elements[ nn[i] * 3 + 0], D_V.elements[ nn[i] * 3 + 1],D_V.elements[ nn[i] * 3 + 2]);
            }

        }

        float projected_distance;
        float euklidean_distance;

        float x = 0.0;
        float y = 0.0;
        float z = 0.0;

        float n_x = 0.0;
        float n_y = 0.0;
        float n_z = 0.0;

        float weight_sum = 0.0;

        for(int i=0; i<k; i++)
        {
            unsigned int point_position =  nn[i];

            float gaussian_factor = getGaussianFactor(i, k/2, k, 5.0);
            weight_sum += gaussian_factor;

            if(point_position > 0 && point_position < D_V.width)
            {
                x += gaussian_factor * D_V.elements[ point_position * 3 + 0];
                y += gaussian_factor * D_V.elements[ point_position * 3 + 1];
                z += gaussian_factor * D_V.elements[ point_position * 3 + 2];

                n_x += gaussian_factor * D_Normals.elements[ point_position * 3 + 0 ];
                n_y += gaussian_factor * D_Normals.elements[ point_position * 3 + 1 ];
                n_z += gaussian_factor * D_Normals.elements[ point_position * 3 + 2 ];
            }

        }

        x = x / weight_sum;
        y = y / weight_sum;
        z = z / weight_sum;

        float n_norm = sqrt(n_x*n_x + n_y*n_y + n_z*n_z );

        n_x = n_x / n_norm;
        n_y = n_y / n_norm;
        n_z = n_z / n_norm;

        calculateDistance(x, y, z, n_x, n_y, n_z, qp_x, qp_y, qp_z, euklidean_distance, projected_distance, tid );

        D_Query_Points[tid].m_distance = projected_distance/15.0;
        D_Query_Points[tid].m_invalid = false;

        // if (euklidean_distance > 2 * 1.7320 * voxel_size)
        // {
        //     D_Query_Points[tid].m_invalid = true;
        // }

        if(tid == 5)
        {
            printf("Projected Distance: %f\n", projected_distance);
            printf("Euklidean Distance: %f\n", euklidean_distance);
            printf("Distance in Kernel: %f\n", D_Query_Points[tid].m_distance);
        }

        free(nn);

    }

}

/// HOST FUNCTIONS ///

void CudaSurface::init(){
    // set default k
    this->m_k = 10;

    // set default ki
    this->m_ki = 10;
    this->m_kd = 5;

    // set default flippoint
    this->m_vx = 1000000.0;
    this->m_vy = 1000000.0;
    this->m_vz = 1000000.0;

    this->m_calc_method = 0;

    this->m_reconstruction_mode = false;
}

CudaSurface::CudaSurface(LBPointArray<float>& points, int device)
{
    this->init();

    this->getCudaInformation(device);

    this->V.dim = points.dim;

    this->V.width = points.width;

    mallocPointArray(V);

    for(int i = 0; i<points.width*points.dim; i++)
    {

        this->V.elements[i] = points.elements[i];

    }

    this->initKdTree();

}

CudaSurface::CudaSurface(floatArr& points, size_t num_points, int device)
{

    this->init();

    this->getCudaInformation(device);

    this->V.dim = 3;

    this->V.width = static_cast<int>(num_points);

    mallocPointArray(V);

    this->V.elements = points.get();

    this->initKdTree();

}

void CudaSurface::getCudaInformation(int device)
{
    m_device = device;
    m_mps = 0;
    m_threads_per_mp = 0;
    m_threads_per_block = 0;
    m_size_thread_block = new int(3);
    m_size_grid = new int(3);
    m_device_global_memory = 0;


    cudaSetDevice(m_device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);


    m_mps = deviceProp.multiProcessorCount;
    m_threads_per_mp = deviceProp.maxThreadsPerMultiProcessor;
    m_threads_per_block = deviceProp.maxThreadsPerBlock;
    m_size_thread_block[0] = deviceProp.maxThreadsDim[0];
    m_size_thread_block[1] = deviceProp.maxThreadsDim[1];
    m_size_thread_block[2] = deviceProp.maxThreadsDim[2];
    m_size_grid[0] = deviceProp.maxGridSize[0];
    m_size_grid[1] = deviceProp.maxGridSize[1];
    m_size_grid[2] = deviceProp.maxGridSize[2];
    m_device_global_memory = (unsigned long long) deviceProp.totalGlobalMem;

}

void CudaSurface::getNormals(LBPointArray<float>& output_normals)
{

    output_normals.dim = this->Result_Normals.dim;
    output_normals.width = this->Result_Normals.width;
    //output_normals.elements = (float*)malloc( this->Result_Normals.dim * this->Result_Normals.width * sizeof(float) ) ;

    for(int i = 0; i< this->Result_Normals.dim * this->Result_Normals.width; i++)
    {
        output_normals.elements[i] = this->Result_Normals.elements[i];
    }

}

void CudaSurface::getNormals(floatArr output_normals)
{

    for(int i = 0; i< this->Result_Normals.dim * this->Result_Normals.width; i++)
    {
        output_normals[i] = this->Result_Normals.elements[i];
    }
}

template <typename T>
void CudaSurface::generateDevicePointArray(LBPointArray<T>& D_m, int width, int dim)
{

    D_m.width = width;
    D_m.dim = dim;
    size_t size = D_m.width * D_m.dim * sizeof(T);
    HANDLE_ERROR( cudaMalloc(&D_m.elements, size) );

}

template <typename T>
void CudaSurface::copyToDevicePointArray(LBPointArray<T>* m, LBPointArray<T>& D_m)
{

    size_t size = m->width * m->dim * sizeof(T);
    HANDLE_ERROR( cudaMemcpy(D_m.elements, m->elements, size, cudaMemcpyHostToDevice) );

}

template <typename T>
void CudaSurface::copyToDevicePointArray(LBPointArray<T>& m, LBPointArray<T>& D_m)
{

    size_t size = m.width * m.dim * sizeof(T);
    HANDLE_ERROR( cudaMemcpy(D_m.elements, m.elements, size, cudaMemcpyHostToDevice) );

}

void CudaSurface::GPU_NN()
{

    unsigned int threadsPerBlock = this->m_threads_per_block;
    unsigned int blocksPerGrid = (D_V.width + threadsPerBlock-1) / threadsPerBlock;

    // kNN-search and Normal calculation
    // Flip directly in kernel

    KNNKernel1<<<blocksPerGrid, threadsPerBlock  >>>(this->D_V, this->D_kd_tree_values, this->D_kd_tree_splits, this->D_Normals,
                                                 this->m_k, this->m_calc_method, this->m_vx, this->m_vy, this->m_vz);
    cudaDeviceSynchronize();

    InterpolationKernel<<<blocksPerGrid, threadsPerBlock >>>(this->D_kd_tree_values, this->D_kd_tree_splits,
                                                        this->D_Normals, this->m_ki );

    cudaDeviceSynchronize();

    size_t size = this->Result_Normals.width * this->Result_Normals.dim * sizeof(float);

    if(!m_reconstruction_mode)
    {
        HANDLE_ERROR( cudaMemcpy(Result_Normals.elements, D_Normals.elements, size, cudaMemcpyDeviceToHost ) );
    }

}

void CudaSurface::initKdTree() {


    kd_tree_gen = boost::shared_ptr<LBKdTree>(new LBKdTree(this->V, OpenMPConfig::getNumThreads() ) );
    this->kd_tree_values = kd_tree_gen->getKdTreeValues().get();
    this->kd_tree_splits = kd_tree_gen->getKdTreeSplits().get();

    generateDevicePointArray( D_kd_tree_values, this->kd_tree_values->width, this->kd_tree_values->dim);
    copyToDevicePointArray( this->kd_tree_values, D_kd_tree_values);

    generateDevicePointArray( D_kd_tree_splits, this->kd_tree_splits->width, this->kd_tree_splits->dim);
    copyToDevicePointArray( this->kd_tree_splits, D_kd_tree_splits);


    //free(this->kd_tree.elements);
}

void CudaSurface::setKn(int kn) {

    this->m_k = kn;

}

void CudaSurface::setKi(int ki)
{
    this->m_ki = ki;
}

void CudaSurface::setKd(int kd)
{
    this->m_kd = kd;
}

void CudaSurface::setFlippoint(float v_x, float v_y, float v_z) {

    this->m_vx = v_x;
    this->m_vy = v_y;
    this->m_vz = v_z;

}

void CudaSurface::setMethod(std::string& method)
{

    if( strcmp( method.c_str(), "PCA") == 0 )
    {
        this->m_calc_method = 0;
    }
    else if( strcmp( method.c_str(), "RANSAC") == 0)
    {
        this->m_calc_method = 1;
    }
    else
    {
        printf("WARNING: Normal Calculation Method is not implemented\n");
    }

}

void CudaSurface::printSettings() {

    printf("    Nearest Neighbors = %d\n",this->m_k);

    printf("    Flip point = (%f, %f, %f)\n", this->m_vx, this->m_vy, this->m_vz);

    switch(this->m_calc_method){
        case 0:
            printf("    Method = 'PCA'\n");
            break;
        case 1:
            printf("    Method = 'RANSAC'\n");
            break;
    }

    printf("\n");

}


void CudaSurface::calculateNormals() {

    generatePointArray( this->Result_Normals, V.width, V.dim);


    generateDevicePointArray( D_V, this->V.width, this->V.dim );

    generateDevicePointArray( D_Normals, this->Result_Normals.width, this->Result_Normals.dim);

    //COPY STUFF
    copyToDevicePointArray( V, D_V );

    //Cuda Kernels
    GPU_NN();

}

void CudaSurface::interpolateNormals() {

}

void CudaSurface::freeGPU() {
    cudaFree(D_V.elements);
    cudaFree(D_kd_tree_values.elements);
    cudaFree(D_kd_tree_splits.elements);
    cudaFree(D_Normals.elements);
}

void CudaSurface::setReconstructionMode(bool mode)
{
    this->m_reconstruction_mode = mode;
}



void CudaSurface::distances(std::vector<QueryPoint<Vec> >& query_points, float voxel_size)
{
    std::cout << "Calculate Distances..." << std::endl;
    std::cout << "Size of entry" << int(sizeof(query_points[0]) ) << std::endl;
    std::cout << "Example query point: " << query_points[5].m_position.x << "|" << query_points[5].m_position.y << "|" << query_points[5].m_position.z << std::endl;
    std::cout << "Distance: " << query_points[5].m_distance << std::endl;
    std::cout << "Invalid: " << query_points[5].m_invalid << std::endl;

    // thrust::device_vector< QueryPointC > d_query_points = query_points;

    // QueryPointC* qpArray = thrust::raw_pointer_cast( &d_query_points[0] );

    QueryPointC *d_query_points;
    cudaMalloc((void**)&d_query_points, sizeof(QueryPointC)*query_points.size() );

    HANDLE_ERROR( cudaMemcpy(d_query_points, &query_points[0], sizeof(QueryPointC)*query_points.size(), cudaMemcpyHostToDevice) );

    int threadsPerBlock = this->m_threads_per_block;
    int blocksPerGrid = (query_points.size() + threadsPerBlock-1)/threadsPerBlock;

    GridDistanceKernel<<<blocksPerGrid,threadsPerBlock>>>(D_V, D_kd_tree_values, D_kd_tree_splits, D_Normals, d_query_points, query_points.size(), this->m_kd, voxel_size);

    cudaDeviceSynchronize();

    HANDLE_ERROR( cudaMemcpy(&query_points[0], d_query_points, sizeof(QueryPointC)*query_points.size(), cudaMemcpyDeviceToHost) );

    cudaFree(d_query_points);

    int num_invalid = 0;

    for(int i=0; i<query_points.size(); i++)
    {
        if(query_points[i].m_invalid)
        {
            num_invalid ++;
        }
    }

    std::cout << "Num invalid: " << num_invalid << std::endl;
    std::cout << "Example query point: " << query_points[5].m_position.x << "|" << query_points[5].m_position.y << "|" << query_points[5].m_position.z << std::endl;
    std::cout << "Distance: " << query_points[5].m_distance << std::endl;
    std::cout << "Invalid: " << query_points[5].m_invalid << std::endl;
    std::cout << "Distance Calculation finished" << std::endl;
}


CudaSurface::~CudaSurface() {

    // clearn up resulting normals and kd_tree
    // Pointcloud has to be cleaned up by user

    if(this->Result_Normals.width > 0){
        free(Result_Normals.elements);
    }

    this->freeGPU();


    // if(this->kd_tree_values->width > 0){
    //     free(this->kd_tree_values.elements);
    // }
}


} /* namespace lvr2 */
