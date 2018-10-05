/*
 * This file is part of cudaNormals.
 *
 * cudaNormals is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Foobar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cudaNormals.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * CudaSurface.cu
 *
 * @author Alexander Mock
 */

#include <lvr2/reconstruction/cuda/CudaSurface.hpp>

namespace lvr2
{

/// Define Kernels

__global__ void FlipNormalsKernel(const LBPointArray<float> D_V, LBPointArray<float> D_Normals, float x, float y, float z);

__global__ void KNNKernel(const LBPointArray<float> D_V,
        const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits ,
        LBPointArray<float> D_Normals, int k, int method,
        float flip_x, float flip_y, float flip_z);

__global__ void KNNKernel2(const LBPointArray<float> D_V,
        const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits ,
        LBPointArray<float> D_Normals, int k, int method,
        float flip_x, float flip_y, float flip_z);

__global__ void KNNKernel3(const LBPointArray<float> D_V,
        const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits ,
        LBPointArray<float> D_Normals, int k, int method,
        float flip_x, float flip_y, float flip_z);

__global__ void KNNKernel4(const LBPointArray<float> D_V,
        const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits ,
        LBPointArray<float> D_Normals, int k, int method,
        float flip_x, float flip_y, float flip_z);

// IN WORK
__global__ void InterpolationKernel(const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits,
        LBPointArray<float> D_Normals, int ki );

// IN WORK
__global__ void GridDistanceKernel(const LBPointArray<float> D_V,const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits,
         const LBPointArray<float> D_Normals, QueryPointC* D_Query_Points, const unsigned int qp_size, int k);



__global__ void FlipNormalsKernel(const LBPointArray<float> D_V,
                                  LBPointArray<float> D_Normals,
                                  float x, float y, float z)
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < D_V.width){
        float x_dir = x - D_V.elements[tid];
        float y_dir = y - D_V.elements[D_V.width + tid];
        float z_dir = z - D_V.elements[2 * D_V.width + tid];

        float scalar = ( x_dir * D_Normals.elements[tid] + y_dir * D_Normals.elements[D_Normals.width + tid] + z_dir * D_Normals.elements[2 * D_Normals.width + tid] );

        // gegebenfalls < durch > ersetzen
        if(scalar < 0)
        {
            D_Normals.elements[tid] = -D_Normals.elements[tid];
            D_Normals.elements[D_Normals.width + tid] = -D_Normals.elements[D_Normals.width + tid];
            D_Normals.elements[2 * D_Normals.width + tid] = -D_Normals.elements[2 * D_Normals.width + tid];
        }
    }
}

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


__device__ void calculateNormalRansa(float* nn_vecs,
                                      int k,
                                      int max_iterations,
                                      float& x, float& y, float& z)
{
    float min_dist = FLT_MAX;
    int iterations = 0;

    for(int i=3; i<k*3; i+=3){
        //~ printf("%f %f %f\n", last_vec[0], last_vec[1], last_vec[2]);

        int j = (i + int(k/3) * 3) % (k * 3);

        float n_x = nn_vecs[j+1]*nn_vecs[i+2] - nn_vecs[j+2]*nn_vecs[i+1];
        float n_y = nn_vecs[j+2]*nn_vecs[i+0] - nn_vecs[j+0]*nn_vecs[i+2];
        float n_z = nn_vecs[j+0]*nn_vecs[i+1] - nn_vecs[j+1]*nn_vecs[i+0];

        float norm = sqrtf( n_x*n_x + n_y*n_y + n_z*n_z );


        if( norm != 0.0 ){

            float norm_inv = 1.0/norm;

            n_x = n_x * norm_inv;
            n_y = n_y * norm_inv;
            n_z = n_z * norm_inv;

            float cum_dist = 0.0;
            for(int j=0; j<k*3; j+=3){
                cum_dist += abs(n_x * nn_vecs[j] + n_y * nn_vecs[j+1] + n_z * nn_vecs[j+2]);
            }

            if(cum_dist < min_dist) {

                iterations = 0;
                min_dist = cum_dist;
                x = n_x;
                y = n_y;
                z = n_z;

            } else if(iterations < max_iterations) {

                iterations ++;

            }else{

                return;

            }
        }

    }
}

__device__ void calculateNormalPCA(float* nn_vecs, int k, float& n_x, float& n_y, float& n_z)
{

    // ilikebigbits.com/blog/2015/3/2/plane-from-points


    //x
    float xx = 0.0;
    float xy = 0.0;
    float xz = 0.0;

    //y
    float yy = 0.0;
    float yz = 0.0;

    //z
    float zz = 0.0;

    for(int i=0; i<k; i++)
    {
        float rx = nn_vecs[i*3+0];
        float ry = nn_vecs[i*3+1];
        float rz = nn_vecs[i*3+2];

        xx += rx * rx;
        xy += rx * ry;
        xz += rx * rz;
        yy += ry * ry;
        yz += ry * rz;
        zz += rz * rz;
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

    n_x = dir_x * invnorm;
    n_y = dir_y * invnorm;
    n_z = dir_z * invnorm;

}

__device__ void getNearestNeighbors(const LBPointArray<float>& D_V,
                                    const LBPointArray<float>& D_kd_tree_values,
                                    const LBPointArray<unsigned char>& D_kd_tree_splits,
                                    int k,
                                    unsigned int subtree_pos,
                                    unsigned int pos,
                                    unsigned int pos_value,
                                    float* nn_vecs)
{

    unsigned int iterator = subtree_pos;
    unsigned int max_nodes = 1;
    bool leaf_reached = false;
    unsigned int i_nn = 0;

    unsigned int query_index = pos_value * D_V.dim;

    float query_x = D_V.elements[ query_index ];
    float query_y = D_V.elements[ query_index + 1 ];
    float query_z = D_V.elements[ query_index + 2 ];

    // like width search
    // go kd-tree up until max_nodes(leaf_nodes of subtree) bigger than needed nodes k
    // iterator = iterator * 2 + 1 -> go to
    for( ; iterator < D_kd_tree_values.width; iterator = iterator * 2 + 1, max_nodes *= 2)
    {
    // collect nodes from current height
        for( unsigned int i=0; i < max_nodes && iterator + i < D_kd_tree_values.width; i++)
        {
            unsigned int current_pos = iterator + i;
            unsigned int leaf_value  = static_cast<unsigned int>(D_kd_tree_values.elements[ current_pos ] + 0.5 );

            if( leaf_reached && i_nn <= k*3 )
            {
                if( leaf_value != pos_value && leaf_value < D_V.width )
                {
                    unsigned int curr_nn_index = leaf_value * D_V.dim;

                    float nn_x = D_V.elements[ curr_nn_index ] - query_x;
                    float nn_y = D_V.elements[ curr_nn_index + 1 ] - query_y;
                    float nn_z = D_V.elements[ curr_nn_index + 2 ] - query_z;

                    if(nn_x != 0.0 || nn_y != 0.0 || nn_z != 0.0)
                    {
                        nn_vecs[ i_nn ]     = nn_x;
                        nn_vecs[ i_nn + 1 ] = nn_y;
                        nn_vecs[ i_nn + 2 ] = nn_z;
                        i_nn += 3;
                    }
                }
            } else if( current_pos * 2 + 1 >= D_kd_tree_values.width ) {

                unsigned int curr_nn_index = leaf_value * D_V.dim;
                //first leaf reached
                leaf_reached = true;
                if( leaf_value != pos_value && i_nn <= k*3 )
                {
                    nn_vecs[i_nn]   = D_V.elements[ curr_nn_index ] - query_x;
                    nn_vecs[i_nn+1] = D_V.elements[ curr_nn_index + 1 ] - query_y;
                    nn_vecs[i_nn+2] = D_V.elements[ curr_nn_index + 2] - query_z;
                    i_nn += 3;
                }
            }
        }
    }

}

__device__ void calculateNormalFromSubtree(const LBPointArray<float>& D_V, const LBPointArray<float>& D_kd_tree_values, const LBPointArray<unsigned char>& D_kd_tree_splits,
                                            int pos, int k, float& x, float& y, float& z, int method )
{
        //~
        //~  Step 1: get upper node
        //~  Step 2: get child nodes != query node
        //~  Step 3: calculate normals
        //~

        unsigned int pos_value = static_cast<unsigned int>(D_kd_tree_values.elements[pos]+0.5);

        unsigned int subtree_pos = pos;
        unsigned int i;
        for(i=1; i<(k+1) && subtree_pos>0; i*=2) {
                subtree_pos = (int)((subtree_pos  - 1) / 2);
        }
        //~ printf("subtree_pos: %d\n",subtree_pos);

        // k+1 FIX
        float * nn_vecs = (float*)malloc(3*(k+1)*sizeof(float));


        getNearestNeighbors(D_V, D_kd_tree_values, D_kd_tree_splits , k, subtree_pos, pos, pos_value, nn_vecs);

        if(method == 0){
                //PCA
                calculateNormalPCA(nn_vecs, k, x, y, z);
        } else if(method == 1) {
                //RANSAC
                calculateNormalRansa(nn_vecs, k, 8, x, y, z);
        }

        free(nn_vecs);

}

//distance function without transformation
__global__ void KNNKernel(const LBPointArray<float> D_V,
        const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits ,
        LBPointArray<float> D_Normals, int k, int method,
        float flip_x, float flip_y, float flip_z)
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < D_V.width){

        unsigned int pos = GetKdTreePosition(D_kd_tree_values, D_kd_tree_splits, D_V.elements[tid * D_V.dim], D_V.elements[tid * D_V.dim + 1], D_V.elements[tid * D_V.dim +2] );

        float result_x = 0.0;
        float result_y = 0.0;
        float result_z = 0.0;

        calculateNormalFromSubtree(D_V, D_kd_tree_values, D_kd_tree_splits , pos, k, result_x, result_y, result_z, method);

        // FLIP NORMALS
        float x_dir = flip_x - D_V.elements[D_V.dim * tid];
        float y_dir = flip_y - D_V.elements[D_V.dim * tid + 1];
        float z_dir = flip_z - D_V.elements[D_V.dim * tid + 2];

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

__global__ void KNNKernel2(const LBPointArray<float> D_V,
        const LBPointArray<float> D_kd_tree_values, const LBPointArray<unsigned char> D_kd_tree_splits ,
        LBPointArray<float> D_Normals, int k, int method,
        float flip_x, float flip_y, float flip_z)
{
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // instant leaf!
    unsigned int pos = tid + D_kd_tree_splits.width;

    if(tid < D_V.width && pos < D_kd_tree_values.width)
    {

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

            D_Normals.elements[vertex_index * D_Normals.dim ] = result_x;
            D_Normals.elements[vertex_index * D_Normals.dim + 1 ] = result_y;
            D_Normals.elements[vertex_index * D_Normals.dim + 2 ] = result_z;


        }
    }


}


__global__ void KNNKernel3(const LBPointArray<float> D_V,
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

__global__ void KNNKernel4(const LBPointArray<float> D_V,
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

    // unsigned int point_index = static_cast<unsigned int>(D_kd_tree_values.elements[pos]+0.5);
    // float nearest_x = D_V.elements[point_index*3 + 0];
    // float nearest_y = D_V.elements[point_index*3 + 1];
    // float nearest_z = D_V.elements[point_index*3 + 2];


    // float current_dist = euklideanDistance(nearest_x-x, nearest_y-y, nearest_z-z);

    // // check a split that could indicate a point not so far away
    // unsigned int better_split = pos;

    // for(int i=0; i<4; i++)
    // {
    //     bool left = isLeftChild(better_split);
    //     better_split = parent(better_split);

    //     current_dim --;
    //     if(current_dim < 0)
    //     {
    //         current_dim = 2;
    //     }

    //     // check dim
    //     float current_dist_dim;
    //     float current_dist_split;
    //     if(current_dim == 0)
    //     {
    //         current_dist_dim = abs(nearest_x - x);
    //         current_dist_split = abs(D_kd_tree_values.elements[better_split] - x );
    //     } else if(current_dim == 1)
    //     {
    //         current_dist_dim = abs(nearest_y - y);
    //         current_dist_split = abs(D_kd_tree_values.elements[better_split] - y );
    //     } else if(current_dim == 2)
    //     {
    //         current_dist_dim = abs(nearest_z - z);
    //         current_dist_split = abs(D_kd_tree_values.elements[better_split] - z );
    //     }

    //     if(tid == 5)
    //     {
    //         printf("Pos %d, Dim %d, Split: %f, Dist to Split: %f, Dist to nearest point: %f\n"
    //             ,better_split,current_dim, D_kd_tree_values.elements[better_split], current_dist_split, current_dist_dim);

    //     }


    //     if(current_dist_split < current_dist_dim)
    //     {
    //         // search in other tree
    //         unsigned int other_tree_pos;
    //         if(left)
    //         {
    //             if(tid == 5)
    //             {
    //                 printf("take right of split axis %f\n",D_kd_tree_values.elements[better_split] );
    //             }
    //             other_tree_pos = rightChild(better_split);
    //         }else{
    //             if(tid == 5)
    //             {
    //                 printf("take left of split axis %f\n",D_kd_tree_values.elements[better_split] );
    //             }
    //             other_tree_pos = leftChild(better_split);
    //         }
    //         unsigned int position_found = other_tree_pos;
    //         getApproximatedKdTreePositionFromSubtree(D_kd_tree_values, D_kd_tree_splits, current_dim, D_V,
    //                                                 x, y, z, position_found, tid);

    //         unsigned int new_nn = static_cast<unsigned int>(D_kd_tree_values.elements[position_found]+0.5);
    //         if(new_nn < D_V.width && new_nn > 0)
    //         {

    //             if(tid == 5)
    //             {

    //                 printf("Dim %d, old nn: %f %f %f , new nn: %f %f %f\n"
    //                 ,current_dim, nearest_x, nearest_y , nearest_z,
    //                 D_V.elements[new_nn * 3 + 0],D_V.elements[new_nn * 3 + 1],D_V.elements[new_nn * 3 + 2]  );

    //             }

    //             if(new_nn < D_V.width)
    //             {
    //                 float new_dist = euklideanDistance(D_V.elements[new_nn * 3 + 0]-x, D_V.elements[new_nn * 3 + 1]-y, D_V.elements[new_nn * 3 + 2]-z);
    //                 if( current_dist > new_dist )
    //                 {
    //                     //printf("Better Point FOUND !\n");
    //                     current_dist = new_dist;
    //                     nearest_x = D_V.elements[new_nn * 3 + 0];
    //                     nearest_y = D_V.elements[new_nn * 3 + 1];
    //                     nearest_z = D_V.elements[new_nn * 3 + 2];
    //                     pos = new_nn;
    //                 }
    //             }

    //         }

    //     }


    // }

}

__device__ void getNNFromIndex(const LBPointArray<float>& D_kd_tree_values, const LBPointArray<unsigned char>& D_kd_tree_splits,
             const unsigned int& kd_pos, unsigned int *nn, int k)
{
     // unsigned int subtree_pos = kd_pos;
    // int i;
    // for(i=1; i<(k+1) && subtree_pos>0; i*=2) {
    //         subtree_pos = (int)((subtree_pos  - 1) / 2);
    // }

    // int iterator = subtree_pos;
    // int max_nodes = 1;
    // bool leaf_reached = false;


    // nn[0] = static_cast<int>(D_kd_tree.elements[kd_pos]+0.5);
    // int i_nn = 1;

    // like width search
    // go kd-tree up until max_nodes(leaf_nodes of subtree) bigger than needed nodes k
    // iterator = iterator * 2 + 1 -> go to
    // for( ; iterator < D_kd_tree.width; iterator = iterator * 2 + 1, max_nodes *= 2)
    // {
    // // collect nodes from current height
    //     for( int i=0; i < max_nodes && iterator + i < D_kd_tree.width; i++)
    //     {
    //         int current_pos = iterator + i;
    //         int leaf_value  = (int)(D_kd_tree.elements[ current_pos ] + 0.5 );

    //         if( leaf_reached && i_nn < k  )
    //         {
    //             if( current_pos != kd_pos )
    //             {
    //                 nn[i_nn++] = leaf_value;
    //             }
    //         } else if( current_pos * 2 + 1 >= D_kd_tree.width )
    //         {
    //             //first leaf reached
    //             leaf_reached = true;
    //             if( leaf_value != kd_pos && i_nn < k )
    //             {
    //                 nn[i_nn++] = leaf_value;
    //             }
    //         }
    //     }
    // }


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
    KNNKernel4<<<blocksPerGrid, threadsPerBlock  >>>(this->D_V, this->D_kd_tree_values, this->D_kd_tree_splits, this->D_Normals,
                                                 this->m_k, this->m_calc_method, this->m_vx, this->m_vy, this->m_vz);
    cudaDeviceSynchronize();

    //TODO: Interpolate
    std::cout << "Start Interpolation..." << std::endl;
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

    //~ struct Matrix test;

    kd_tree_gen = boost::shared_ptr<LBKdTree>(new LBKdTree(this->V) );
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

    // for(unsigned int i = 0; i< 100; i++)
    // {
    //     this->debug(i, this->m_k);
    // }
    //this->debug(28,this->m_k);
    //this->debug2(28,this->m_k);


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

/// DEEEEBUUUUUUG

void CudaSurface::debug(unsigned int query_index, int k)
{

    std::cout << "Debug " << query_index << std::endl;
    float* query_point = (float*)malloc(sizeof(float) * 3);
    query_point[0] = this->V.elements[query_index*3];
    query_point[1] = this->V.elements[query_index*3+1];
    query_point[2] = this->V.elements[query_index*3+2];

    unsigned int* nn = (unsigned int*)malloc(sizeof(unsigned int) * k );



    nearestNeighborsHost(query_point, nn, k);
    std::stringstream ss;
    ss << "debug" << query_index << ".ply";

    this->writeToPly(query_point, nn, k, ss.str() );

    free(query_point);
    free(nn);
}

void CudaSurface::debug2(unsigned int query_index, int k)
{

    std::cout << "Debug2 " << query_index << std::endl;
    float* query_point = (float*)malloc(sizeof(float) * 3);
    query_point[0] = this->V.elements[query_index*3];
    query_point[1] = this->V.elements[query_index*3+1];
    query_point[2] = this->V.elements[query_index*3+2];

    unsigned int* nn = (unsigned int*)malloc(sizeof(unsigned int) * k );



    nearestNeighborsHost(query_point, nn, k, 1);
    std::stringstream ss;
    ss << "debug" << query_index << "_b.ply";

    this->writeToPly(query_point, nn, k, ss.str() );

    free(query_point);
    free(nn);
}

// size*3 = real length of nn array
void CudaSurface::writeToPly(float* point, float* nn, int size, std::string filename)
{
    std::ofstream ply_file;
      ply_file.open( filename.c_str() );
      ply_file << "ply" << std::endl;
    ply_file << "format ascii 1.0" << std::endl;
    ply_file << "comment made by amock" << std::endl;
    ply_file << "element vertex " << 1+size << std::endl;
    ply_file << "property float32 x" << std::endl;
    ply_file << "property float32 y" << std::endl;
    ply_file << "property float32 z" << std::endl;
    ply_file << "property uchar red" << std::endl;
    ply_file << "property uchar green" << std::endl;
    ply_file << "property uchar blue" << std::endl;
    ply_file << "end_header" << std::endl;

    ply_file << point[0] << " " << point[1] << " " << point[2] << " 255 0 0" << std::endl;
    for(int i=0; i<size; i++)
    {
        ply_file << nn[i*3+0] << " " << nn[i*3+1] << " " << nn[i*3+2] << " 0 255 0" << std::endl;
    }

      ply_file.close();
}

void CudaSurface::writeToPly(float* point, unsigned int* nn_i, int size, std::string filename)
{
    float* nn = (float*)malloc( sizeof(float) * 3 * size );
    for(int i=0; i< size; i++)
    {
        nn[i*3] = this->V.elements[ nn_i[i] * 3 ];
        nn[i*3 + 1] = this->V.elements[ nn_i[i] * 3 + 1 ];
        nn[i*3 + 2] = this->V.elements[ nn_i[i] * 3 + 2 ];
    }
    this->writeToPly(point, nn, size, filename);
    free(nn);
}

void CudaSurface::writeToPly(unsigned int point_i, unsigned int* nn_i, int size, std::string filename)
{
    float* point = (float*)malloc(sizeof(float) * 3);
    point[0] = this->V.elements[point_i*3];
    point[1] = this->V.elements[point_i*3+1];
    point[2] = this->V.elements[point_i*3+2];

    this->writeToPly(point, nn_i, size, filename);

    free(point);
}

void CudaSurface::nearestNeighborsHost(float* point, unsigned int* nn, int k, int mode)
{
    unsigned int kd_pos = this->getKdTreePosition(point[0], point[1], point[2]);
    unsigned int point_index = static_cast<unsigned int>(this->kd_tree_values->elements[kd_pos]);

    float x = this->V.elements[point_index*3];
    float y = this->V.elements[point_index*3+1];
    float z = this->V.elements[point_index*3+2];

    if( x!=point[0] || y!=point[1] || z!=point[2] )
    {
        std::cout << "Query Point: " << point[0] << " " << point[1] << " " << point[2] << std::endl;
        std::cout << "Found Point: " << x << " " << y << " " << z << std::endl;

        unsigned int kd_posB = this->getKdTreePosition(x, y, z);
        unsigned int point_index_correction = static_cast<unsigned int>(this->kd_tree_values->elements[kd_posB]);
        float x_corr = this->V.elements[point_index_correction*3];
        float y_corr = this->V.elements[point_index_correction*3+1];
        float z_corr = this->V.elements[point_index_correction*3+2];
        std::cout << "? "<< x_corr << " " << y_corr << " " << z_corr << std::endl;
    }

    if(mode == 0)
    {
        this->getNNFromIndex(kd_pos, nn, k);
    }else if(mode == 1)
    {
        this->getNNFromIndex2(kd_pos, nn, k);
    }

}

unsigned int CudaSurface::getKdTreePosition(float x, float y, float z)
{
    bool debug = false;
    if(debug)
        std::cout << "search for query point: "<< x << "|" << y << "|" << z << std::endl;

     unsigned int pos = 0;
    int current_dim;


    while(pos < this->kd_tree_splits->width)
    {
        current_dim = static_cast<int>(this->kd_tree_splits->elements[pos]);

        if(debug)
        {
            std::cout << "Current dim: " << current_dim << std::endl;
            std::cout << "KD-POS: " << pos << ", ";
        }

        if(current_dim == 0)
        {
            if(debug)
            {
                std::cout << "Current Dim: x, ";
                std::cout << "Split: " << this->kd_tree_values->elements[pos] << ", ";
            }

            if(x <= this->kd_tree_values->elements[pos] )
            {
                if(debug)
                {
                    std::cout << "LEFT";
                }
                pos = pos*2+1;
            } else {
                if(debug)
                {
                    std::cout << "RIGHT";
                }

                pos = pos*2+2;
            }
        } else if(current_dim == 1) {
            if(debug)
            {
                std::cout << "Current Dim: y, ";
                std::cout << "Split: " << this->kd_tree_values->elements[pos] << ", ";
            }
            if(y <= this->kd_tree_values->elements[pos] ){
                if(debug){
                    std::cout << "LEFT";
                }
                pos = pos*2+1;
            }else{
                if(debug)
                {
                    std::cout << "RIGHT";
                }
                pos = pos*2+2;
            }
        } else {
            if(debug)
            {
                std::cout << "Current Dim: z, ";
                std::cout << "Split: " << this->kd_tree_values->elements[pos] << ", ";
            }
            if(z <= this->kd_tree_values->elements[pos] ){
                if(debug)
                {
                    std::cout << "LEFT";
                }
                pos = pos*2+1;
            }else{
                if(debug)
                {
                    std::cout << "RIGHT";
                }
                pos = pos*2+2;
            }
        }
        if(debug)
        {
            std::cout << std::endl;
        }
    }
    std::cout << "Found index: " << this->kd_tree_values->elements[pos] << std::endl;
    return pos;
}

void CudaSurface::getNNFromIndex2(const unsigned int& kd_pos, unsigned int *nn, int k)
{
     unsigned int start = kd_pos - k/2;
     unsigned int end = kd_pos + (k+1)/2;

     if(end > this->kd_tree_values->width)
     {
         unsigned int diff = end - this->kd_tree_values->width;
         end = this->kd_tree_values->width;
         start -= diff;
     }

     if(start < this->kd_tree_splits->width)
     {
         unsigned int diff = this->kd_tree_splits->width - start;
         start = this->kd_tree_splits->width;
         end += diff;
     }

     for(int i=0; start < end; start++, i++)
     {
         nn[i] = static_cast<int>(this->kd_tree_values->elements[ start ] + 0.5 );
     }
}

void CudaSurface::getNNFromIndex(const unsigned int& kd_pos, unsigned int *nn, int k)
{
     unsigned int subtree_pos = kd_pos;
    int i;
    for(i=1; i<(k+1) && subtree_pos>0; i*=2) {
            subtree_pos = static_cast<unsigned int>((subtree_pos  - 1) / 2);
    }

    int iterator = subtree_pos;
    int max_nodes = 1;
    bool leaf_reached = false;


    nn[0] = static_cast<int>(this->kd_tree_values->elements[kd_pos]+0.5);
    int i_nn = 1;

    // like width search
    // go kd-tree up until max_nodes(leaf_nodes of subtree) bigger than needed nodes k
    // iterator = iterator * 2 + 1 -> go to
    for( ; iterator < this->kd_tree_values->width; iterator = iterator * 2 + 1, max_nodes *= 2)
    {
    // collect nodes from current height
        for( int i=0; i < max_nodes && iterator + i < this->kd_tree_values->width; i++)
        {
            int current_pos = iterator + i;
            int leaf_value  = (int)(this->kd_tree_values->elements[ current_pos ] + 0.5 );

            if( leaf_reached && i_nn < k  )
            {
                if( current_pos != kd_pos )
                {
                    nn[i_nn++] = leaf_value;
                }
            } else if( current_pos * 2 + 1 >= this->kd_tree_values->width )
            {
                //first leaf reache
                leaf_reached = true;
                if( leaf_value != kd_pos && i_nn < k )
                {
                    nn[i_nn++] = leaf_value;
                }
            }
        }
    }

}

} /* namespace lvr2 */
