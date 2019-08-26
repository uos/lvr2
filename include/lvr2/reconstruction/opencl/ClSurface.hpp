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

#ifndef __ClSurface_H
#define __ClSurface_H

#include "lvr2/reconstruction/QueryPoint.hpp"
#include "lvr2/reconstruction/LBKdTree.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/LBPointArray.hpp"

#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif
#include "lvr2/reconstruction/opencl/cl_helper.h"

#define MAX_SOURCE_SIZE (0x1024)

namespace lvr2
{

typedef boost::shared_array<float> floatArr;

using Vec = BaseVector<float>;
typedef QueryPoint<Vec> QueryPointC;

class ClSurface {
public:
    ClSurface(floatArr& points, size_t num_points, int device = 0);
    ~ClSurface();

    /**
    * @brief Starts calculation the normals on GPU
    *
    */
    void calculateNormals();

    /**
     * @brief Get the resulting normals of the normal calculation. After calling "start".
     *
     * @param output_normals     PointArray as return value
     */
    void getNormals(floatArr output_normals);

    /**
     * @brief Set the number of k nearest neighbors
     *        k-neighborhood
     *
     * @param k             The size of the used k-neighborhood
     *
     */
    void setKn(int kn);

    /**
     * @brief Set the number of k nearest neighbors
     *        k-neighborhood for interpolation
     *
     * @param k             The size of the used k-neighborhood
     *
     */
    void setKi(int ki);

    /**
     * @brief Set the number of k nearest neighbors
     *        k-neighborhood for distance
     *
     * @param k             The size of the used k-neighborhood
     *
     */
    void setKd(int kd);

    /**
     * @brief Set the viewpoint to orientate the normals
     *
     * @param v_x     Coordinate X axis
     * @param v_y     Coordinate Y axis
     * @param v_z     Coordinate Z axis
     *
     */
    void setFlippoint(float v_x, float v_y, float v_z);

    /**
     * @brief Set Method for normal calculation
     *
     * @param method   "PCA","RANSAC"
     *
     */
    void setMethod(std::string method);

    /**
    * Reconstuction Mode:
    * Points stay in gpu until reconstruction is finished
    */
    void setReconstructionMode(bool mode = true);

    /**
    * TODO:
    *    Implement
    */
    void distances(std::vector<QueryPoint<Vec> >& query_points, float voxel_size);

    void freeGPU();

private:

    void init();

    const char *getErrorString(cl_int error);

    void initKdTree();

    void getDeviceInformation(int platform_id=0, int device_id=0);

    void loadEstimationKernel();

    void loadInterpolationKernel();

    void initCl();

    void finalizeCl();

    // V->points and normals
    LBPointArray<float> V;
    LBPointArray<float>* kd_tree_values;
    LBPointArray<unsigned char>* kd_tree_splits;

    LBPointArray<float> Result_Normals;
    boost::shared_ptr<LBKdTree> kd_tree_gen;

    float m_vx, m_vy, m_vz;
    int m_k, m_ki, m_kd;


    int m_calc_method;
    bool m_reconstruction_mode;

    // Device Information
    cl_platform_id m_platform_id;
    cl_device_id m_device_id;
    cl_uint m_mps;
    cl_uint m_threads_per_block;
    cl_ulong m_device_global_memory;
    cl_int m_ret;
    cl_context m_context;
    cl_command_queue m_command_queue;
    cl_program m_program_es;
    cl_program m_program_in;
    cl_kernel m_kernel_normal_estimation;
    cl_kernel m_kernel_normal_interpolation;

    cl_mem D_V;
    cl_mem D_kd_tree_values;
    cl_mem D_kd_tree_splits;
    cl_mem D_Normals;


const char *NORMAL_ESTIMATION_KERNEL_STRING = "\n"
"unsigned int GetKdTreePosition(__global const float* D_kd_tree_values,"
"const unsigned int num_values,__global const unsigned char* D_kd_tree_splits,"
"const unsigned int num_splits, float x, float y, float z) \n"
"{ \n"
"    unsigned int pos = 0; \n"
"    unsigned int current_dim = 0; \n"
"    while(pos < num_splits) \n"
"    { \n"
"        current_dim = (unsigned int)(D_kd_tree_splits[pos]); \n"
"        if(current_dim == 0) \n"
"        { \n"
"            if(x <= D_kd_tree_values[pos] ) \n"
"            { \n"
"                pos = pos*2+1; \n"
"            } else { \n"
"                pos = pos*2+2; \n"
"            } \n"
"        } else if(current_dim == 1) { \n"
"            if(y <= D_kd_tree_values[pos] ){ \n"
"                pos = pos*2+1; \n"
"            }else{ \n"
"                pos = pos*2+2; \n"
"            } \n"
"        } else { \n"
"            if(z <= D_kd_tree_values[pos] ){ \n"
"                pos = pos*2+1; \n"
"            }else{ \n"
"                pos = pos*2+2; \n"
"            } \n"
"        } \n"
"    } \n"
"    return pos; \n"
"} \n"
" \n"
"__kernel void NormalEstimationKernel(__global const float* D_V, const unsigned int num_points,"
"__global const float* D_kd_tree_values, const unsigned int num_values,"
"__global const unsigned char* D_kd_tree_splits , const unsigned int num_splits,"
"__global float* D_Normals, const unsigned int num_pointnormals,const unsigned int k,"
"const float flip_x, const float flip_y, const float flip_z) \n"
"{ \n"
"    unsigned int loc_id = get_local_id(0); \n"
"    unsigned int loc_size = get_local_size(0); \n"
"    unsigned int glob_id = get_global_id(0); \n"
"    unsigned int glob_size = get_global_size(0); \n"
"    unsigned int group_id = get_group_id(0); \n"
"    unsigned int group_size = get_num_groups(0); \n"
"    unsigned int tid = glob_id; \n"
"    const unsigned int offset = glob_size; \n"
"    for(;tid < num_points; tid += offset) \n"
"    { \n"
"        unsigned int pos = GetKdTreePosition(D_kd_tree_values, "
"num_values,D_kd_tree_splits, num_splits,D_V[tid * 3], D_V[tid * 3 + 1], D_V[tid * 3 +2] ); \n"
"        unsigned int vertex_index = (unsigned int)(D_kd_tree_values[pos]+ 0.5); \n"
"        if(vertex_index < num_points) \n"
"        { \n"
"            float vertex_x = D_V[ vertex_index * 3 + 0 ]; \n"
"            float vertex_y = D_V[ vertex_index * 3 + 1 ]; \n"
"            float vertex_z = D_V[ vertex_index * 3 + 2 ]; \n"
"            unsigned int nearest_index; \n"
"            int start = pos-(k/2); \n"
"            int end = pos+((k+1)/2); \n"
"            int correct = 0; \n"
"            if(start < num_splits) \n"
"            { \n"
"                correct = num_splits - start; \n"
"            }else if(end > num_values) \n"
"            { \n"
"                correct = num_values - end; \n"
"            } \n"
"            start += correct; \n"
"            end += correct; \n"
"            float result_x = 0.0; \n"
"            float result_y = 0.0; \n"
"            float result_z = 0.0; \n"
"            float xx = 0.0; \n"
"            float xy = 0.0; \n"
"            float xz = 0.0; \n"
"            float yy = 0.0; \n"
"            float yz = 0.0; \n"
"            float zz = 0.0; \n"
"            for(unsigned int i = start; i < end && i<num_values; i++ ) \n"
"            { \n"
"                if(i != pos) \n"
"                { \n"
"                    nearest_index = (unsigned int)(D_kd_tree_values[i]+ 0.5); \n"
"                    if(nearest_index < num_points) \n"
"                    { \n"
"                        float rx = D_V[ nearest_index * 3 + 0 ] - vertex_x; \n"
"                        float ry = D_V[ nearest_index * 3 + 1 ] - vertex_y; \n"
"                        float rz = D_V[ nearest_index * 3 + 2 ] - vertex_z; \n"
"                        xx += rx * rx; \n"
"                        xy += rx * ry; \n"
"                        xz += rx * rz; \n"
"                        yy += ry * ry; \n"
"                        yz += ry * rz; \n"
"                        zz += rz * rz; \n"
"                    } \n"
"                } \n"
"            } \n"
"            float det_x = yy * zz - yz * yz; \n"
"            float det_y = xx * zz - xz * xz; \n"
"            float det_z = xx * yy - xy * xy; \n"
"            float dir_x; \n"
"            float dir_y; \n"
"            float dir_z; \n"
"            if( det_x >= det_y && det_x >= det_z) \n"
"            { \n"
"                dir_x = 1.0; \n"
"                dir_y = (xz * yz - xy * zz) / det_x; \n"
"                dir_z = (xy * yz - xz * yy) / det_x; \n"
"            } \n"
"            else if( det_y >= det_x && det_y >= det_z) \n"
"            { \n"
"                dir_x = (yz * xz - xy * zz) / det_y; \n"
"                dir_y = 1.0; \n"
"                dir_z = (xy * xz - yz * xx) / det_y; \n"
"            } \n"
"            else{ \n"
"                dir_x = (yz * xy - xz * yy ) / det_z; \n"
"                dir_y = (xz * xy - yz * xx ) / det_z; \n"
"                dir_z = 1.0; \n"
"            } \n"
"            float invnorm = 1/sqrt( dir_x * dir_x + dir_y * dir_y + dir_z * dir_z ); \n"
"            result_x = dir_x * invnorm; \n"
"            result_y = dir_y * invnorm; \n"
"            result_z = dir_z * invnorm; \n"
"            float x_dir = flip_x - vertex_x; \n"
"            float y_dir = flip_y - vertex_y; \n"
"            float z_dir = flip_z - vertex_z; \n"
"            float scalar = x_dir * result_x + y_dir * result_y + z_dir * result_z; \n"
"            if(scalar < 0) \n"
"            { \n"
"                result_x = -result_x; \n"
"                result_y = -result_y; \n"
"                result_z = -result_z; \n"
"            } \n"
"            D_Normals[tid * 3 ] = result_x; \n"
"            D_Normals[tid * 3 + 1 ] = result_y; \n"
"            D_Normals[tid * 3 + 2 ] = result_z; \n"
"        } \n"
"    } \n"
"} \n";

const char *NORMAL_INTERPOLATION_KERNEL_STRING = "\n"
"float getGaussianFactor(const unsigned int index, const unsigned int middle_i, "
"const unsigned int ki, const float norm) \n"
"{ \n"
"    float val = (float)(index); \n"
"    float middle = (float)(middle_i); \n"
"    float ki_2 = (float)(ki)/2.0; \n"
"    if(val > middle) \n"
"    { \n"
"        val = val - middle; \n"
"    }else{ \n"
"        val = middle - val; \n"
"    } \n"
"    if(val > ki_2) \n"
"    { \n"
"        return 0.0; \n"
"    }else{ \n"
"        float border_val = 0.2; \n"
"        float gaussian = 1.0 - pow((float)val/ki_2, (float)2.0) * (1.0-border_val); \n"
"        return gaussian * norm; \n"
"    } \n"
"} \n"
" \n"
"__kernel void NormalInterpolationKernel(__global float* D_kd_tree_values,"
"const unsigned int num_values, __global float* D_kd_tree_splits, "
"const unsigned int num_splits, __global float* D_Normals, "
"const unsigned int num_pointnormals, const unsigned int ki) \n"
"{ \n"
"    unsigned int loc_id = get_local_id(0); \n"
"    unsigned int loc_size = get_local_size(0); \n"
"    unsigned int glob_id = get_global_id(0); \n"
"    unsigned int glob_size = get_global_size(0); \n"
"    unsigned int group_id = get_group_id(0); \n"
"    unsigned int group_size = get_num_groups(0); \n"
"    unsigned int tid = glob_id; \n"
"    const unsigned int offset = glob_size; \n"
"    for(;tid < num_pointnormals; tid += offset) \n"
"    { \n"
"        int c = 0; \n"
"        unsigned int offset = num_splits; \n"
"        unsigned int query_index = (unsigned int)(D_kd_tree_values[offset + tid]+ 0.5); \n"
"        unsigned int nearest_index; \n"
"        float gaussian = 5.0; \n"
"        if(query_index < num_pointnormals) \n"
"        { \n"
"            float n_x = D_Normals[query_index * 3 + 0]; \n"
"            float n_y = D_Normals[query_index * 3 + 1]; \n"
"            float n_z = D_Normals[query_index * 3 + 2]; \n"
"            if(tid > 1) \n"
"            { \n"
"                for(unsigned int i = tid-1; i > 0 && c < ki/2; i--,c++ ) \n"
"                { \n"
"                    nearest_index = (unsigned int)(D_kd_tree_values[i + offset]+ 0.5); \n"
"                    if(nearest_index < num_pointnormals) \n"
"                    { \n"
"                        gaussian = getGaussianFactor(i, tid, ki, 5.0); \n"
"                        n_x += gaussian * D_Normals[nearest_index * 3 + 0]; \n"
"                        n_y += gaussian * D_Normals[nearest_index * 3 + 1]; \n"
"                        n_z += gaussian * D_Normals[nearest_index * 3 + 2]; \n"
"                    } \n"
"                } \n"
"            } \n"
"            if(tid < num_pointnormals-1) \n"
"            { \n"
"                for(unsigned int i = tid+1; i < num_pointnormals && c < ki; i++,c++ ) \n"
"                { \n"
"                    nearest_index = (unsigned int)(D_kd_tree_values[i + offset]+ 0.5); \n"
"                    if(nearest_index < num_pointnormals) \n"
"                    { \n"
"                        gaussian = getGaussianFactor(i, tid, ki, 5.0); \n"
"                        n_x += gaussian * D_Normals[nearest_index * 3 + 0]; \n"
"                        n_y += gaussian * D_Normals[nearest_index * 3 + 1]; \n"
"                        n_z += gaussian * D_Normals[nearest_index * 3 + 2]; \n"
"                    } \n"
"                } \n"
"            } \n"
"            float norm = sqrt(pow(n_x,2) + pow(n_y,2) + pow(n_z,2)); \n"
"            n_x = n_x/norm; \n"
"            n_y = n_y/norm; \n"
"            n_z = n_z/norm; \n"
"            D_Normals[query_index * 3 + 0] = n_x; \n"
"            D_Normals[query_index * 3 + 1] = n_y; \n"
"            D_Normals[query_index * 3 + 2] = n_z; \n"
"        } \n"
"    } \n"
"} \n";

};

} /* namespace lvr2 */

#endif // !__ClSurface_H
