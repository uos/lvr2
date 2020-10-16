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

#ifndef __CLSOR_H
#define __CLSOR_H

#include "lvr2/reconstruction/QueryPoint.hpp"
#include "lvr2/reconstruction/LBKdTree.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/LBPointArray.hpp"
#include "lvr2/io/DataStruct.hpp"

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

class ClSOR {
public:
    ClSOR(floatArr& points, size_t num_points, int k, int device = 0);
    ~ClSOR();

    
    /**
     * @brief Set the number of k nearest neighbors
     *        k-neighborhood
     *
     * @param k             The size of the used k-neighborhood
     *
     */
    void setK(int k);
    void setMult(float std_dev_mult) { m_mult_ = std_dev_mult;};

    void calcDistances();
    void calcStatistics();
    
    int getInliers(lvr2::indexArray& inliers);

    void freeGPU();

private:

    void init();

    const char *getErrorString(cl_int error);

    void initKdTree();

    void getDeviceInformation(int platform_id=0, int device_id=0);

    void loadSORKernel();

    void initCl();

    void finalizeCl();
    
    //
    int m_k;
    double m_mult_;
    double m_mean_;
    double m_std_dev_;
    // V->points and normals
    LBPointArray<float> V;
    LBPointArray<float>* kd_tree_values;
    LBPointArray<unsigned char>* kd_tree_splits;

//    LBPointArray<float> Result_Normals;
    LBPointArray<float> m_distances;
    boost::shared_ptr<LBKdTree> kd_tree_gen;




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
    cl_kernel m_kernel_sor;
//    cl_kernel m_kernel_normal_interpolation;

    cl_mem D_V;
    cl_mem D_kd_tree_values;
    cl_mem D_kd_tree_splits;
    cl_mem D_Distances;


const char *SOR_KERNEL_STRING = "\n"
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
"__kernel void SORKernel(__global const float* D_V, const unsigned unsigned int num_points,"
"__global const float* D_kd_tree_values, const unsigned int num_values,"
"__global const unsigned char* D_kd_tree_splits , const unsigned int num_splits,"
"__global float* D_distances, const unsigned int k) \n"
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
"            float mean     = 0.0; \n"
"            unsigned int j = 0; \n"
"            for(unsigned int i = start; i < end && i<num_values; i++ ) \n"
"            { \n"
"                if(i != pos) \n"
"                { \n"
"                    nearest_index = (unsigned int)(D_kd_tree_values[i]+ 0.5); \n"
"                    if(nearest_index < num_points) \n"
"                    { \n"
"                        "
"                        // calculate distances and mean \n"
"                        float diff_x = D_V[ nearest_index * 3 + 0 ] - vertex_x; \n"
"                        float diff_y = D_V[ nearest_index * 3 + 1 ] - vertex_y; \n"
"                        float diff_z = D_V[ nearest_index * 3 + 2 ] - vertex_z; \n"
"                        float dist  = sqrt((pow(diff_x, 2) + pow(diff_y, 2) + pow(diff_z, 2)));\n"
"                        mean += dist; \n"
"                        j++;\n"
"                    } \n"
"                } \n"
"            } \n"
"           D_distances[tid] = mean/k;"
"        } \n"
"    } \n"
"} \n";


};

} /* namespace lvr2 */

#endif // !__ClSOR_H
