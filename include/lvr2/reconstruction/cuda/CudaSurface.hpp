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
 * CudaSurface.h
 *
 * @author Alexander Mock
 */

#ifndef __CudaSurface_H
#define __CudaSurface_H

#include "lvr2/reconstruction/QueryPoint.hpp"
#include "lvr2/reconstruction/LBKdTree.hpp"
#include "lvr2/geometry/LBPointArray.hpp"
#include "lvr2/geometry/ColorVertex.hpp"
#include "lvr2/geometry/BaseVector.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <driver_types.h>

#include <boost/shared_array.hpp>

namespace lvr2
{

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

using Vec = BaseVector<float>;
typedef boost::shared_array<float> floatArr;
typedef ColorVertex<float, unsigned char> cVertex ;
typedef QueryPoint<Vec> QueryPointC;


class CudaSurface {

public:
    /**
     * @brief Constructor.
     *
     * @param points Input Pointcloud for kd-tree construction
     */
    CudaSurface(LBPointArray<float>& points, int device = 0);

    CudaSurface(floatArr& points, size_t num_points, int device=0 );

    ~CudaSurface();

    /**
     * @brief Starts calculation the normals on GPU
     *
     */
    void calculateNormals();

    void interpolateNormals();

    /**
     * @brief Get the resulting normals of the normal calculation. After calling "start".
     *
     * @param output_normals     PointArray as return value
     */
    void getNormals(LBPointArray<float>& output_normals);

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
    void setMethod(std::string& method);

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
    //~ Hostfunctions
    void init();

    void printSettings();

    void getCudaInformation(int device);

    void calculateBlocksThreads(int n, int elements, int element_size,
            int max_mem_shared, int max_threads_per_block,
            int& out_blocks_per_grid, int& out_threads_per_block, int& needed_shared_memory);

    template <typename T>
    void generateDevicePointArray(LBPointArray<T>& D_m, int width, int dim);

    template <typename T>
    void copyToDevicePointArray(LBPointArray<T>* m, LBPointArray<T>& D_m);

    template <typename T>
    void copyToDevicePointArray(LBPointArray<T>& m, LBPointArray<T>& D_m);


    void copyToHostPointArray(LBPointArray<float>& D_m, LBPointArray<float>* m);

    // Divice Function
    void GPU_NN();

    void initKdTree();

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
    int m_device;
    int m_mps;
    int m_threads_per_mp;
    int m_threads_per_block;
    int* m_size_thread_block;
    int* m_size_grid;
    unsigned long long m_device_global_memory;

    LBPointArray<float> D_V;
    LBPointArray<float> D_kd_tree_values;
    LBPointArray<unsigned char> D_kd_tree_splits;
    LBPointArray<float> D_Normals;

};

} /* namespace lvr2 */

#endif // !__CudaSurface_H
