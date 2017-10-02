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
 * CudaSurface.h
 *
 * @author Alexander Mock
 */

#ifndef __CudaSurface_H
#define __CudaSurface_H

#include "lvr/reconstruction/QueryPoint.hpp"
#include "lvr/reconstruction/LBKdTree.hpp"
#include "lvr/geometry/LBPointArray.hpp"
#include "lvr/geometry/ColorVertex.hpp"

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

namespace lvr {

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


//~ Device Functions
//~ __global__ void KNNKernel(const LBPointArray D_V, const LBPointArray D_kd_tree, LBPointArray D_Result_Normals, int k=50);

typedef boost::shared_array<float> floatArr;
typedef lvr::ColorVertex<float, unsigned char> cVertex ;
typedef lvr::QueryPoint<cVertex> QueryPointC;


class CudaSurface {

public:
    /**
     * @brief Constructor.
     *
     * @param points Input Pointcloud for kd-tree construction
     */
    CudaSurface(LBPointArray<float>& points);

    CudaSurface(floatArr& points, size_t num_points, size_t dim = 3);

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
    void distances(std::vector<lvr::QueryPoint<cVertex> >& query_points, float voxel_size);

    void freeGPU();

private:
    //~ Hostfunctions
    void init();

    void printSettings();

    void getCudaInformation();

    void calculateBlocksThreads(int n, int elements, int element_size, int max_mem_shared, int max_threads_per_block,
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

    //debugging - testing
    void debug(unsigned int query_index, int k);
    void debug2(unsigned int query_index, int k);


    void writeToPly(float* point, float* nn, int size, std::string filename);
    void writeToPly(float* point, unsigned int* nn_i, int size, std::string filename);
    void writeToPly(unsigned int point_i, unsigned int* nn_i, int size, std::string filename);

    void nearestNeighborsHost(float* point, unsigned int* nn, int k, int mode=0);

    unsigned int getKdTreePosition(float x, float y, float z);

    void getNNFromIndex(const unsigned int& kd_pos, unsigned int *nn, int k);
    void getNNFromIndex2(const unsigned int& kd_pos, unsigned int *nn, int k);


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

} /* namespace lvr */

#endif // !__CudaSurface_H