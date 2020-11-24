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

#include "lvr2/reconstruction/opencl/ClStatisticalOutlierFilter.hpp"
#include "lvr2/config/lvropenmp.hpp"

namespace lvr2
{

ClSOR::ClSOR(floatArr& points, size_t num_points, int k, int device) : m_k(k)
{
//    this->init();

    this->getDeviceInformation(0, device);

    this->initCl();

    this->V.dim = 3;

    this->V.width = static_cast<unsigned int>(num_points);

    //mallocPointArray(V);

    this->V.elements = points.get();

    this->initKdTree();


}

ClSOR::~ClSOR()
{
//    this->finalizeCl();
    // free(this->V.elements);
    free(this->m_distances.elements);
}

void ClSOR::calcDistances()
{
    // Allocate Result Memory
    // Is a plain malloc good enough TODO
    generatePointArray(this->m_distances, this->V.width, 1);
    

    // std::cout << "Allocate GPU Memory" << std::endl;
    // tree and points and result normals to GPU
    D_V = clCreateBuffer(m_context, CL_MEM_READ_WRITE,
            this->V.width * this->V.dim * sizeof(float), NULL, &m_ret);
    D_kd_tree_values = clCreateBuffer(m_context, CL_MEM_READ_WRITE,
            this->kd_tree_values->width * this->kd_tree_values->dim * sizeof(float), NULL, &m_ret);
    D_kd_tree_splits = clCreateBuffer(m_context, CL_MEM_READ_WRITE,
            this->kd_tree_splits->width * this->kd_tree_splits->dim * sizeof(unsigned char),
            NULL, &m_ret);
    D_Distances = clCreateBuffer(m_context, CL_MEM_READ_WRITE,
            this->V.width * this->V.dim * sizeof(float), NULL, &m_ret);

    // std::cout << "Copy Points and Kd Tree to Gpu Memory" << std::endl;
    /* Copy input data to memory buffer */
    m_ret = clEnqueueWriteBuffer(m_command_queue, D_V, CL_TRUE, 0,
            this->V.width * this->V.dim * sizeof(float), V.elements, 0, NULL, NULL);
    m_ret |= clEnqueueWriteBuffer(m_command_queue, D_kd_tree_values, CL_TRUE, 0,
            this->kd_tree_values->width * this->kd_tree_values->dim * sizeof(float),
            this->kd_tree_values->elements, 0, NULL, NULL);
    m_ret |= clEnqueueWriteBuffer(m_command_queue, D_kd_tree_splits, CL_TRUE, 0,
            this->kd_tree_splits->width * this->kd_tree_splits->dim * sizeof(unsigned char),
            this->kd_tree_splits->elements, 0, NULL, NULL);

    if(m_ret != CL_SUCCESS)
        std::cerr << getErrorString(m_ret) << std::endl;

    // KNNKernel


    // unsigned int threadsPerBlock = this->m_threads_per_block;
    unsigned int warpSize = 32;
    //unsigned int threadsPerBlock = 16384;
    unsigned int threadsPerBlock = this->m_threads_per_block;
    //unsigned int blocksPerGrid = ( (V.width + threadsPerBlock-1) / threadsPerBlock) / warpSize;

    size_t local_item_size = static_cast<size_t>(warpSize);
    size_t global_item_size = static_cast<size_t>(threadsPerBlock);
    //size_t global_group_size = static_cast<size_t>(blocksPerGrid);

    // std::cout << "Set Kernel Arguments: Normal Estimation" << std::endl;

    m_ret = clSetKernelArg(m_kernel_sor, 0, sizeof(cl_mem), (void *)&D_V);
    m_ret |= clSetKernelArg(m_kernel_sor, 1, sizeof(unsigned int), &V.width );
    m_ret |= clSetKernelArg(m_kernel_sor, 2, sizeof(cl_mem),
            (void *)&D_kd_tree_values);
    m_ret |= clSetKernelArg(m_kernel_sor, 3, sizeof(unsigned int),
            &kd_tree_values->width );
    m_ret |= clSetKernelArg(m_kernel_sor, 4, sizeof(cl_mem),
            (void *)&D_kd_tree_splits);
    m_ret |= clSetKernelArg(m_kernel_sor, 5, sizeof(unsigned int),
            &kd_tree_splits->width );
    m_ret |= clSetKernelArg(m_kernel_sor, 6, sizeof(cl_mem), (void *)&D_Distances);
//    m_ret |= clSetKernelArg(m_kernel_sor, 7, sizeof(unsigned int), &V.width );
    m_ret |= clSetKernelArg(m_kernel_sor, 7, sizeof(unsigned int), &this->m_k);


    if(m_ret != CL_SUCCESS)
        std::cerr << getErrorString(m_ret) << std::endl;

    // std::cout << "Start Normal Estimation Kernel" << std::endl;
    // std::cout << "local_item_size: "<< local_item_size << std::endl;
    //std::cout << "global_item_size: " << global_item_size << std::endl;

    m_ret = clEnqueueNDRangeKernel(m_command_queue, m_kernel_sor, 1, NULL,
        &global_item_size, &local_item_size, 0, NULL, NULL);


    if(m_ret != CL_SUCCESS)
        std::cerr << getErrorString(m_ret) << std::endl;

    // Normals back to host
    m_ret = clEnqueueReadBuffer(m_command_queue, D_Distances, CL_TRUE, 0,
            this->m_distances.width * this->m_distances.dim * sizeof(float),
            this->m_distances.elements, 0, NULL, NULL);

    if(m_ret != CL_SUCCESS)
        std::cerr << getErrorString(m_ret) << std::endl;

}

void ClSOR::calcStatistics()
{
    m_mean_ = 0.0;
    m_std_dev_ = 0.0;

    double sum = 0.0;
    double sq_sum = 0.0;
    // TODO PARALLELIZE VARIANCE AND MEAN
    for(int i = 0; i < (this->m_distances.dim * this->m_distances.width); i++)
    {
        sum += static_cast<double>(this->m_distances.elements[i]);
        sq_sum += static_cast<double>(this->m_distances.elements[i]) * static_cast<double>(this->m_distances.elements[i]);
    }
    // does this need casts
    std::cout  << "sum " << sum << "sq_sum " << sq_sum << std::endl;
    m_mean_ = sum/static_cast<double>(m_distances.width);
    m_std_dev_ = (sq_sum - sum * sum / static_cast<double>(this->m_distances.width)) / static_cast<double>((this->m_distances.width - 1));
    m_std_dev_ = std::sqrt(m_std_dev_);
    std::cout  << "Mean " << m_mean_ << "dev " << m_std_dev_ << std::endl;
}

int ClSOR::getInliers(lvr2::indexArray& inliers)
{
    int j = 0;
    for(int i = 0; i < (this->m_distances.dim * this->m_distances.width); i++)
    {
        if(this->m_distances.elements[i] <= (m_mean_ + m_mult_ * m_std_dev_))
        {
            inliers[j] = i;
            j++;
        }
    }
    return j;
}

void ClSOR::setK(int k)
{
    this->m_k = k;
}




void ClSOR::freeGPU()
{

}

/// PRIVATE ///

//void ClSOR::init(){
//    // set default k
//    this->m_k = 10;
//
//    // set default ki
//    this->m_ki = 10;
//    this->m_kd = 5;
//
//    // set default flippoint
//    this->m_vx = 1000000.0;
//    this->m_vy = 1000000.0;
//    this->m_vz = 1000000.0;
//
//    this->m_calc_method = 0;
//
//    this->m_reconstruction_mode = false;
//}

void ClSOR::initKdTree() {

    kd_tree_gen = boost::shared_ptr<LBKdTree>(new LBKdTree(this->V, OpenMPConfig::getNumThreads() ) );
    this->kd_tree_values = kd_tree_gen->getKdTreeValues().get();
    this->kd_tree_splits = kd_tree_gen->getKdTreeSplits().get();

}

void ClSOR::initCl()
{

    this->m_context = clCreateContext(NULL, 1, &this->m_device_id, NULL, NULL, &this->m_ret);
    if(m_ret != CL_SUCCESS)
        std::cerr << getErrorString(m_ret) << std::endl;

    this->m_command_queue = clCreateCommandQueue(this->m_context, this->m_device_id, 0, &this->m_ret);
    if(m_ret != CL_SUCCESS)
        std::cerr << getErrorString(m_ret) << std::endl;

    this->loadSORKernel();
}

void ClSOR::finalizeCl()
{
    m_ret = clFlush(m_command_queue);
    m_ret = clFinish(m_command_queue);
    m_ret = clReleaseKernel(m_kernel_sor);

    m_ret = clReleaseProgram(m_program_es);
    m_ret = clReleaseProgram(m_program_in);

    m_ret = clReleaseMemObject(D_V);
    m_ret = clReleaseMemObject(D_kd_tree_values);
    m_ret = clReleaseMemObject(D_kd_tree_splits);
    m_ret = clReleaseMemObject(D_Distances);

    m_ret = clReleaseCommandQueue(m_command_queue);
    m_ret = clReleaseContext(m_context);

}

void ClSOR::loadSORKernel()
{
    // std::cout << "Loading estimation Kernel ..." << std::endl;

    // create program
    m_program_es = clCreateProgramWithSource(m_context, 1,
            (const char **) &SOR_KERNEL_STRING , NULL, &m_ret);
    if(m_ret != CL_SUCCESS)
    {
        std::cerr << "ClSOR::loadKernel() - Create Program " << getErrorString(m_ret) << std::endl;
    }

    if (!m_program_es)
    {
        printf("Error: Failed to create compute program!\n");
        exit(1);
    }

    // Build the program executable
    //
    m_ret = clBuildProgram(m_program_es, 0, NULL, NULL, NULL, NULL);
    if (m_ret != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(m_program_es, m_device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
                buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // create kernels
    m_kernel_sor = clCreateKernel(m_program_es, "SORKernel", &m_ret);
    if(m_ret != CL_SUCCESS)
    {
        std::cerr << "ClSOR::loadKernel() - Estimation " << getErrorString(m_ret) << std::endl;
        exit(1);
    }

}

const char *ClSOR::getErrorString(cl_int error)
{
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
        }
}


void ClSOR::getDeviceInformation(int platform_id, int device_id)
{

    char buffer[1024];

    cl_uint num_platforms;
    checkOclErrors(clGetPlatformIDs(0, NULL, &num_platforms));

    if(platform_id >= num_platforms)
    {
        std::cerr << "Wrong platform id " << std::endl;
        exit(1);
    }
//    printf("%d PLATFORMS FOUND\n", num_platforms);
    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    checkOclErrors(clGetPlatformIDs(num_platforms, platforms, NULL));

    cl_platform_id platform = platforms[platform_id];
    this->m_platform_id = platform;
    //printf("CL_PLATFORM: %d\n", k);
    checkOclErrors(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL));
    // printf("CL_PLATFORM_NAME: %s\n", buffer);
    checkOclErrors(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(buffer), buffer, NULL));
    // printf("CL_PLATFORM_VENDOR: %s\n", buffer);
    checkOclErrors(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL));
    // printf("CL_PLATFORM_VERSION: %s\n", buffer);
    checkOclErrors(clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, sizeof(buffer), buffer, NULL));
    // printf("CL_PLATFORM_PROFILE: %s\n", buffer);
    checkOclErrors(clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, sizeof(buffer), buffer, NULL));
    // printf("CL_PLATFORM_EXTENSIONS: %s\n", buffer);
    // printf("\n");

    cl_uint num_devices;
    checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));
//        printf("%d DEVICES FOUND\n", num_devices);
    if(device_id >= num_devices)
    {
        std::cerr << "Wrong device id " << std::endl;
        exit(1);
    }

    cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
    checkOclErrors(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL));

    cl_device_id device = devices[device_id];
    // std::cout << "Device: " << device << std::endl;
    this->m_device_id = device;
    // printf("CL_DEVICE: %d\n", j);
    cl_device_type type;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL));
    // if (type & CL_DEVICE_TYPE_DEFAULT) printf("CL_DEVICE_TYPE: %s\n", "CL_DEVICE_TYPE_DEFAULT");
    // if (type & CL_DEVICE_TYPE_CPU) printf("CL_DEVICE_TYPE: %s\n", "CL_DEVICE_TYPE_CPU");
    // if (type & CL_DEVICE_TYPE_GPU) printf("CL_DEVICE_TYPE: %s\n", "CL_DEVICE_TYPE_GPU");
    // if (type & CL_DEVICE_TYPE_CUSTOM) printf("CL_DEVICE_TYPE: %s\n", "CL_DEVICE_TYPE_CUSTOM");
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
    // printf("CL_DEVICE_NAME: %s\n", buffer);


    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
    // printf("CL_DEVICE_VENDOR: %s\n", buffer);
    cl_uint vendor_id;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id, NULL));
    // printf("CL_DEVICE_VENDOR_ID: %d\n", vendor_id);
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
    // printf("CL_DEVICE_VERSION: %s\n", buffer);
    checkOclErrors(clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));
    // printf("CL_DRIVER_VERSION: %s\n", buffer);
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(buffer), buffer, NULL));
    // printf("CL_DEVICE_OPENCL_C_VERSION: %s\n", buffer);
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(buffer), buffer, NULL));
    // printf("CL_DEVICE_PROFILE: %s\n", buffer);
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(buffer), buffer, NULL));
    // printf("CL_DEVICE_EXTENSIONS: %s\n", buffer);
    cl_uint max_compute_units;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(max_compute_units), &max_compute_units, NULL));
    this->m_mps = max_compute_units;

    // printf("CL_DEVICE_MAX_COMPUTE_UNITS: %u\n", max_compute_units);
    cl_uint max_work_item_dimensions;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                sizeof(max_work_item_dimensions), &max_work_item_dimensions, NULL));
    // printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: %u\n", max_work_item_dimensions);
    size_t* max_work_item_sizes = (size_t*)malloc(sizeof(size_t) * max_work_item_dimensions);
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                sizeof(size_t) * max_work_item_dimensions, max_work_item_sizes, NULL));
    this->m_threads_per_block = max_work_item_sizes[0];
    free(max_work_item_sizes);
    size_t max_work_group_size;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                sizeof(max_work_group_size), &max_work_group_size, NULL));
    // printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %lu\n", max_work_group_size);

    // ?
    //this->m_threads_per_block = max_work_group_size;


    cl_uint preferred_vector_width_char;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                sizeof(preferred_vector_width_char), &preferred_vector_width_char, NULL));
    // printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: %u\n", preferred_vector_width_char);
    cl_uint preferred_vector_width_short;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                sizeof(preferred_vector_width_short), &preferred_vector_width_short, NULL));
    // printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: %u\n", preferred_vector_width_short);
    cl_uint preferred_vector_width_int;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                sizeof(preferred_vector_width_int), &preferred_vector_width_int, NULL));
    // printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: %u\n", preferred_vector_width_int);
    cl_uint preferred_vector_width_long;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                sizeof(preferred_vector_width_long), &preferred_vector_width_long, NULL));
    // printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: %u\n", preferred_vector_width_long);
    cl_uint preferred_vector_width_float;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                sizeof(preferred_vector_width_float), &preferred_vector_width_float, NULL));
    // printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: %u\n", preferred_vector_width_float);
    cl_uint preferred_vector_width_double;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                sizeof(preferred_vector_width_double), &preferred_vector_width_double, NULL));
    // printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: %u\n", preferred_vector_width_double);
    cl_uint preferred_vector_width_half;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
                sizeof(preferred_vector_width_half), &preferred_vector_width_half, NULL));
    // printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF: %u\n", preferred_vector_width_half);
    cl_uint native_vector_width_char;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
                sizeof(native_vector_width_char), &native_vector_width_char, NULL));
    // printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR: %u\n", native_vector_width_char);
    cl_uint native_vector_width_short;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
                sizeof(native_vector_width_short), &native_vector_width_short, NULL));
    // printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT: %u\n", native_vector_width_short);
    cl_uint native_vector_width_int;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
                sizeof(native_vector_width_int), &native_vector_width_int, NULL));
    // printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_INT: %u\n", native_vector_width_int);
    cl_uint native_vector_width_long;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
                sizeof(native_vector_width_long), &native_vector_width_long, NULL));
    // printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG: %u\n", native_vector_width_long);
    cl_uint native_vector_width_float;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
                sizeof(native_vector_width_float), &native_vector_width_float, NULL));
    // printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT: %u\n", native_vector_width_float);
    cl_uint native_vector_width_double;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
                sizeof(native_vector_width_double), &native_vector_width_double, NULL));
    // printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE: %u\n", native_vector_width_double);
    cl_uint native_vector_width_half;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
                sizeof(native_vector_width_half), &native_vector_width_half, NULL));
    // printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF: %u\n", native_vector_width_half);
    cl_uint max_clock_frequency;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                sizeof(max_clock_frequency), &max_clock_frequency, NULL));
    // printf("CL_DEVICE_MAX_CLOCK_FREQUENCY: %u MHz\n", max_clock_frequency);
    cl_uint address_bits;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS,
                sizeof(address_bits), &address_bits, NULL));
    // printf("CL_DEVICE_ADDRESS_BITS: %u\n", address_bits);
    cl_ulong max_mem_alloc_size;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL));
    cl_bool image_support;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support),
                &image_support, NULL));
    // printf("CL_DEVICE_IMAGE_SUPPORT: %u\n", image_support);
    size_t max_parameter_size;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_MAX_PARAMETER_SIZE,
                sizeof(max_parameter_size), &max_parameter_size, NULL));
    // printf("CL_DEVICE_MAX_PARAMETER_SIZE: %lu B\n", max_parameter_size);
    cl_device_mem_cache_type global_mem_cache_type;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                sizeof(global_mem_cache_type), &global_mem_cache_type, NULL));
    cl_uint global_mem_cacheline_size;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                sizeof(global_mem_cacheline_size), &global_mem_cacheline_size, NULL));
    cl_ulong global_mem_cache_size;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                sizeof(global_mem_cache_size), &global_mem_cache_size, NULL));
    cl_ulong global_mem_size;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                sizeof(global_mem_size), &global_mem_size, NULL));
    this->m_device_global_memory = global_mem_size;

    cl_ulong max_constant_buffer_size;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                sizeof(max_constant_buffer_size), &max_constant_buffer_size, NULL));
    cl_uint max_constant_args;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(max_constant_args),
                &max_constant_args, NULL));
    // printf("CL_DEVICE_MAX_CONSTANT_ARGS: %u\n", max_constant_args);
    cl_device_local_mem_type local_mem_type;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type),
                &local_mem_type, NULL));
    // if (local_mem_type == CL_NONE) printf("CL_DEVICE_LOCAL_MEM_TYPE: %s\n", "CL_NONE");
    // if (local_mem_type == CL_LOCAL) printf("CL_DEVICE_LOCAL_MEM_TYPE: %s\n", "CL_LOCAL");
    // if (local_mem_type == CL_GLOBAL) printf("CL_DEVICE_LOCAL_MEM_TYPE: %s\n", "CL_GLOBAL");
    cl_ulong local_mem_size;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size),
                &local_mem_size, NULL));
    // printf("CL_DEVICE_LOCAL_MEM_SIZE: %lu B = %lu KB\n", local_mem_size, local_mem_size / 1024);
    cl_bool error_correction_support;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT,
                sizeof(error_correction_support), &error_correction_support, NULL));
    // printf("CL_DEVICE_ERROR_CORRECTION_SUPPORT: %u\n", error_correction_support);
    cl_bool host_unified_memory;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY,
                sizeof(host_unified_memory), &host_unified_memory, NULL));
    // printf("CL_DEVICE_HOST_UNIFIED_MEMORY: %u\n", host_unified_memory);
    size_t profiling_timer_resolution;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                sizeof(profiling_timer_resolution), &profiling_timer_resolution, NULL));
    // printf("CL_DEVICE_PROFILING_TIMER_RESOLUTION: %lu ns\n", profiling_timer_resolution);
    cl_bool endian_little;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, sizeof(endian_little),
                &endian_little, NULL));
    // printf("CL_DEVICE_ENDIAN_LITTLE: %u\n", endian_little);
    cl_bool available;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(available), &available, NULL));
    // printf("CL_DEVICE_AVAILABLE: %u\n", available);
    cl_bool compier_available;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE,
                sizeof(compier_available), &compier_available, NULL));
    // printf("CL_DEVICE_COMPILER_AVAILABLE: %u\n", compier_available);
    cl_bool linker_available;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_LINKER_AVAILABLE, sizeof(linker_available),
                &linker_available, NULL));
    // printf("CL_DEVICE_LINKER_AVAILABLE: %u\n", linker_available);
    cl_device_exec_capabilities exec_capabilities;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_EXECUTION_CAPABILITIES,
                sizeof(exec_capabilities), &exec_capabilities, NULL));
    cl_command_queue_properties queue_properties;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties),
                &queue_properties, NULL));
    size_t printf_buffer_size;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_PRINTF_BUFFER_SIZE,
                sizeof(printf_buffer_size), &printf_buffer_size, NULL));
    cl_bool preferred_interop_user_sync;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,
                sizeof(preferred_interop_user_sync), &preferred_interop_user_sync, NULL));
    // printf("CL_DEVICE_PREFERRED_INTEROP_USER_SYNC: %u\n", preferred_interop_user_sync);
//            cl_device_id parent_device;
//            printf("CL_DEVICE_PARENT_DEVICE: %u\n", parent_device);
    cl_uint reference_count;
    checkOclErrors(clGetDeviceInfo(device, CL_DEVICE_REFERENCE_COUNT, sizeof(reference_count),
                &reference_count, NULL));
    // printf("CL_DEVICE_REFERENCE_COUNT: %u\n", reference_count);
    // printf("\n");

    free(devices);

    free(platforms);

}

} /* namespace lvr2 */
