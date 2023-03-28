#include <iostream>

namespace lvr2
{

template<typename BaseVecT>
CudaKSearchSurface<BaseVecT>::CudaKSearchSurface()
{
    this->setKn(50);
}

template<typename BaseVecT>
CudaKSearchSurface<BaseVecT>::CudaKSearchSurface(
    PointBufferPtr pbuffer,
    size_t k
) : PointsetSurface<BaseVecT>(pbuffer), m_tree(32, true, true)
{
    this->setKn(k);

    floatArr points = this->m_pointBuffer->getPointArray();
    float* points_raw = &points[0];
    int num_points = this->m_pointBuffer->numPoints();

    this->m_tree.build(points_raw, num_points);
}

template<typename BaseVecT>
std::pair<typename BaseVecT::CoordType, typename BaseVecT::CoordType>
    CudaKSearchSurface<BaseVecT>::distance(BaseVecT p) const
{
    // Not implemented here
    throw std::runtime_error("Do not use this function!");
    return std::pair<typename BaseVecT::CoordType, typename BaseVecT::CoordType>(0.0, 0.0);
}

template<typename BaseVecT>
void CudaKSearchSurface<BaseVecT>::calculateSurfaceNormals()
{
    floatArr points = this->m_pointBuffer->getPointArray();
    
    float* points_raw = &points[0];
    int num_points = this->m_pointBuffer->numPoints();
    
    BaseVecT* query = reinterpret_cast<BaseVecT*>(points_raw);

    int K = this->m_kn;

    size_t size =  3 * num_points;

    // Get the queries
    size_t num_queries = num_points;

    float* queries = points_raw;

    // Create the normal array
    float* normals = (float*) malloc(sizeof(float) * num_queries * 3);

    // Use combined knn and normal calculation kernel
    this->m_tree.knn_normals(
        K,
        normals,
        num_queries
    );
   
    // Set the normals in the point buffer
    this->m_pointBuffer->setNormalArray(floatArr(&normals[0]), num_points);

}

} // namespace lvr2