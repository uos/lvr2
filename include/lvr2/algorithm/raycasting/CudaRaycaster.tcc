namespace lvr2 {

template <typename BaseVecT>
CudaRaycaster<BaseVecT>::CudaRaycaster(const MeshBufferPtr mesh)
:RaycasterBase<BaseVecT>(mesh)
{
    
}

template <typename BaseVecT>
bool CudaRaycaster<BaseVecT>::castRay(
    const Point<BaseVecT>& origin,
    const Vector<BaseVecT>& direction,
    Point<BaseVecT>& intersection
)
{

    return false;
}

template <typename BaseVecT>
void CudaRaycaster<BaseVecT>::castRays(
    const Point<BaseVecT>& origin,
    const std::vector<Vector<BaseVecT> >& directions,
    std::vector<Point<BaseVecT> >& intersections,
    std::vector<uint8_t>& hits
)
{
    

}

template <typename BaseVecT>
void CudaRaycaster<BaseVecT>::castRays(
    const std::vector<Point<BaseVecT> >& origins,
    const std::vector<Vector<BaseVecT> >& directions,
    std::vector<Point<BaseVecT> >& intersections,
    std::vector<uint8_t>& hits
)
{
    
}

} // namespace lvr2