namespace lvr2 {

template <typename BaseVecT>
BVHRaycaster<BaseVecT>::BVHRaycaster(const MeshBufferPtr mesh)
:RaycasterBase<BaseVecT>(mesh)
,m_bvh(mesh)
{
    
}

template <typename BaseVecT>
bool BVHRaycaster<BaseVecT>::castRay(
    const Point<BaseVecT>& origin,
    const Vector<BaseVecT>& direction,
    Point<BaseVecT>& intersection
) const
{
    // Cast one ray from one origin
    // TODO

    return false;
}

template <typename BaseVecT>
void BVHRaycaster<BaseVecT>::castRays(
    const Point<BaseVecT>& origin,
    const std::vector<Vector<BaseVecT> >& directions,
    std::vector<Point<BaseVecT> >& intersections,
    std::vector<uint8_t>& hits
) const
{
    // Cast multiple rays from one origin
    // TODO
}

template <typename BaseVecT>
void BVHRaycaster<BaseVecT>::castRays(
    const std::vector<Point<BaseVecT> >& origins,
    const std::vector<Vector<BaseVecT> >& directions,
    std::vector<Point<BaseVecT> >& intersections,
    std::vector<uint8_t>& hits
) const
{
    // Cast multiple rays from multiple origins
    // TODO
}

} // namespace lvr2