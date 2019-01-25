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
    bool success = false;

    // TODO

    return success;
}

template <typename BaseVecT>
std::vector<bool> BVHRaycaster<BaseVecT>::castRays(
    const Point<BaseVecT>& origin,
    const std::vector<Vector<BaseVecT> >& directions,
    std::vector<Point<BaseVecT> >& intersections
) const
{
    // Cast multiple rays from one origin
    std::vector<bool> hits;
    // TODO
    return hits;
}

template <typename BaseVecT>
std::vector<bool> BVHRaycaster<BaseVecT>::castRays(
    const std::vector<Point<BaseVecT> >& origins,
    const std::vector<Vector<BaseVecT> >& directions,
    std::vector<Point<BaseVecT> >& intersections
) const
{
    // Cast multiple rays from multiple origins
    std::vector<bool> hits;
    // TODO
    return hits;
}

} // namespace lvr2