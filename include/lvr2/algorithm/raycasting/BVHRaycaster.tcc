namespace lvr2 {

template <typename BaseVecT>
BVHRaycaster<BaseVecT>::BVHRaycaster(const MeshBufferPtr mesh)
:RaycasterBase<BaseVecT>(mesh)
,m_bvh(mesh)
{
    
}

template <typename BaseVecT>
Point<BaseVecT> BVHRaycaster<BaseVecT>::castRay(
    const Point<BaseVecT>& origin,
    const Vector<BaseVecT>& direction
) const
{
    // Cast one ray from one origin

    Point<BaseVecT> dst = {0.0, 0.0, 0.0};

    // TODO

    return dst;
}

template <typename BaseVecT>
std::vector<Point<BaseVecT> > BVHRaycaster<BaseVecT>::castRays(
    const Point<BaseVecT>& origin,
    const std::vector<Vector<BaseVecT> >& directions
) const
{
    // Cast multiple rays from one origin
    std::vector<Point<BaseVecT> > dst;
    // TODO
    return dst;
}

template <typename BaseVecT>
std::vector<Point<BaseVecT> > BVHRaycaster<BaseVecT>::castRays(
    const std::vector<Point<BaseVecT> >& origins,
    const std::vector<Vector<BaseVecT> >& directions
) const
{
    // Cast multiple rays from multiple origins
    std::vector<Point<BaseVecT> > dst;
    // TODO
    return dst;
}

} // namespace lvr2