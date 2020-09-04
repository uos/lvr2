namespace lvr2
{

template<typename IntT>
RaycasterBase<IntT>::RaycasterBase(const MeshBufferPtr mesh)
:m_mesh(mesh)
{

}

template<typename IntT>
void RaycasterBase<IntT>::castRays(
    const Vector3f& origin,
    const std::vector<Vector3f>& directions,
    std::vector<IntT>& intersections,
    std::vector<uint8_t>& hits)
{
    intersections.resize(directions.size());
    hits.resize(directions.size(), false);

    #pragma omp parallel for
    for(int i=0; i<directions.size(); i++)
    {
        hits[i] = castRay(origin, directions[i], intersections[i]);
    }
}

template<typename IntT>
void RaycasterBase<IntT>::castRays(
    const Vector3f& origin,
    const std::vector<std::vector<Vector3f> >& directions,
    std::vector< std::vector<IntT> >& intersections,
    std::vector< std::vector<uint8_t> >& hits)
{
    intersections.resize(directions.size());
    hits.resize(directions.size());

    #pragma omp parallel for
    for(size_t i=0; i<directions.size(); i++)
    {
        castRays(origin, directions[i], intersections[i], hits[i]);
    }
}

template<typename IntT>
void RaycasterBase<IntT>::castRays(
    const std::vector<Vector3f>& origins,
    const std::vector<Vector3f>& directions,
    std::vector<IntT>& intersections,
    std::vector<uint8_t>& hits)
{
    intersections.resize(directions.size());
    hits.resize(directions.size(), false);

    #pragma omp parallel for
    for(size_t i=0; i<directions.size(); i++)
    {
        hits[i] = castRay(origins[i], directions[i], intersections[i]);
    }
}

template<typename IntT>
void RaycasterBase<IntT>::castRays(
    const std::vector<Vector3f>& origins,
    const std::vector<std::vector<Vector3f> >& directions,
    std::vector<std::vector<IntT> >& intersections,
    std::vector<std::vector<uint8_t> >& hits)
{
    intersections.resize(directions.size());
    hits.resize(directions.size());

    #pragma omp parallel for
    for(size_t i=0; i<directions.size(); i++)
    {
        castRays(origins[i], directions[i], intersections[i], hits[i]);
    }
}

} // namespace lvr2