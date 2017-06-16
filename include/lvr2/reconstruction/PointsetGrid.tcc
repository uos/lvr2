/*
 * PointsetGrid.tcc
 *
 *  Created on: Nov 27, 2014
 *      Author: twiemann
 */

namespace lvr2
{

template<typename BaseVecT, typename BoxT>
PointsetGrid<BaseVecT, BoxT>::PointsetGrid(
    float cellSize,
    PointsetSurfacePtr<BaseVecT> surface,
    BoundingBox<BaseVecT> bb,
    bool isVoxelsize,
    bool extrude
) :
    HashGrid<BaseVecT, BoxT>(cellSize, bb, isVoxelsize, extrude),
    m_surface(surface)
{
    auto v_min = this->m_boundingBox.getMin();
    auto v_max = this->m_boundingBox.getMax();

    // Get indexed point buffer pointer
    auto numPoint = m_surface->pointBuffer()->getNumPoints();

    cout << lvr::timestamp << "Creating Grid..." << endl;

    // Iterator over all points, calc lattice indices and add lattice points to the grid
    for(size_t i = 0; i < numPoint; i++)
    {
        auto index = (m_surface->pointBuffer()->getPoint(i) - v_min) / this->m_voxelsize;
        this->addLatticePoint(calcIndex(index.x), calcIndex(index.y), calcIndex(index.z));
    }
}


template<typename BaseVecT, typename BoxT>
void PointsetGrid<BaseVecT, BoxT>::calcDistanceValues()
{
    // Status message output
    string comment = lvr::timestamp.getElapsedTime() + "Calculating distance values ";
    lvr::ProgressBar progress(this->m_queryPoints.size(), comment);

    lvr::Timestamp ts;

    // Calculate a distance value for each query point
    #pragma omp parallel for
    for( int i = 0; i < (int)this->m_queryPoints.size(); i++){
        float projectedDistance;
        float euklideanDistance;

        //cout << euklideanDistance << " " << projectedDistance << endl;

        this->m_surface->distance(this->m_queryPoints[i].m_position, projectedDistance, euklideanDistance);
        if (euklideanDistance > 1.7320 * this->m_voxelsize)
        {
            this->m_queryPoints[i].m_invalid = true;
        }
        this->m_queryPoints[i].m_distance = projectedDistance;
        ++progress;
    }
    cout << endl;
    cout << lvr::timestamp << "Elapsed time: " << ts << endl;
}

} // namespace lvr2
