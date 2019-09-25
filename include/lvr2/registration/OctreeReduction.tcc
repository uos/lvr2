namespace lvr2
{

template<typename T>
void OctreeReduction::createOctree(lvr2::Channel<T>& points, size_t s, size_t n, bool* flagged, const lvr2::Vector3<T>& min, const lvr2::Vector3<T>& max, const int& level)
{
    if (n <= m_minPointsPerVoxel)
    {
        return;
    }

    int axis = level % 3;
    Vector3<T> center = (max + min) / 2.0;

    if (max[axis] - min[axis] <= m_voxelSize)
    {
        // keep the Point closest to the center
        int closest = s;
        double minDist = (Vector3f(points[closest][0],  points[closest][1], points[closest][2]) - center).squaredNorm();
        for (int i = 1; i < n; i++)
        {
            double dist = (Vector3f(points[s + i][0],  points[s + i][1], points[s + i][2]) - center).squaredNorm();
            if (dist < minDist)
            {
                closest = i;
                minDist = dist;
            }
        }
        // flag all other Points for deletion
        for (int i = s; i < n; i++)
        {
            flagged[i] = i != closest;
        }
        return;
    }

    int l = splitPoints(points, s, n, axis, center[axis]);

    Vector3<T> lMin = min, lMax = max;
    Vector3<T> rMin = min, rMax = max;

    lMax[axis] = center[axis];
    rMin[axis] = center[axis];

    if (l > m_minPointsPerVoxel)
    {
        #pragma omp task
        createOctree<T>(points, s,  l, flagged, lMin, lMax, level + 1);
    }

    if (n - l > m_minPointsPerVoxel)
    {
        #pragma omp task
        createOctree<T>(points, l, n - l, flagged + l, rMin, rMax, level + 1);
    }
}

template<typename T>
size_t OctreeReduction::splitPoints(lvr2::Channel<T>& points, size_t s, size_t n, const int axis, const double& splitValue)
{
    size_t l = s, r = s + n - 1;

    while (l < r)
    {
        while (l < r && points[l][axis] < splitValue)
        {
            ++l;
        }
        while (r > l && points[r][axis] >= splitValue)
        {
            --r;
        }
        if (l < r)
        {
            // std::swap(points[l], points[r]) is not (yet?) compatible with 
            // channels. Using manual swap...
            Vector3f tmp(points[l][0], points[l][1], points[l][2]);
            points[l][0] = points[r][0];
            points[l][1] = points[r][1];
            points[l][2] = points[r][2];

            points[r][0] = tmp[0];
            points[r][1] = tmp[1];
            points[r][2] = tmp[2];
        }
    }

    return l;
}

template<typename T>
size_t OctreeReduction::splitPoints(T* points, const size_t& n, const int axis, const double& splitValue)
{
    size_t l = 0, r = n - 1;

    while (l < r)
    {
        while (l < r && points[l][axis] < splitValue)
        {
            ++l;
        }
        while (r > l && points[r][axis] >= splitValue)
        {
            --r;
        }
        if (l < r)
        {
            std::swap(points[l], points[r]);
        }
    }

    return l;
}

template<typename T>
void OctreeReduction::createOctree(T* points, const int& n, bool* flagged, const T& min, const T& max, const int& level)
{
    if (n <= m_minPointsPerVoxel)
    {
        return;
    }

    int axis = level % 3;
    T center = (max + min) / 2.0;

    if (max[axis] - min[axis] <= m_voxelSize)
    {
        // keep the Point closest to the center
        int closest = 0;
        double minDist = (points[closest] - center).squaredNorm();
        for (int i = 1; i < n; i++)
        {
            double dist = (points[i] - center).squaredNorm();
            if (dist < minDist)
            {
                closest = i;
                minDist = dist;
            }
        }
        // flag all other Points for deletion
        for (int i = 0; i < n; i++)
        {
            flagged[i] = i != closest;
        }
        return;
    }

    int l = splitPoints(points, n, axis, center[axis]);

    T lMin = min, lMax = max;
    T rMin = min, rMax = max;

    lMax[axis] = center[axis];
    rMin[axis] = center[axis];

    if (l > m_minPointsPerVoxel)
    {
        #pragma omp task
        createOctree<T>(points    , l    , flagged    , lMin, lMax, level + 1);
    }

    if (n - l > m_minPointsPerVoxel)
    {
        #pragma omp task
        createOctree<T>(points + l, n - l, flagged + l, rMin, rMax, level + 1);
    }
}


} // namespace lvr2