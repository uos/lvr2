namespace lvr2
{

template<typename T>
size_t OctreeReduction::splitPoints(T* points, const size_t& n, const int&  axis, const double& splitValue)
{
    size_t l = 0, r = n - 1;

    while (l < r)
    {
        while (l < r && points[l](axis) < splitValue)
        {
            ++l;
        }
        while (r > l && points[r](axis) >= splitValue)
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
void OctreeRediction::createOctree(T* points, const int& n, bool* flagged, const T& min, const T& max, const int& level)
{
     if (n <= maxLeafSize)
    {
        return;
    }

    int axis = level % 3;
    Vector3d center = (max + min) / 2.0;

    if (max[axis] - min[axis] <= voxelSize)
    {
        // keep the Point closest to the center
        int closest = 0;
        double minDist = (points[closest].cast<double>() - center).squaredNorm();
        for (int i = 1; i < n; i++)
        {
            double dist = (points[i].cast<double>() - center).squaredNorm();
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

    Vector3d lMin = min, lMax = max;
    Vector3d rMin = min, rMax = max;

    lMax[axis] = center[axis];
    rMin[axis] = center[axis];

    if (l > maxLeafSize)
    {
        #pragma omp task
        createOctree(points    , l    , flagged    , lMin, lMax, level + 1, voxelSize, maxLeafSize);
    }

    if (n - l > maxLeafSize)
    {
        #pragma omp task
        createOctree(points + l, n - l, flagged + l, rMin, rMax, level + 1, voxelSize, maxLeafSize);
    }
}


} // namespace lvr2