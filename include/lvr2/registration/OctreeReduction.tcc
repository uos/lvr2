namespace lvr2
{

template<typename T>
void OctreeReduction::createOctree(
    lvr2::PointBufferPtr& points, 
    size_t startIndex, 
    size_t pointsInVoxel, 
    bool* flagged, 
    const lvr2::Vector3<T>& min, 
    const lvr2::Vector3<T>& max, const int& level)
{
    // Stop recursion - not enough points in voxel
    if (pointsInVoxel <= m_minPointsPerVoxel)
    {
        return;
    }

    // Determine split axis and compute new center
    int axis = level % 3;
    Vector3<T> center = (max + min) / 2.0;

    typename lvr2::Channel<T>::Optional opt = points->getChannel<T>("points");
    if(opt)
    {
        lvr2::Channel<T> point_c = *opt;
        // Stop recursion if voxel size is below given limit
        if (max[axis] - min[axis] <= m_voxelSize)
        {
            // Keep the Point closest to the center
            int closest = startIndex;
            double minDist = (Vector3f(point_c[closest][0], point_c[closest][1], point_c[closest][2]) - center).squaredNorm();
            for (int i = 1; i < pointsInVoxel; i++)
            {
                double dist = (Vector3f(point_c[startIndex + i][0], point_c[startIndex + i][1], point_c[startIndex + i][2]) - center).squaredNorm();
                if (dist < minDist)
                {
                    closest = startIndex + i;
                    minDist = dist;
                }
            }
            // Flag all other Points for deletion
            for (int i = 0; i < pointsInVoxel; i++)
            {
                size_t pos = startIndex + i;
                flagged[pos] = pos != closest;
            }
            return;
        }

        // Sort and get new split index
        int newRight = splitPoints<T>(points, startIndex, pointsInVoxel, axis, center[axis]);

        Vector3<T> lMin = min, lMax = max;
        Vector3<T> rMin = min, rMax = max;

        lMax[axis] = center[axis];
        rMin[axis] = center[axis];

        size_t numPointsLeft = newRight - startIndex;
        size_t numPointsRight = (startIndex + pointsInVoxel) - newRight;

        // If number of points in new intervall is higher then
        // the number of points in the new interval, start recursion
        if (numPointsLeft > m_minPointsPerVoxel)
        {
            #pragma omp task
            createOctree<T>(points, startIndex, numPointsLeft, flagged, lMin, lMax, level + 1);
        }

        if (numPointsRight > m_minPointsPerVoxel)
        {
            #pragma omp task
            createOctree<T>(points, newRight, numPointsRight, flagged, rMin, rMax, level + 1);
        }
    }
    else
    {
        std::cout << timestamp << "Error: OctreeReduction: Unable to get point channel." << std::endl;
    }
}

template<typename T>
void OctreeReduction::swapAllChannelsOfType(lvr2::PointBufferPtr& points, const size_t& l, const size_t& r)
{
    // Go over all supported channel types and sub-sample
    std::vector<std::pair<std::string, Channel<T>>> channels;
    points->getAllChannelsOfType(channels);      
    for(auto i : channels)
    {
        swapInChannel(i.second, l, r);
    }
}

template<typename T>
void OctreeReduction::swapInChannel(lvr2::Channel<T>& ch, const size_t& l, const size_t& r)
{
    // std::swap(points[l], points[r]) is not (yet?) compatible with
    // channels. Using manual swap...
    size_t w = ch.width();

    T* tmp = new T[w];
    for(size_t i = 0; i < w; i++)
    {
        tmp[i] = ch[l][i];

        ch[l][i] = ch[r][i];
        ch[r][i] = tmp[i];
    }
    delete[] tmp;
 }

template<typename T>
size_t OctreeReduction::splitPoints(
    PointBufferPtr& points, 
    size_t startIndex, 
    size_t numPoints, 
    const int axis, 
    const double& splitValue)
{
    size_t l = startIndex;
    size_t r = startIndex + numPoints - 1;

    // Should be safe since we checked the optional in create octree
    lvr2::Channel<T> point_c = points->get<T>("points");
   
    while (l < r)
    {
        while (l < r && point_c[l][axis] < splitValue)
        {
            ++l;
        }
        while (r > l && point_c[r][axis] >= splitValue)
        {
            --r;
        }
        if (l < r)
        {
            // // std::swap(points[l], points[r]) is not (yet?) compatible with 
            // // channels. Using manual swap...
            // Vector3f tmp(points[l][0], points[l][1], points[l][2]);
            // points[l][0] = points[r][0];
            // points[l][1] = points[r][1];
            // points[l][2] = points[r][2];

            // points[r][0] = tmp[0];
            // points[r][1] = tmp[1];
            // points[r][2] = tmp[2];

            // Swap elements in all channels 

            swapAllChannelsOfType<char>(points, l, r);
            swapAllChannelsOfType<unsigned char>(points, l, r);
            swapAllChannelsOfType<short>(points, l, r);
            swapAllChannelsOfType<int>(points, l, r);
            swapAllChannelsOfType<unsigned int>(points, l, r);
            swapAllChannelsOfType<float>(points, l, r);
            swapAllChannelsOfType<double>(points, l, r);

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