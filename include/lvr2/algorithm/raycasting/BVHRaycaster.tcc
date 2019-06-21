namespace lvr2 {

template <typename PointT, typename NormalT>
BVHRaycaster<PointT, NormalT>::BVHRaycaster(const MeshBufferPtr mesh)
:RaycasterBase<PointT, NormalT>(mesh)
,m_bvh(mesh)
{
    
}

template <typename PointT, typename NormalT>
bool BVHRaycaster<PointT, NormalT>::castRay(
    const PointT& origin,
    const NormalT& direction,
    PointT& intersection
)
{
    // Cast one ray from one origin
    std::vector<uint8_t> tmp(1);

    const float *origin_f = reinterpret_cast<const float*>(&origin.x);
    const float *direction_f = reinterpret_cast<const float*>(&direction.x);
    const unsigned int* clBVHindicesOrTriLists = m_bvh.getIndexesOrTrilists().data();
    const float* clBVHlimits = m_bvh.getLimits().data();
    const float* clTriangleIntersectionData = m_bvh.getTrianglesIntersectionData().data();
    const unsigned int* clTriIdxList = m_bvh.getTriIndexList().data();
    float* result = reinterpret_cast<float*>(&intersection.x);
    uint8_t* result_hits = tmp.data();

    cast_rays_one_one(origin_f, 
        direction_f, 
        clBVHindicesOrTriLists,
        clBVHlimits,
        clTriangleIntersectionData,
        clTriIdxList,
        result,
        result_hits);

    bool success = tmp[0];

    return success;
}

template <typename PointT, typename NormalT>
void BVHRaycaster<PointT, NormalT>::castRays(
    const PointT& origin,
    const std::vector<NormalT >& directions,
    std::vector<PointT >& intersections,
    std::vector<uint8_t>& hits
)
{
    intersections.resize(directions.size());
    hits.resize(directions.size());


    const float *origin_f = reinterpret_cast<const float*>(&origin.x);
    const float *direction_f = reinterpret_cast<const float*>(directions.data());
    const unsigned int* clBVHindicesOrTriLists = m_bvh.getIndexesOrTrilists().data();
    const float* clBVHlimits = m_bvh.getLimits().data();
    const float* clTriangleIntersectionData = m_bvh.getTrianglesIntersectionData().data();
    const unsigned int* clTriIdxList = m_bvh.getTriIndexList().data();
    float* result = reinterpret_cast<float*>(intersections.data());
    uint8_t* result_hits = hits.data();

    size_t num_rays = directions.size();

    cast_rays_one_multi(origin_f, 
        direction_f, 
        num_rays,
        clBVHindicesOrTriLists,
        clBVHlimits,
        clTriangleIntersectionData,
        clTriIdxList,
        result,
        result_hits);

    // Cast multiple rays from one origin
}

template <typename PointT, typename NormalT>
void BVHRaycaster<PointT, NormalT>::castRays(
    const std::vector<PointT >& origins,
    const std::vector<NormalT >& directions,
    std::vector<PointT >& intersections,
    std::vector<uint8_t>& hits
)
{
    intersections.resize(directions.size());
    hits.resize(directions.size());

    const float *origin_f = reinterpret_cast<const float*>(origins.data());
    const float *direction_f = reinterpret_cast<const float*>(directions.data());
    const unsigned int* clBVHindicesOrTriLists = m_bvh.getIndexesOrTrilists().data();
    const float* clBVHlimits = m_bvh.getLimits().data();
    const float* clTriangleIntersectionData = m_bvh.getTrianglesIntersectionData().data();
    const unsigned int* clTriIdxList = m_bvh.getTriIndexList().data();
    float* result = reinterpret_cast<float*>(intersections.data());
    uint8_t* result_hits = hits.data();

    size_t num_rays = directions.size();

    cast_rays_multi_multi(origin_f, 
        direction_f, 
        num_rays,
        clBVHindicesOrTriLists,
        clBVHlimits,
        clTriangleIntersectionData,
        clTriIdxList,
        result,
        result_hits);

}


// PRIVATE FUNCTIONS
template <typename PointT, typename NormalT>
bool BVHRaycaster<PointT, NormalT>::rayIntersectsBox(
    PointT origin,
    Ray ray,
    const float* boxPtr)
{
    const float* limitsX2 = boxPtr;
    const float* limitsY2 = boxPtr+2;
    const float* limitsZ2 = boxPtr+4;

    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin =  (limitsX2[    ray.rayDirSign.x] - origin.x) * ray.invDir.x;
    tmax =  (limitsX2[1 - ray.rayDirSign.x] - origin.x) * ray.invDir.x;
    tymin = (limitsY2[    ray.rayDirSign.y] - origin.y) * ray.invDir.y;
    tymax = (limitsY2[1 - ray.rayDirSign.y] - origin.y) * ray.invDir.y;

    // std::cout << "Statisics: " << std::endl;
    // std::cout << tmin << std::endl;
    // std::cout << tmax << std::endl;
    // std::cout << tymin << std::endl;
    // std::cout << tymax << std::endl;
    // std::cout << std::endl;

    if ((tmin > tymax) || (tymin > tmax))
    {
        return false;
    }
    if (tymin >tmin)
    {
        tmin = tymin;
    }
    if (tymax < tmax)
    {
        tmax = tymax;
    }

    tzmin = (limitsZ2[    ray.rayDirSign.z] - origin.z) * ray.invDir.z;
    tzmax = (limitsZ2[1 - ray.rayDirSign.z] - origin.z) * ray.invDir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
    {
        return false;
    }
    if (tzmin > tmin)
    {
        tmin = tzmin;
    }
    if (tzmax < tmax)
    {
        tmax = tzmax;
    }

    return true;
}

template <typename PointT, typename NormalT>
typename BVHRaycaster<PointT, NormalT>::TriangleIntersectionResult BVHRaycaster<PointT, NormalT>::intersectTrianglesBVH(
    const unsigned int* clBVHindicesOrTriLists,
    PointT origin,
    Ray ray,
    const float* clBVHlimits,
    const float* clTriangleIntersectionData,
    const unsigned int* clTriIdxList
)
{

    int tid_scale = 4;
    int bvh_limits_scale = 2;

    TriangleIntersectionResult result;
    result.hit = false;
    unsigned int pBestTriId = 0;
    float bestTriDist = std::numeric_limits<float>::max();

    unsigned int stack[BVH_STACK_SIZE];

    int stackId = 0;
    stack[stackId++] = 0;
    PointT hitpoint;

    // while stack is not empty
    while (stackId)
    {
        
        unsigned int boxId = stack[stackId - 1];

        stackId--;

        // the first bit of the data of a bvh node indicates whether it is a leaf node, by performing a bitwise and
        // with 0x80000000 all other bits are set to zero and the value of that one bit can be checked
        if (!(clBVHindicesOrTriLists[4 * boxId + 0] & 0x80000000)) // inner node
        {
            
            // if ray intersects inner node, push indices of left and right child nodes on the stack
            if (rayIntersectsBox(origin, ray, &clBVHlimits[bvh_limits_scale * 3 * boxId]))
            {
                stack[stackId++] = clBVHindicesOrTriLists[4 * boxId + 1];
                stack[stackId++] = clBVHindicesOrTriLists[4 * boxId + 2];

                // return if stack size is exceeded
                if ( stackId > BVH_STACK_SIZE)
                {
                    printf("BVH stack size exceeded!");
                    result.hit = 0;
                    return result;
                }
            }
        }
        else // leaf node
        {
            
            // iterate over all triangles in this leaf node
            for (
                unsigned int i = clBVHindicesOrTriLists[4 * boxId + 3];
                i < (clBVHindicesOrTriLists[4 * boxId + 3] + (clBVHindicesOrTriLists[4* boxId + 0] & 0x7fffffff));
                i++
            )
            {
                unsigned int idx = clTriIdxList[i];
                const float* normal = clTriangleIntersectionData + tid_scale * 4 * idx;

                float k = normal[0] * ray.dir[0] + normal[1] * ray.dir[1] + normal[2] * ray.dir[2];
                if (k == 0.0f)
                {
                    continue; // this triangle is parallel to the ray -> ignore it
                }
                float s = (normal[3] - (normal[0] * origin[0] + normal[1] * origin[1] + normal[2] * origin[2])  ) / k;
                if (s <= 0.0f)
                {
                    continue; // this triangle is "behind" the origin
                }
                if (s <= EPSILON)
                {
                    continue; // epsilon
                }
                PointT hit = ray.dir * s;
                hit += origin;

                // ray triangle intersection
                // check if the intersection with the triangle's plane is inside the triangle
                const float* ee1 = clTriangleIntersectionData + tid_scale * 4 * idx + tid_scale*1;
                float kt1 = ee1[0] * hit[0] + ee1[1] * hit[1] + ee1[2] * hit[2] - ee1[3];
                if (kt1 < 0.0f)
                {
                    continue;
                }
                const float* ee2 = clTriangleIntersectionData + tid_scale * 4 * idx + tid_scale * 2;
                float kt2 = ee2[0] * hit[0] + ee2[1] * hit[1] + ee2[2] * hit[2] - ee2[3];
                if (kt2 < 0.0f)
                {
                    continue;
                }
                const float* ee3 = clTriangleIntersectionData + tid_scale * 4 * idx + tid_scale * 3;
                float kt3 = ee3[0] * hit[0] + ee3[1] * hit[1] + ee3[2] * hit[2] - ee3[3];
                if (kt3 < 0.0f)
                {
                    continue;
                }

                // ray intersects triangle, "hit" is the coordinate of the intersection
                {
                    // check if this intersection closer than others
                    // use quadratic distance for comparison to save some root calculations
                    float hitZ = distanceSquare(origin, hit);
                    if (hitZ < bestTriDist)
                    {
                        bestTriDist = hitZ;
                        pBestTriId = idx;
                        result.hit = true;
                        hitpoint = hit;
                    }
                }

            }
        }
    }

    result.pBestTriId = pBestTriId ;
    result.pointHit = hitpoint;
    result.hitDist = sqrt(bestTriDist);

    return result;
}

template <typename PointT, typename NormalT>
void BVHRaycaster<PointT, NormalT>::cast_rays_one_one(
        const float* ray_origin,
        const float* rays,
        const unsigned int* clBVHindicesOrTriLists,
        const float* clBVHlimits,
        const float* clTriangleIntersectionData,
        const unsigned int* clTriIdxList,
        float* result,
        uint8_t* result_hits
    )
{
     // get direction and origin of the ray for the current pose
    NormalT ray_d = {rays[0], rays[1], rays[2]};
    PointT ray_o = {ray_origin[0], ray_origin[1], ray_origin[2]};

    // initialize result memory with zeros
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result_hits[0] = 0;

    // precompute ray values to speed up intersection calculation
    Ray ray;
    ray.dir = ray_d;
    ray.invDir = NormalT(1.0 / ray_d.x, 1.0 / ray_d.y, 1.0 / ray_d.z);
    ray.rayDirSign.x = ray.invDir.x < 0;
    ray.rayDirSign.y = ray.invDir.y < 0;
    ray.rayDirSign.z = ray.invDir.z < 0;

    // intersect all triangles stored in the BVH
    TriangleIntersectionResult resultBVH = intersectTrianglesBVH(
        clBVHindicesOrTriLists,
        ray_o,
        ray,
        clBVHlimits,
        clTriangleIntersectionData,
        clTriIdxList
    );

    // if a triangle was hit, store the calculated hit point in the result at the current id
    if (resultBVH.hit)
    {
        result[0] = resultBVH.pointHit.x;
        result[1] = resultBVH.pointHit.y;
        result[2] = resultBVH.pointHit.z;
        result_hits[0] = 1;
    }

}

template <typename PointT, typename NormalT>
void BVHRaycaster<PointT, NormalT>::cast_rays_one_multi(
        const float* ray_origin,
        const float* rays,
        size_t num_rays,
        const unsigned int* clBVHindicesOrTriLists,
        const float* clBVHlimits,
        const float* clTriangleIntersectionData,
        const unsigned int* clTriIdxList,
        float* result,
        uint8_t* result_hits
    )
{
     // get direction and origin of the ray for the current pose
    PointT ray_o(ray_origin[0], ray_origin[1], ray_origin[2]);

    #pragma omp for
    for(size_t i=0; i< num_rays; i++)
    {
        NormalT ray_d(rays[i*3], rays[i*3+1], rays[i*3+2]);
        // initialize result memory with zeros
        result[i*3] = 0;
        result[i*3+1] = 0;
        result[i*3+2] = 0;
        result_hits[i] = 0;

        // precompute ray values to speed up intersection calculation
        Ray ray;
        ray.dir = ray_d;
        ray.invDir = NormalT(1.0 / ray_d.x, 1.0 / ray_d.y, 1.0 / ray_d.z);
        ray.rayDirSign.x = ray.invDir.x < 0;
        ray.rayDirSign.y = ray.invDir.y < 0;
        ray.rayDirSign.z = ray.invDir.z < 0;

        // intersect all triangles stored in the BVH
        TriangleIntersectionResult resultBVH = intersectTrianglesBVH(
            clBVHindicesOrTriLists,
            ray_o,
            ray,
            clBVHlimits,
            clTriangleIntersectionData,
            clTriIdxList
        );

        // if a triangle was hit, store the calculated hit point in the result at the current id
        if (resultBVH.hit)
        {
            result[i*3] = resultBVH.pointHit.x;
            result[i*3+1] = resultBVH.pointHit.y;
            result[i*3+2] = resultBVH.pointHit.z;
            result_hits[i] = 1;
        }
    }
}


template <typename PointT, typename NormalT>
void BVHRaycaster<PointT, NormalT>::cast_rays_multi_multi(
        const float* ray_origin,
        const float* rays,
        size_t num_rays,
        const unsigned int* clBVHindicesOrTriLists,
        const float* clBVHlimits,
        const float* clTriangleIntersectionData,
        const unsigned int* clTriIdxList,
        float* result,
        uint8_t* result_hits
    )
{
     // get direction and origin of the ray for the current pose
    #pragma omp for
    for(size_t i=0; i< num_rays; i++)
    {
        NormalT ray_d(rays[i*3], rays[i*3+1], rays[i*3+2]);
        PointT ray_o(ray_origin[i*3], ray_origin[i*3+1], ray_origin[i*3+2]);

        // initialize result memory with zeros
        result[i*3] = 0;
        result[i*3+1] = 0;
        result[i*3+2] = 0;
        result_hits[i] = 0;

        // precompute ray values to speed up intersection calculation
        Ray ray;
        ray.dir = ray_d;
        ray.invDir = NormalT(1.0 / ray_d.x, 1.0 / ray_d.y, 1.0 / ray_d.z);
        ray.rayDirSign.x = ray.invDir.x < 0;
        ray.rayDirSign.y = ray.invDir.y < 0;
        ray.rayDirSign.z = ray.invDir.z < 0;

        // intersect all triangles stored in the BVH
        TriangleIntersectionResult resultBVH = intersectTrianglesBVH(
            clBVHindicesOrTriLists,
            ray_o,
            ray,
            clBVHlimits,
            clTriangleIntersectionData,
            clTriIdxList
        );

        // if a triangle was hit, store the calculated hit point in the result at the current id
        if (resultBVH.hit)
        {
            result[i*3] = resultBVH.pointHit.x;
            result[i*3+1] = resultBVH.pointHit.y;
            result[i*3+2] = resultBVH.pointHit.z;
            result_hits[i] = 1;
        }
    }

}

} // namespace lvr2