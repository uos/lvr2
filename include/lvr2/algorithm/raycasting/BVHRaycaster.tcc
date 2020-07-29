
namespace lvr2 {

template<typename IntT>
BVHRaycaster<IntT>::BVHRaycaster(const MeshBufferPtr mesh, unsigned int stack_size)
:RaycasterBase<IntT>(mesh)
,m_bvh(mesh)
,m_faces(mesh->getFaceIndices())
,m_vertices(mesh->getVertices())
,m_BVHindicesOrTriLists(m_bvh.getIndexesOrTrilists().data())
,m_BVHlimits(m_bvh.getLimits().data())
,m_TriangleIntersectionData(m_bvh.getTrianglesIntersectionData().data())
,m_TriIdxList(m_bvh.getTriIndexList().data())
,m_stack_size(stack_size)
{
    
}

template<typename IntT>
bool BVHRaycaster<IntT>::castRay(
    const Vector3f& origin,
    const Vector3f& direction,
    IntT& intersection)
{
    Ray ray;
    ray.dir = direction;
    // wtf /0 ???? 
    ray.invDir = {1.0f / ray.dir.x(), 1.0f / ray.dir.y(), 1.0f / ray.dir.z() };

    ray.rayDirSign.x() = ray.invDir.x() < 0;
    ray.rayDirSign.y() = ray.invDir.y() < 0;
    ray.rayDirSign.z() = ray.invDir.z() < 0;

    TriangleIntersectionResult result 
        = intersectTrianglesBVH(
            m_BVHindicesOrTriLists,
            origin, 
            ray, 
            m_BVHlimits, 
            m_TriangleIntersectionData, 
            m_TriIdxList);
    
    // FINISHING
    // translate to IntT
    if constexpr(IntT::template has<intelem::Point>())
    {
        intersection.point = result.pointHit;
    }

    if constexpr(IntT::template has<intelem::Distance>())
    {
        intersection.dist = result.hitDist;
    }

    if constexpr(IntT::template has<intelem::Normal>())
    {
        unsigned int v1id = m_faces[result.pBestTriId * 3 + 0];
        unsigned int v2id = m_faces[result.pBestTriId * 3 + 1];
        unsigned int v3id = m_faces[result.pBestTriId * 3 + 2];

        Vector3f v1(m_vertices[v1id * 3 + 0], m_vertices[v1id * 3 + 1], m_vertices[v1id * 3 + 2]);
        Vector3f v2(m_vertices[v2id * 3 + 0], m_vertices[v2id * 3 + 1], m_vertices[v2id * 3 + 2]);
        Vector3f v3(m_vertices[v3id * 3 + 0], m_vertices[v3id * 3 + 1], m_vertices[v3id * 3 + 2]);

        intersection.normal = (v3 - v1).cross((v2 - v1));
        intersection.normal.normalize();
        if(direction.dot(intersection.normal) > 0.0)
        {
            intersection.normal = -intersection.normal;
        }
    }

    if constexpr(IntT::template has<intelem::Face>())
    {
        intersection.face_id = result.pBestTriId;
    }

    if constexpr(IntT::template has<intelem::Barycentrics>())
    {
        unsigned int v1id = m_faces[result.pBestTriId * 3 + 0];
        unsigned int v2id = m_faces[result.pBestTriId * 3 + 1];
        unsigned int v3id = m_faces[result.pBestTriId * 3 + 2];
        
        Vector3f v1(m_vertices[v1id * 3 + 0], m_vertices[v1id * 3 + 1], m_vertices[v1id * 3 + 2]);
        Vector3f v2(m_vertices[v2id * 3 + 0], m_vertices[v2id * 3 + 1], m_vertices[v2id * 3 + 2]);
        Vector3f v3(m_vertices[v3id * 3 + 0], m_vertices[v3id * 3 + 1], m_vertices[v3id * 3 + 2]);
    
        Vector3f bary = barycentric(result.pointHit, v1, v2, v3);
        intersection.b_uv.x() = bary.x();
        intersection.b_uv.y() = bary.y();
    }

    if constexpr(IntT::template has<intelem::Mesh>())
    {
        // TODO
        intersection.mesh_id = 0;
    }

    return result.hit;
}

// PRIVATE
template<typename IntT>
typename BVHRaycaster<IntT>::TriangleIntersectionResult 
BVHRaycaster<IntT>::intersectTrianglesBVH(
    const unsigned int* clBVHindicesOrTriLists,
    Vector3f origin,
    Ray ray,
    const float* clBVHlimits,
    const float* clTriangleIntersectionData,
    const unsigned int* clTriIdxList)
{
    int tid_scale = 4;
    int bvh_limits_scale = 2;

    TriangleIntersectionResult result;
    result.hit = false;
    unsigned int pBestTriId = 0;
    float bestTriDist = std::numeric_limits<float>::max();

    unsigned int stack[m_stack_size];

    int stackId = 0;
    stack[stackId++] = 0;
    Vector3f hitpoint;

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
                if ( stackId > m_stack_size)
                {
                    printf("BVH stack size exceeded!\n");
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
                Vector3f hit = ray.dir * s;
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

template<typename IntT>
bool BVHRaycaster<IntT>::rayIntersectsBox(
    Vector3f origin,
    Ray ray,
    const float* boxPtr)
{
    const float* limitsX2 = boxPtr;
    const float* limitsY2 = boxPtr+2;
    const float* limitsZ2 = boxPtr+4;

    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin =  (limitsX2[    ray.rayDirSign.x()] - origin.x()) * ray.invDir.x();
    tmax =  (limitsX2[1 - ray.rayDirSign.x()] - origin.x()) * ray.invDir.x();
    tymin = (limitsY2[    ray.rayDirSign.y()] - origin.y()) * ray.invDir.y();
    tymax = (limitsY2[1 - ray.rayDirSign.y()] - origin.y()) * ray.invDir.y();

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

    tzmin = (limitsZ2[    ray.rayDirSign.z()] - origin.z()) * ray.invDir.z();
    tzmax = (limitsZ2[1 - ray.rayDirSign.z()] - origin.z()) * ray.invDir.z();

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

} // namespace lvr2