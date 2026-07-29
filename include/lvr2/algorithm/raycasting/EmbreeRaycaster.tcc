
namespace lvr2 {


template<typename IntT>
EmbreeRaycaster<IntT>::EmbreeRaycaster(const MeshBufferPtr mesh)
: m_mesh(mesh)
{
    m_device = initializeDevice();
    m_scene = initializeScene(m_device, mesh);
    #if LVR2_EMBREE_VERSION == 3
    rtcInitIntersectContext(&m_context);
    #endif
}

template<typename IntT>
EmbreeRaycaster<IntT>::~EmbreeRaycaster()
{
    rtcReleaseScene(m_scene);
    rtcReleaseDevice(m_device);
}

template<typename IntT>
bool EmbreeRaycaster<IntT>::castRay(
    const Vector3f& origin,
    const Vector3f& direction,
    IntT& intersection)
{
    RTCRayHit rayhit = lvr2embree(origin, direction);
    
    #if LVR2_EMBREE_VERSION == 3
    rtcIntersect1(m_scene, &m_context, &rayhit);
    #else // == 4
    rtcIntersect1(m_scene, &rayhit);
    #endif

    if constexpr(IntT::template has<intelem::Point>())
    {
        intersection.point.x() = rayhit.ray.org_x + rayhit.ray.tfar * rayhit.ray.dir_x;
        intersection.point.y() = rayhit.ray.org_y + rayhit.ray.tfar * rayhit.ray.dir_y;
        intersection.point.z() = rayhit.ray.org_z + rayhit.ray.tfar * rayhit.ray.dir_z;
    }

    if constexpr(IntT::template has<intelem::Distance>())
    {
        intersection.dist = rayhit.ray.tfar;
    }

    if constexpr(IntT::template has<intelem::Normal>())
    {
        intersection.normal.x() = rayhit.hit.Ng_x;
        intersection.normal.y() = rayhit.hit.Ng_y;
        intersection.normal.z() = rayhit.hit.Ng_z;
        intersection.normal.normalize();
        
        if(direction.dot(intersection.normal) > 0.0)
        {
            intersection.normal = -intersection.normal;
        }
    }

    if constexpr(IntT::template has<intelem::Face>())
    {
        intersection.face_id = rayhit.hit.primID;
    }

    if constexpr(IntT::template has<intelem::Barycentrics>())
    {
        intersection.b_uv.x() = rayhit.hit.u;
        intersection.b_uv.y() = rayhit.hit.v;
    }

    if constexpr(IntT::template has<intelem::Mesh>())
    {
        intersection.mesh_id = rayhit.hit.geomID;
    }

    return (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
}

template<typename IntT>
std::optional<ClosestSurfacePointQueryResult>
EmbreeRaycaster<IntT>::getClosestPoint(const Vector3f& query) const
{
    if (!m_mesh || !m_mesh->hasFaces())
        return std::nullopt;

    RTCPointQuery rtcQuery;
    rtcQuery.x      = query.x();
    rtcQuery.y      = query.y();
    rtcQuery.z      = query.z();
    rtcQuery.time   = 0.0f;
    rtcQuery.radius = std::numeric_limits<float>::infinity();

    RTCPointQueryContext context;
    rtcInitPointQueryContext(&context);

    auto vertices = m_mesh->getVertices();
    auto faces    = m_mesh->getFaceIndices();

    EmbreeClosestPointState state;
    state.vertices = vertices.get();
    state.faces    = faces.get();
    state.found    = false;

    rtcPointQuery(m_scene, &rtcQuery, &context, embreeClosestPointCallback, &state);

    if (state.found)
        return state.result;
    return std::nullopt;
}

// PRIVATE

template<typename IntT>
RTCDevice EmbreeRaycaster<IntT>::initializeDevice()
{
    RTCDevice device = rtcNewDevice(NULL);

    if (!device)
    {
        std::cerr << "error " << rtcGetDeviceError(NULL) << ": cannot create device" << std::endl;
    }

    rtcSetDeviceErrorFunction(device, EmbreeErrorFunction, NULL);
    return device;
}

template<typename IntT>
RTCScene EmbreeRaycaster<IntT>::initializeScene(
    RTCDevice device,
    const MeshBufferPtr mesh)
{
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    auto lvr_vertices = mesh->getVertices();
    int num_vertices = mesh->numVertices();

    auto lvr_indices = mesh->getFaceIndices();
    int num_faces = mesh->numFaces();

    float* embree_vertices = (float*) rtcSetNewGeometryBuffer(geom,
                                                        RTC_BUFFER_TYPE_VERTEX,
                                                        0,
                                                        RTC_FORMAT_FLOAT3,
                                                        3*sizeof(float),
                                                        num_vertices);

    unsigned* embree_indices = (unsigned*) rtcSetNewGeometryBuffer(geom,
                                                            RTC_BUFFER_TYPE_INDEX,
                                                            0,
                                                            RTC_FORMAT_UINT3,
                                                            3*sizeof(unsigned),
                                                            num_faces);

    if (embree_vertices && embree_indices)
    {
        // copy mesh to embree buffers
        const float* lvr_vertices_begin = lvr_vertices.get();
        const float* lvr_vertices_end = lvr_vertices_begin + num_vertices * 3;
        std::copy(lvr_vertices_begin, lvr_vertices_end, embree_vertices);

        const unsigned int* lvr_indices_begin = lvr_indices.get();
        const unsigned int* lvr_indices_end = lvr_indices_begin + num_faces * 3;
        std::copy(lvr_indices_begin, lvr_indices_end, embree_indices);
    }

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    return scene;
}

} // namespace lvr2
