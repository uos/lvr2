#include <algorithm>
#include <iterator>

namespace lvr2 {

template <typename PointT, typename NormalT>
EmbreeRaycaster<PointT, NormalT>::EmbreeRaycaster(const MeshBufferPtr mesh)
:RaycasterBase<PointT, NormalT>(mesh)
{
    m_device = initializeDevice();
    m_scene = initializeScene(m_device, mesh);
    rtcInitIntersectContext(&m_context);
}


template <typename PointT, typename NormalT>
EmbreeRaycaster<PointT, NormalT>::~EmbreeRaycaster()
{
    rtcReleaseScene(m_scene);
    rtcReleaseDevice(m_device);
}

template <typename PointT, typename NormalT>
bool EmbreeRaycaster<PointT, NormalT>::castRay(
    const PointT& origin,
    const NormalT& direction,
    PointT& intersection)
{
    RTCRayHit rayhit = lvr2embree(origin, direction);
    rtcIntersect1(m_scene, &m_context, &rayhit);

    intersection.x = rayhit.ray.org_x + rayhit.ray.tfar * rayhit.ray.dir_x;
    intersection.y = rayhit.ray.org_y + rayhit.ray.tfar * rayhit.ray.dir_y;
    intersection.z = rayhit.ray.org_z + rayhit.ray.tfar * rayhit.ray.dir_z;
    return (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
}

template <typename PointT, typename NormalT>
void EmbreeRaycaster<PointT, NormalT>::castRays(
    const PointT& origin,
    const std::vector<NormalT>& directions,
    std::vector<PointT >& intersections,
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

template <typename PointT, typename NormalT>
void EmbreeRaycaster<PointT, NormalT>::castRays(
    const std::vector<PointT >& origins,
    const std::vector<NormalT >& directions,
    std::vector<PointT >& intersections,
    std::vector<uint8_t>& hits)
{
    
    intersections.resize(directions.size());
    hits.resize(directions.size(), false);

    #pragma omp parallel for
    for(int i=0; i<directions.size(); i++)
    {
        hits[i] = castRay(origins[i], directions[i], intersections[i]);
    }
}

// PROTECTED

template <typename PointT, typename NormalT>
RTCDevice EmbreeRaycaster<PointT, NormalT>::initializeDevice()
{
    RTCDevice device = rtcNewDevice(NULL);

    if (!device)
    {
        std::cerr << "error " << rtcGetDeviceError(NULL) << ": cannot create device" << std::endl;
    }

    rtcSetDeviceErrorFunction(device, errorFunction, NULL);
    return device;
}

template <typename PointT, typename NormalT>
RTCScene EmbreeRaycaster<PointT, NormalT>::initializeScene(
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