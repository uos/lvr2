#include <iostream>
#include <memory>
#include <tuple>
#include <stdlib.h>

#include <boost/optional.hpp>
#include <chrono>

// lvr2 includes
#include "lvr2/util/Synthetic.hpp"
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/geometry/BaseVector.hpp"

#include "lvr2/algorithm/raycasting/RaycasterBase.hpp"
#include "lvr2/algorithm/raycasting/Intersection.hpp"

// LVR2 internal raycaster that is always available
#include "lvr2/algorithm/raycasting/BVHRaycaster.hpp"
#if defined LVR2_USE_OPENCL
#include "lvr2/algorithm/raycasting/CLRaycaster.hpp"
#endif
#if defined LVR2_USE_EMBREE
#include "lvr2/algorithm/raycasting/EmbreeRaycaster.hpp"
#endif

using std::unique_ptr;
using std::make_unique;

using namespace lvr2;

void singleRay()
{
    // contruct a sphere mesh
    MeshBufferPtr mesh = synthetic::genSphere(50, 50);

    // construct a single ray
    Vector3f ray_origin = {0.0,0.0,0.0};
    Vector3f ray_dir = {1.0,0.0,0.0};

    // Choose intersection elements and combine them 
    // by Intersection Container
    // predefined types are also available: PointInt, FaceInt, AllInt
    using MyIntType = Intersection<
        intelem::Point,
        intelem::Distance,
        intelem::Face
    >;

    // construct raycaster
    RaycasterBasePtr<MyIntType> rc;
    rc.reset(new BVHRaycaster<MyIntType>(mesh));


    MyIntType intersection;
    // use the raycaster
    if(rc->castRay(ray_origin, ray_dir, intersection))
    {
        std::cout << "Hit the mesh!" << std::endl;
        std::cout << intersection << std::endl;
    }
}

void multiRay1()
{
    // contruct a sphere mesh
    MeshBufferPtr mesh = synthetic::genSphere(50, 50);

    size_t num_rays = 100000;

    // construct a rays single origin multiple directions
    Vector3f ray_origin = {0.0,0.0,0.0};
    std::vector<Vector3f> ray_dirs(num_rays, {1.0,0.0,0.0});

    // Choose intersection elements and combine them 
    // by Intersection Container
    // predefined types are also available: PointInt, FaceInt, AllInt
    using MyIntType = Intersection<
        intelem::Face,
        intelem::Normal
    >;

    // construct raycaster
    RaycasterBasePtr<MyIntType> rc;
    rc.reset(new BVHRaycaster<MyIntType>(mesh));


    std::vector<MyIntType> intersections;
    std::vector<uint8_t> hits; 

    // use the raycaster
    rc->castRays(ray_origin, ray_dirs, intersections, hits);

    if(hits.back())
    {
        std::cout << "Hit the mesh!" << std::endl;
        std::cout << intersections.back() << std::endl;
    }
}

void multiRay2()
{
    // contruct a sphere mesh
    MeshBufferPtr mesh = synthetic::genSphere(50, 50);

    size_t num_rays = 100000;

    // construct a rays single origin multiple directions
    std::vector<Vector3f> ray_origins(num_rays, {0.0,0.0,0.0});
    std::vector<Vector3f> ray_dirs(num_rays, {1.0,0.0,0.0});

    // Choose intersection elements and combine them 
    // by Intersection Container
    // predefined types are also available: PointInt, FaceInt, AllInt
    // using MyIntType = Intersection<
    //     intelem::Point,
    //     intelem::Distance,
    //     intelem::Face
    // >;

    // construct raycaster
    RaycasterBasePtr<AllInt> rc;
    rc.reset(new BVHRaycaster<AllInt>(mesh));


    std::vector<AllInt> intersections;
    std::vector<uint8_t> hits; 

    // use the raycaster
    rc->castRays(ray_origins, ray_dirs, intersections, hits);

    if(hits.back())
    {
        std::cout << "Hit the mesh!" << std::endl;
        std::cout << intersections.back() << std::endl;
    }
}

int main(int argc, char** argv)
{
    std::cout << "1. Shoot a single ray onto a mesh" << std::endl;
    singleRay();
    std::cout << "2. Shoot rays contraining same origin" << std::endl;
    multiRay1();
    std::cout << "3. Shoot multi rays" << std::endl;
    multiRay2();
    return 0;
}