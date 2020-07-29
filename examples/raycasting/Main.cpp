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

// void genRays(std::vector<Vector3f>& origins,
// std::vector<Vector3f>& directions,
// float scale = 1.0,
// bool flip_axis = false)
// {
//     origins.resize(0);
//     directions.resize(0);

//     // gen origins

//     origins.push_back({0.0,0.0,0.0});
//     origins.push_back({0.0,1.0,0.0});
//     origins.push_back({0.0,2.0,0.0});
//     origins.push_back({0.0,3.0,0.0});
//     origins.push_back({0.0,4.0,0.0});
//     origins.push_back({0.0,5.0,0.0});

//     origins.push_back({0.0,6.0,1.0});
//     origins.push_back({0.0,6.0,2.0});

//     origins.push_back({0.0,7.0,3.0});
//     origins.push_back({0.0,7.0,4.0});
//     origins.push_back({0.0,7.0,5.0});

//     origins.push_back({0.0,8.0,6.0});
//     origins.push_back({0.0,8.0,7.0});
//     origins.push_back({0.0,8.0,8.0});
//     origins.push_back({0.0,8.0,9.0});
//     origins.push_back({0.0,8.0,10.0});
//     origins.push_back({0.0,8.0,11.0});
//     origins.push_back({0.0,8.0,12.0});
//     origins.push_back({0.0,8.0,13.0});

//     origins.push_back({0.0,7.0,14.0});

//     origins.push_back({0.0,6.0,15.0});

//     origins.push_back({0.0,5.0,15.0});
//     origins.push_back({0.0,5.0,14.0});
//     origins.push_back({0.0,5.0,13.0});
//     origins.push_back({0.0,5.0,12.0});

//     origins.push_back({0.0,4.0,16.0});
//     origins.push_back({0.0,3.0,16.0});

//     origins.push_back({0.0,2.0,16.0});
//     origins.push_back({0.0,2.0,15.0});
//     origins.push_back({0.0,2.0,14.0});
//     origins.push_back({0.0,2.0,13.0});
//     origins.push_back({0.0,2.0,17.0});
//     origins.push_back({0.0,2.0,18.0});
//     origins.push_back({0.0,2.0,19.0});
//     origins.push_back({0.0,2.0,20.0});
//     origins.push_back({0.0,2.0,21.0});
//     origins.push_back({0.0,2.0,22.0});

//     origins.push_back({0.0,1.0,23.0});
//     origins.push_back({0.0,0.0,23.0});

//     origins.push_back({0.0,-1.0,22.0});
//     origins.push_back({0.0,-1.0,21.0});
//     origins.push_back({0.0,-1.0,20.0});
//     origins.push_back({0.0,-1.0,19.0});
//     origins.push_back({0.0,-1.0,18.0});
//     origins.push_back({0.0,-1.0,17.0});
//     origins.push_back({0.0,-1.0,16.0});
//     origins.push_back({0.0,-1.0,15.0});
//     origins.push_back({0.0,-1.0,14.0});

//     origins.push_back({0.0,-2.0,16.0});
//     origins.push_back({0.0,-3.0,16.0});

//     origins.push_back({0.0,-4.0,15.0});
//     origins.push_back({0.0,-4.0,14.0});
//     origins.push_back({0.0,-4.0,13.0});
//     origins.push_back({0.0,-4.0,12.0});
//     origins.push_back({0.0,-4.0,11.0});

//     origins.push_back({0.0,-5.0,12.0});

//     origins.push_back({0.0,-6.0,13.0});
//     origins.push_back({0.0,-7.0,13.0});

//     origins.push_back({0.0,-8.0,12.0});

//     origins.push_back({0.0,-9.0,11.0});
//     origins.push_back({0.0,-9.0,10.0});

//     origins.push_back({0.0,-8.0,9.0});

//     origins.push_back({0.0,-7.0,8.0});
//     origins.push_back({0.0,-7.0,7.0});

//     origins.push_back({0.0,-6.0,6.0});
//     origins.push_back({0.0,-6.0,5.0});

//     origins.push_back({0.0,-5.0,4.0});
//     origins.push_back({0.0,-5.0,3.0});

//     origins.push_back({0.0,-4.0,2.0});
//     origins.push_back({0.0,-4.0,1.0});

//     origins.push_back({0.0,-3.0,0.0});
//     origins.push_back({0.0,-2.0,0.0});
//     origins.push_back({0.0,-1.0,0.0});

//     // same rays and scaling and flipping
//     for(int i=0; i<origins.size(); i++)
//     {
//         origins[i] *= scale;

//         if(flip_axis)
//         {
//             float tmp = origins[i].y();
//             origins[i].y() = origins[i].z();
//             origins[i].z() = tmp;
//         }

//         directions.push_back({1.0,0.0,0.0});
//     }

// }

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
        intelem::Point,
        intelem::Distance,
        intelem::Face
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