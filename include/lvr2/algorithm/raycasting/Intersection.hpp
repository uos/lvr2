#ifndef LVR2_RAYCASTING_INTERSECTION_HPP
#define LVR2_RAYCASTING_INTERSECTION_HPP

#include <string>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <iostream>
#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2 {

// Intersect flags
// weird intersection construction

namespace intelem {

/**
 * @brief Intersection point (x,y,z)
 * 
 */
struct Point {
    Vector3f point;
};

/**
 * @brief Intersection distance(float)
 * 
 */
struct Distance {
    float dist;
};

/**
 * @brief Raycaster should compute the normal of the intersected face
 *          flipped towards the ray.
 * 
 */
struct Normal {
    Vector3f normal;
};

/**
 * @brief Intersection face as uint id
 * 
 */
struct Face {
    unsigned int face_id;
};

/**
 * @brief Barycentric coordinates of the intersection point.
 * 
 * @code
 * float u = b_uv.x();
 * float v = b_uv.y();
 * float w = 1.0 - u - v;
 * 
 * // v1 - v3: vertices of face
 * // p: intersection point
 * Vector3f p = u * v1 + v * v2 + w * v3
 * @endcode
 *  
 */
struct Barycentrics {
    Vector2f b_uv;
};

/**
 * @brief Receive the intersected Mesh. TODO
 * 
 */
struct Mesh {
    unsigned int mesh_id;
};

} // namespace intelem

/**
 * @brief CRTP Container for User defined intersection elements.
 * 
 * Define your own intersection type that is passed to the Raycaster
 * you like. Allowed elements are in lvr2::intelem
 * 
 * @code
 * // define you intersection type
 * using MyIntType = Intersection<intelem::Point, intelem::Face>;
 * // creating the raycaster
 * RaycasterBasePtr<MyIntType> rc(new BVHRaycaster(meshbuffer));
 * // cast a ray
 * MyIntType intersection_result;
 * bool hit = rc->castRay(ray_origin, ray_direction, intersection_result);
 * // print:
 * if(hit)
 * {
 *      std::cout << intersection_result << std::endl;
 * }
 * @endcode
 * 
 * @tparam Tp List of intelem::* types  
 */
template<typename ...Tp>
struct Intersection : public Tp... 
{
public:
    static constexpr std::size_t N = sizeof...(Tp);
    using elems = std::tuple<Tp...>;

private:
    template <typename T, typename Tuple>
    struct has_type;

    template <typename T>
    struct has_type<T, std::tuple<>> : std::false_type {};

    template <typename T, typename U, typename... Ts>
    struct has_type<T, std::tuple<U, Ts...>> : has_type<T, std::tuple<Ts...>> {};


    template <typename T, typename... Ts>
    struct has_type<T, std::tuple<T, Ts...>> : std::true_type {};    

    template<typename F> 
    struct has_elem {
        static constexpr bool value = has_type<F, elems>::type::value;
    };

public:
    /**
     * @brief Check if Intersection container has a specific intelem (lvr2::intelem).
     * 
     * @tparam F lvr2::intelem type
     * @return true 
     * @return false 
     */
    template<typename F>
    static constexpr bool has() {
        return has_elem<F>::value;
    }
};

// Common types
using PointInt = Intersection<intelem::Point>;
using NormalInt = Intersection<intelem::Normal>;
using DistInt = Intersection<intelem::Distance>;
using FaceInt = Intersection<intelem::Face, intelem::Barycentrics>;
using AllInt = Intersection<
    intelem::Point, 
    intelem::Distance,
    intelem::Normal,
    intelem::Face,
    intelem::Barycentrics,
    intelem::Mesh>;

} // namespace lvr2


template<typename ...T>
std::ostream& operator<<(std::ostream& os, const lvr2::Intersection<T...>& intersection)
{
    using IntT = lvr2::Intersection<T...>;
    os << "Raycaster Intersection: \n";

    if constexpr(IntT::template has<lvr2::intelem::Point>())
    {
        os << "-- point: " << intersection.point.transpose() << "\n";
    }

    if constexpr(IntT::template has<lvr2::intelem::Distance>())
    {
        os << "-- dist: " << intersection.dist << "\n";  
    }

    if constexpr(IntT::template has<lvr2::intelem::Normal>())
    {
        os << "-- normal: " << intersection.normal.transpose() << "\n";
    }

    if constexpr(IntT::template has<lvr2::intelem::Face>())
    {
        os << "-- face: " << intersection.face_id << "\n";
    }

    if constexpr(IntT::template has<lvr2::intelem::Barycentrics>())
    {
        os << "-- barycentrics: " << intersection.b_uv.transpose() << "\n";
    }

    if constexpr(IntT::template has<lvr2::intelem::Mesh>())
    {
        os << "-- mesh: " << intersection.mesh_id << "\n";
    }

    return os;
}

#endif // LVR2_RAYCASTING_INTERSECTION_HPP