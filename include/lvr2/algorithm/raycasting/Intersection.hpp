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

struct Point {
    Vector3f point;
};

struct Distance {
    float dist;
};

struct Normal {
    Vector3f normal;
};

struct Face {
    unsigned int face_id;
};

struct Barycentrics {
    Vector2f b_uv;
};

struct Mesh {
    unsigned int mesh_id;
};

} // namespace intelem

template<typename ...Tp>
struct Intersection : public Tp... 
{
    static constexpr std::size_t N = sizeof...(Tp);
    using elems = std::tuple<Tp...>;

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
// using 

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