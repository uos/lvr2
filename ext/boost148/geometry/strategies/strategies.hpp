// Boost.Geometry (aka GGL, Generic Geometry Library)

// Copyright (c) 2007-2011 Barend Gehrels, Amsterdam, the Netherlands.
// Copyright (c) 2008-2011 Bruno Lalande, Paris, France.
// Copyright (c) 2009-2011 Mateusz Loskot, London, UK.

// Parts of Boost.Geometry are redesigned from Geodan's Geographic Library
// (geolib/GGL), copyright (c) 1995-2010 Geodan, Amsterdam, the Netherlands.

// Use, modification and distribution is subject to the Boost Software License,
// Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_GEOMETRY_STRATEGIES_STRATEGIES_HPP
#define BOOST_GEOMETRY_STRATEGIES_STRATEGIES_HPP


#include <boost148/geometry/strategies/tags.hpp>

#include <boost148/geometry/strategies/area.hpp>
#include <boost148/geometry/strategies/centroid.hpp>
#include <boost148/geometry/strategies/compare.hpp>
#include <boost148/geometry/strategies/convex_hull.hpp>
#include <boost148/geometry/strategies/distance.hpp>
#include <boost148/geometry/strategies/intersection.hpp>
#include <boost148/geometry/strategies/side.hpp>
#include <boost148/geometry/strategies/transform.hpp>
#include <boost148/geometry/strategies/within.hpp>

#include <boost148/geometry/strategies/cartesian/area_surveyor.hpp>
#include <boost148/geometry/strategies/cartesian/box_in_box.hpp>
#include <boost148/geometry/strategies/cartesian/centroid_bashein_detmer.hpp>
#include <boost148/geometry/strategies/cartesian/centroid_weighted_length.hpp>
#include <boost148/geometry/strategies/cartesian/distance_pythagoras.hpp>
#include <boost148/geometry/strategies/cartesian/distance_projected_point.hpp>
#include <boost148/geometry/strategies/cartesian/point_in_box.hpp>
#include <boost148/geometry/strategies/cartesian/point_in_poly_franklin.hpp>
#include <boost148/geometry/strategies/cartesian/point_in_poly_crossings_multiply.hpp>
#include <boost148/geometry/strategies/cartesian/side_by_triangle.hpp>

#include <boost148/geometry/strategies/spherical/area_huiller.hpp>
#include <boost148/geometry/strategies/spherical/distance_haversine.hpp>
#include <boost148/geometry/strategies/spherical/distance_cross_track.hpp>
#include <boost148/geometry/strategies/spherical/compare_circular.hpp>
#include <boost148/geometry/strategies/spherical/ssf.hpp>

#include <boost148/geometry/strategies/agnostic/hull_graham_andrew.hpp>
#include <boost148/geometry/strategies/agnostic/point_in_box_by_side.hpp>
#include <boost148/geometry/strategies/agnostic/point_in_poly_winding.hpp>
#include <boost148/geometry/strategies/agnostic/simplify_douglas_peucker.hpp>

#include <boost148/geometry/strategies/strategy_transform.hpp>

#include <boost148/geometry/strategies/transform/matrix_transformers.hpp>
#include <boost148/geometry/strategies/transform/map_transformer.hpp>
#include <boost148/geometry/strategies/transform/inverse_transformer.hpp>


#endif // BOOST_GEOMETRY_STRATEGIES_STRATEGIES_HPP
