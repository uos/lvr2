// Boost.Geometry (aka GGL, Generic Geometry Library)

// Copyright (c) 2007-2011 Barend Gehrels, Amsterdam, the Netherlands.
// Copyright (c) 2008-2011 Bruno Lalande, Paris, France.
// Copyright (c) 2009-2011 Mateusz Loskot, London, UK.

// Parts of Boost.Geometry are redesigned from Geodan's Geographic Library
// (geolib/GGL), copyright (c) 1995-2010 Geodan, Amsterdam, the Netherlands.

// Use, modification and distribution is subject to the Boost Software License,
// Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_GEOMETRY_MULTI_HPP
#define BOOST_GEOMETRY_MULTI_HPP


#include <boost148/geometry/multi/core/closure.hpp>
#include <boost148/geometry/multi/core/geometry_id.hpp>
#include <boost148/geometry/multi/core/is_areal.hpp>
#include <boost148/geometry/multi/core/interior_rings.hpp>
#include <boost148/geometry/multi/core/point_order.hpp>
#include <boost148/geometry/multi/core/point_type.hpp>
#include <boost148/geometry/multi/core/ring_type.hpp>
#include <boost148/geometry/multi/core/tags.hpp>
#include <boost148/geometry/multi/core/topological_dimension.hpp>


#include <boost148/geometry/multi/algorithms/area.hpp>
#include <boost148/geometry/multi/algorithms/centroid.hpp>
#include <boost148/geometry/multi/algorithms/clear.hpp>
#include <boost148/geometry/multi/algorithms/correct.hpp>
#include <boost148/geometry/multi/algorithms/distance.hpp>
#include <boost148/geometry/multi/algorithms/envelope.hpp>
#include <boost148/geometry/multi/algorithms/equals.hpp>
#include <boost148/geometry/multi/algorithms/for_each.hpp>
#include <boost148/geometry/multi/algorithms/intersection.hpp>
#include <boost148/geometry/multi/algorithms/length.hpp>
#include <boost148/geometry/multi/algorithms/num_geometries.hpp>
#include <boost148/geometry/multi/algorithms/num_interior_rings.hpp>
#include <boost148/geometry/multi/algorithms/num_points.hpp>
#include <boost148/geometry/multi/algorithms/perimeter.hpp>
#include <boost148/geometry/multi/algorithms/reverse.hpp>
#include <boost148/geometry/multi/algorithms/simplify.hpp>
#include <boost148/geometry/multi/algorithms/transform.hpp>
#include <boost148/geometry/multi/algorithms/unique.hpp>
#include <boost148/geometry/multi/algorithms/within.hpp>

#include <boost148/geometry/multi/algorithms/detail/modify_with_predicate.hpp>
#include <boost148/geometry/multi/algorithms/detail/multi_sum.hpp>

#include <boost148/geometry/multi/algorithms/detail/sections/range_by_section.hpp>
#include <boost148/geometry/multi/algorithms/detail/sections/sectionalize.hpp>

#include <boost148/geometry/multi/algorithms/detail/overlay/copy_segment_point.hpp>
#include <boost148/geometry/multi/algorithms/detail/overlay/copy_segments.hpp>
#include <boost148/geometry/multi/algorithms/detail/overlay/get_ring.hpp>
#include <boost148/geometry/multi/algorithms/detail/overlay/get_turns.hpp>
#include <boost148/geometry/multi/algorithms/detail/overlay/self_turn_points.hpp>

#include <boost148/geometry/multi/geometries/concepts/check.hpp>
#include <boost148/geometry/multi/geometries/concepts/multi_point_concept.hpp>
#include <boost148/geometry/multi/geometries/concepts/multi_linestring_concept.hpp>
#include <boost148/geometry/multi/geometries/concepts/multi_polygon_concept.hpp>

#include <boost148/geometry/multi/views/detail/range_type.hpp>
#include <boost148/geometry/multi/strategies/cartesian/centroid_average.hpp>

#include <boost148/geometry/multi/util/write_dsv.hpp>



#endif // BOOST_GEOMETRY_MULTI_HPP
