/**
 * VertexTraits.hpp
 *
 *  @date 17.06.2011
 *  @author Thomas Wiemann
 */

#ifndef VERTEXTRAITS_HPP_
#define VERTEXTRAITS_HPP_

#include "ColorVertex.hpp"

namespace lssr
{

template<typename VertexT>
struct VertexTraits
{
    static bool HAS_COLOR;
};


template < >
template<typename CoordType>
struct VertexTraits<ColorVertex<CoordType> >
{
    static bool HAS_COLOR;
};

}
#include "VertexTraits.tcc"

#endif /* VERTEXTRAITS_HPP_ */

