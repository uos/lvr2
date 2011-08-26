/**
 * ColorVertex.h
 *
 *  @date 17.06.2011
 *  @author Thomas Wiemann
 */

#ifndef COLORVERTEX_H_
#define COLORVERTEX_H_

#include "Vertex.hpp"

typedef unsigned char uchar;

namespace lssr
{

template<typename CoordT, typename ColorT>
class ColorVertex: public Vertex<CoordT>
{
public:
    ColorVertex() : r(0), g(0), b(0) {};
    ColorVertex(CoordT x, CoordT y, CoordT z) : Vertex<CoordT>(x, y, z), r(0), g(0), b(0) {}
    ColorVertex(CoordT x, CoordT y, CoordT z, ColorT red, ColorT blue, ColorT green)
        : Vertex<CoordT>(x, y, z), r(red), b(blue), g(green) {}

    virtual ~ColorVertex() {};

    ColorT      r;
    ColorT      g;
    ColorT      b;
};

typedef ColorVertex<float, uchar> uColorVertex;
typedef ColorVertex<float, float> fColorVertex;

} // namespace llsr

#endif /* COLORVERTEX_H_ */
