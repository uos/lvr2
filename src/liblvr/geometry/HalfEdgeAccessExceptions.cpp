#include "geometry/HalfEdgeAccessExceptions.hpp"

namespace lvr
{

ostream& operator<<(ostream& os, const HalfEdgeAccessException e)
{
    os << e.what() << endl;
    return os;
}

} // namespace lvr

