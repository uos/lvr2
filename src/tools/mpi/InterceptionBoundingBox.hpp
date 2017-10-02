#ifndef LAS_VEGAS_INTERSEPTBB_HPP
#define LAS_VEGAS_INTERSEPTBB_HPP
#include <lvr/geometry/BoundingBox.hpp>
#include <iostream>
#include <set>
#include <unordered_map>
#include "sortPoint.hpp"
#include <vector>
namespace lvr
{

struct pwithID
{
    std::vector<float>::iterator it;
    size_t globalID;
    
    
    bool operator<(const pwithID& r) const
    {
//         if(this->operator==(r)) return false;
        std::vector<float>::iterator lhs = it;
        std::vector<float>::iterator rhs = r.it;
        if(*(lhs) != *(rhs)) return (*lhs) < (*rhs);
        if(*(lhs+1) != *(rhs+1)) return *(lhs+1) < *(rhs+1);
        return *(lhs+2) < *(rhs+2);
    }
    
    bool operator> (const pwithID& rhs) const
    {
        return ! (this->operator <(rhs));
    }
    
    bool operator==(const pwithID& rhs) const
    {
        
        return std::fabs((float)(*it) - (*rhs.it)) <= 0.0001 &&
           std::fabs((float)(*(it+1)) - (*(rhs.it+1))) <= 0.0001 &&
           std::fabs((float)(*(it+2)) - (*(rhs.it+2))) <= 0.0001;
    }
    
};
    
struct vertexptrComp {
  bool operator() (pwithID l,pwithID r) const
    {
        std::vector<float>::iterator lhs = l.it;
        std::vector<float>::iterator rhs = r.it;
        if(*(lhs) != *(rhs)) return (*lhs) < (*rhs);
        if(*(lhs+1) != *(lhs+1)) return *(lhs+1) < *(rhs+1);
        return *(lhs+2) < *(rhs+2);
      
    }
};
    
class InterseptionBoundingBox
{
public:
    InterseptionBoundingBox(BoundingBox<Vertexf> bb) : m_boundingBox(bb){}
    BoundingBox<Vertexf> m_boundingBox;
    std::set<pwithID> m_points;
    
};
}
#endif //LAS_VEGAS_INTERSEPTBB_HPP