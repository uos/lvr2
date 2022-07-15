
#ifndef LVR2_IO_YAML_AABB_HPP
#define LVR2_IO_YAML_AABB_HPP

#include <sstream>

#include <yaml-cpp/yaml.h>
#include "lvr2/io/YAML.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/util/YAMLUtil.hpp"

namespace YAML {

template<>
struct convert<lvr2::BoundingBox< lvr2::BaseVector<float> > > 
{
    static Node encode(const lvr2::BoundingBox< lvr2::BaseVector<float> >& boundingBox) {
        
        Node node;

        Eigen::Matrix<double, 2, 3> aabb;
        aabb(0,0) = boundingBox.getMin().x;
        aabb(0,1) = boundingBox.getMin().y;
        aabb(0,2) = boundingBox.getMin().z;

        aabb(1,0) = boundingBox.getMax().x;
        aabb(1,1) = boundingBox.getMax().y;
        aabb(1,2) = boundingBox.getMax().z;

        node = aabb;

        return node;
    }

    static bool decode(const Node& node, lvr2::BoundingBox< lvr2::BaseVector<float> >& boundingBox) 
    {

        Eigen::Matrix<double, 2, 3> aabb;
        if(!convert<Eigen::Matrix<double, 2, 3> >::decode(node, aabb) )
        {
            return false;
        }

        lvr2::BaseVector<float> aabb_min;
        lvr2::BaseVector<float> aabb_max;

        aabb_min.x = aabb(0,0);
        aabb_min.y = aabb(0,1);
        aabb_min.z = aabb(0,2);
        aabb_max.x = aabb(1,0);
        aabb_max.y = aabb(1,1);
        aabb_max.z = aabb(1,2);

        boundingBox.expand(aabb_min);
        boundingBox.expand(aabb_max);

        return true;
    }

};

} // namespace YAML

#endif // LVR2_IO_YAML_AABB_HPP