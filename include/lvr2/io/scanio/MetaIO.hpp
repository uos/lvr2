#ifndef LVR2_IO_scanio_META_IO_HPP
#define LVR2_IO_scanio_META_IO_HPP

#include "lvr2/io/scanio/ScanProjectSchema.hpp"

namespace lvr2 {

template <typename FeatureBase>
class MetaIO
{
public:
    
    boost::optional<YAML::Node> load(
        Description d) const;
    
    void save(
        Description d, 
        YAML::Node node) const;
protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
};

} // namespace lvr2

#include "MetaIO.tcc"

#endif // LVR2_IO_scanio_META_IO_HPP