#ifndef METAIO
#define METAIO

#include "lvr2/io/schema/ScanProjectSchema.hpp"

namespace lvr2 {

namespace scanio
{

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

} // namespace scanio

} // namespace lvr2

#include "MetaIO.tcc"

#endif // METAIO
