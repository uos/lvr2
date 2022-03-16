#ifndef METAIO
#define METAIO

#include "lvr2/io/schema/ScanProjectSchema.hpp"

namespace lvr2 {

namespace baseio
{

template <typename BaseIO>
class MetaIO
{
public:
    
    boost::optional<YAML::Node> load(
        Description d) const;
    
    void save(
        Description d, 
        YAML::Node node) const;
protected:
    BaseIO* m_BaseIO = static_cast<BaseIO*>(this);
};

} // namespace baseio

} // namespace lvr2

#include "MetaIO.tcc"

#endif // METAIO
