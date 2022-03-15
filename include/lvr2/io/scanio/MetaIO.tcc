
namespace lvr2 {

namespace scanio
{

template <typename FeatureBase>
boost::optional<YAML::Node> MetaIO<FeatureBase>::load(
    Description d) const
{
    boost::optional<YAML::Node> ret;

    if(!d.meta)
    {
        return ret;
    }

    if(!d.metaRoot)
    {
        d.metaRoot = d.dataRoot;
    }

    if(!m_featureBase->m_kernel->exists(*d.metaRoot))
    {
        return ret;
    }

    YAML::Node meta;
    if(!m_featureBase->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
    {
        return ret;
    }
    ret = meta;
    return ret;
}

template <typename FeatureBase>
void MetaIO<FeatureBase>::save(
    Description d,
    YAML::Node node) const
{

}

} // namespace scanio

} // namespace lvr2