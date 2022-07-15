
namespace lvr2 
{
namespace baseio
{

template <typename BaseIO>
boost::optional<YAML::Node> MetaIO<BaseIO>::load(
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

    if(!m_BaseIO->m_kernel->exists(*d.metaRoot))
    {
        return ret;
    }

    YAML::Node meta;
    if(!m_BaseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
    {
        return ret;
    }
    ret = meta;
    return ret;
}

template <typename BaseIO>
void MetaIO<BaseIO>::save(
    Description d,
    YAML::Node node) const
{

}

} // namespace baseio
} // namespace lvr2