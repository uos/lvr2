namespace lvr2 
{

template<typename FeatureBase>
void PointCloudIO<FeatureBase>::save(
    const std::string& group, 
    const std::string& name,
    PointBufferPtr pcl) const
{
    boost::filesystem::path p(name);
    if(p.extension() == "")
    {
        std::string groupandname = group + "/" + name;
        save(groupandname, pcl);
    } else {
        m_featureBase->m_kernel->savePointBuffer(group, name, pcl);
    }
}

template<typename FeatureBase>
void PointCloudIO<FeatureBase>::save(
    const std::string& groupandname, 
    PointBufferPtr pcl) const
{
    std::cout << "Store each channel individually" << std::endl;
    for(auto elem : *pcl)
    {
        m_vchannel_io->save(groupandname, elem.first, elem.second);
    }
}

template<typename FeatureBase>
void PointCloudIO<FeatureBase>::save(
    const size_t& posNo,
    const size_t& lidarNo,
    const size_t& scanNo,
    PointBufferPtr pcl) const
{
    std::cout <<  "SSAAAAVFE " << std::endl;

    auto Dgen = m_featureBase->m_description;

    std::unordered_map<std::string, PointBufferPtr> container_map;

    // DATA
    for(auto elem : *pcl)
    {
        Description d = Dgen->scanChannel(posNo, lidarNo, scanNo, elem.first);
        boost::filesystem::path proot(*d.dataRoot);
        if(proot.extension() != "")
        {
            if(container_map.find(proot.string()) == container_map.end())
            {
                container_map[proot.string()] = PointBufferPtr(new PointBuffer);
            }
            (*container_map[proot.string()])[elem.first] = elem.second;
        } else {
            m_vchannel_io->save(*d.dataRoot, *d.data, elem.second);
        }        
    }

    // found file format in dataRoot, example: ply, obj, zip. Need to store grouped
    for(auto elem : container_map)
    {
        m_featureBase->savePointCloud("", elem.first, elem.second);
    }

    // META
    for(auto elem : *pcl)
    {
        Description d = Dgen->scanChannel(posNo, lidarNo, scanNo, elem.first);
        
        if(d.meta)
        {
            YAML::Node meta;
            meta = elem.second;
            m_featureBase->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, meta);
        }
    }

}

template<typename FeatureBase>
PointBufferPtr PointCloudIO<FeatureBase>::load(
    const std::string& group, 
    const std::string& name) const
{
    // std::cout << "[IO: PointCloudIO - load]: " << group << ", " << name << std::endl;
    boost::filesystem::path p(name);
    if(p.extension() == "") {
        // no extension: assuming to store each channel
        return loadPointCloud(group + "/" + name);
    } else {
        return m_featureBase->m_kernel->loadPointBuffer(group, name);
    }
}

template<typename FeatureBase>
PointBufferPtr PointCloudIO<FeatureBase>::load(
    const std::string& group) const
{
    // std::cout << "[IO: PointCloudIO - load]: " << group << std::endl;
    PointBufferPtr ret;

    using VChannelT = typename PointBuffer::val_type;

    // load all channel in group
    for(auto meta : m_featureBase->m_kernel->metas(group, "Channel") )
    {
        boost::optional<VChannelT> copt = m_vchannel_io->template loadVariantChannel<VChannelT>(group, meta.first);
        if(copt)
        {
            if(!ret)
            {
                ret.reset(new PointBuffer);
            }
            // add channel
            (*ret)[meta.first] = *copt;
        }
    }

    return ret;
}

template<typename FeatureBase>
PointBufferPtr PointCloudIO<FeatureBase>::load( 
    const std::string& group,
    const std::string& container, 
    ReductionAlgorithmPtr reduction) const
{
    if(reduction)
    {
        PointBufferPtr buffer = loadPointCloud(group, container);
        reduction->setPointBuffer(buffer);
        return reduction->getReducedPoints();
    } else {
        return loadPointCloud(group, container);
    }
}

template<typename FeatureBase>
PointBufferPtr PointCloudIO<FeatureBase>::load(
    const size_t& posNo, 
    const size_t& lidarNo,
    const size_t& scanNo) const
{
    PointBufferPtr ret;

    // auto Dgen = m_featureBase->m_description;
    // Description d = Dgen->position(posNo);
    // d = Dgen->lidar(d, lidarNo);
    // d = Dgen->scan(d, scanNo);

    // if(d.dataSetName)
    // {
    //     // load pointbuffer from dataset
    // } else {
    //     // parse group for channel/pcl like objects and group them together
    
    //     Description d_channel = Dgen->channel(d, "test");
    //     for(auto meta : m_featureBase->m_kernel->metas(*d_channel.groupName, "Channel") )
    //     {
    //         std::cout << "Meta: " << meta.first << std::endl;
    //     }
    // }

    return ret;
}

template<typename FeatureBase>
void PointCloudIO<FeatureBase>::savePointCloud(
    const std::string& group, 
    const std::string& name, 
    PointBufferPtr pcl) const
{
    save(group, name, pcl);
}

template<typename FeatureBase>
void PointCloudIO<FeatureBase>::savePointCloud(
    const std::string& groupandname,
    PointBufferPtr pcl) const
{
    save(groupandname, pcl);
}

template<typename FeatureBase>
PointBufferPtr PointCloudIO<FeatureBase>::loadPointCloud(
    const std::string& group, 
    const std::string& name) const
{
    return load(group, name);
}

template<typename FeatureBase>
PointBufferPtr PointCloudIO<FeatureBase>::loadPointCloud(
    const std::string& group) const
{
    return load(group);
}

template<typename FeatureBase>
PointBufferPtr PointCloudIO<FeatureBase>::loadPointCloud( 
    const std::string& group,
    const std::string& container, 
    ReductionAlgorithmPtr reduction) const
{
    return load(group, container, reduction);
}

} // namespace lvr2 