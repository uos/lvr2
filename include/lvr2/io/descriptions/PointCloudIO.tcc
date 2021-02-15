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
    // save 
    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->position(posNo);
    d = Dgen->lidar(d, lidarNo);
    d = Dgen->scan(d, scanNo);
    
    if(d.dataSetName)
    {
        // dataSetName given. Store as data as one dataset (e.g. PLY file or group of channels for hdf5)
        std::cout << "Store PCL as dataset: " << *d.dataSetName << std::endl;
    
    
    } else {
        // dataSetName not specified: try to get per channel storage description
        
        std::cout << "Store PCL per channel NEW: " << std::endl;

        

        // example of container_map:
        // points.ply maps to [normals, points, colors]
        // *.bin maps to rest 
        std::unordered_map<std::string, std::vector<std::string> > container_map;
        
        for (auto elem : *pcl)
        {
            Description d_channel = Dgen->channel(d, elem.first);
            
            // boost::filesystem::path pg(*d_channel.groupName);
            // boost::filesystem::path pds(*d_channel.dataSetName);
            

            std::cout << "Channel Description: " << std::endl;
            std::cout << d_channel << std::endl;

            // std::string container = *d_channel.groupName + "/" + *d_channel.dataSetName;

            // if(container_map.find(container) == container_map.end()) {
            //     container_map[container] = {elem.first};
            // } else {
            //     container_map[container].push_back(elem.first);
            // }
        }

        // std::cout << "Container Map: " << std::endl;
        // for(auto elem : container_map)
        // {

        //     boost::filesystem::path container_path(elem.first);

        //     std::cout << "Saving ";
        //     for(auto channel_name : elem.second)
        //     {
        //         std::cout << channel_name << " ";
        //     }
        //     std::cout << " to " << container_path << std::endl;


        //     // we have a path with a potential extension
        //     // like
        //     // 000/000/points.ply
        //     // or
        //     // 000/000/points
        //     // 000/000/normals

        //     // how to get information about the file holding multiple datasets?
        //     // you could also zip then:
        //     // 000/000/points.zip



        //     Description d_channel = Dgen->channel(d, elem.second[0]);



        //     if(elem.second.size() > 1)
        //     {
        //         PointBufferPtr channel_group(new PointBuffer);
        //         for(auto channel_name : elem.second)
        //         {
        //             (*channel_group)[channel_name] = (*pcl)[channel_name];
        //         }
        //         save(*d_channel.groupName, *d_channel.dataSetName, channel_group);
        //         // store meta information per channel
        //     } else {
        //         // store single channel
        //         m_vchannel_io->save(*d_channel.groupName, elem.second[0], (*pcl)[elem.second[0]]);
        //     }
        // }

        // // Store Keys of map to better restore data
        // for(auto elem : *pcl)
        // {
        //     Description d_channel_meta = Dgen->channel(d, elem.first);
        //     if(d_channel_meta.metaName)
        //     {
        //         YAML::Node meta;
        //         meta = elem.second;
        //         meta["channel_name"] = elem.first;
        //         m_featureBase->m_kernel->saveMetaYAML(*d_channel_meta.groupName, *d_channel_meta.metaName, meta);
        //     }
        // }
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

    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->position(posNo);
    d = Dgen->lidar(d, lidarNo);
    d = Dgen->scan(d, scanNo);

    if(d.dataSetName)
    {
        // load pointbuffer from dataset
    } else {
        // parse group for channel/pcl like objects and group them together
    
        Description d_channel = Dgen->channel(d, "test");
        for(auto meta : m_featureBase->m_kernel->metas(*d_channel.groupName, "Channel") )
        {
            std::cout << "Meta: " << meta.first << std::endl;
        }
    }

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