namespace lvr2 
{

namespace scanio
{

template<typename BaseIO>
void PointCloudIO<BaseIO>::save(
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
        m_baseIO->m_kernel->savePointBuffer(group, name, pcl);
    }
}

template<typename BaseIO>
void PointCloudIO<BaseIO>::save(
    const std::string& groupandname, 
    PointBufferPtr pcl) const
{
    std::cout << "Store each channel individually" << std::endl;
    for(auto elem : *pcl)
    {
        m_vchannel_io->save(groupandname, elem.first, elem.second);
    }
}

template<typename BaseIO>
PointBufferPtr PointCloudIO<BaseIO>::load(
    const std::string& group, 
    const std::string& name) const
{
    // std::cout << "[IO: PointCloudIO - load]: " << group << ", " << name << std::endl;
    boost::filesystem::path p(name);
    if(p.extension() == "") {
        // no extension: assuming to store each channel
        return loadPointCloud(group + "/" + name);
    } else {
        return m_baseIO->m_kernel->loadPointBuffer(group, name);
    }
}

template<typename BaseIO>
PointBufferPtr PointCloudIO<BaseIO>::load(
    const std::string& group) const
{
    // std::cout << "[IO: PointCloudIO - load]: " << group << std::endl;
    PointBufferPtr ret;

    using VChannelT = typename PointBuffer::val_type;

    // load all channel in group
    for(auto meta : m_baseIO->m_kernel->metas(group, "Channel") )
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

template<typename BaseIO>
PointBufferPtr PointCloudIO<BaseIO>::load( 
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

// template<typename BaseIO>
// std::unordered_map<std::string, YAML::Node> PointCloudIO<BaseIO>::loadMeta(
//     const size_t& posNo, 
//     const size_t& lidarNo,
//     const size_t& scanNo) const
// {

// }

// template<typename BaseIO>
// PointBufferPtr PointCloudIO<BaseIO>::load(
//     const size_t& posNo, 
//     const size_t& lidarNo,
//     const size_t& scanNo) const
// {
//     std::cout << "[PointCloudIO - load] " << std::endl;

//     PointBufferPtr ret;

//     auto Dgen = m_baseIO->m_description;
//     Description d = Dgen->scan(posNo, lidarNo, scanNo);


//     std::vector<std::string> known_channel_names;
//     if(d.meta)
//     {
//         YAML::Node scan_meta;
//         m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, scan_meta);
//     }

    
//     if(d.data)
//     {
//         // scan contains data: load them
//     } else {
//         // scan is a group load each channel individually
//         Description d_channel = Dgen->scanChannel(posNo, lidarNo, scanNo, "test");
//         std::cout << "Load channel from " << *d_channel.dataRoot << std::endl;

//         if(d_channel.meta)
//         {
//             // there should lie some metas
//             std::string metaGroup = *d_channel.metaRoot;
//             std::string metaFile = *d_channel.meta;
//             std::tie(metaGroup, metaFile) = hdf5util::validateGroupDataset(metaGroup, metaFile);
//             std::cout << "Channel Located in group " << metaGroup << " at " << metaFile << std::endl;

//             // channelName -> meta
//             std::unordered_map<std::string, YAML::Node> meta_map;

//             for(auto meta : m_baseIO->m_kernel->metas(metaGroup, "Channel"))
//             {
//                 std::cout << "found channel meta at file " << meta.first << std::endl;
//                 std::cout << meta.second << std::endl;

//                 if(meta.second["channel_name"])
//                 {
//                     meta_map[meta.second["channel_name"].template as<std::string>()] = meta.second;
//                 } else {
//                     // assuming the name to be the channel name
//                     meta_map[meta.first] = meta.second;
//                 }
//             }
//         }
//     }
    
    
//     // Description d = Dgen->position(posNo);
//     // d = Dgen->lidar(d, lidarNo);
//     // d = Dgen->scan(d, scanNo);

//     // if(d.dataSetName)
//     // {
//     //     // load pointbuffer from dataset
//     // } else {
//     //     // parse group for channel/pcl like objects and group them together
    
//     //     Description d_channel = Dgen->channel(d, "test");
//     //     for(auto meta : m_baseIO->m_kernel->metas(*d_channel.groupName, "Channel") )
//     //     {
//     //         std::cout << "Meta: " << meta.first << std::endl;
//     //     }
//     // }

//     return ret;
// }

template<typename BaseIO>
void PointCloudIO<BaseIO>::savePointCloud(
    const std::string& group, 
    const std::string& name, 
    PointBufferPtr pcl) const
{
    save(group, name, pcl);
}

template<typename BaseIO>
void PointCloudIO<BaseIO>::savePointCloud(
    const std::string& groupandname,
    PointBufferPtr pcl) const
{
    save(groupandname, pcl);
}

template<typename BaseIO>
PointBufferPtr PointCloudIO<BaseIO>::loadPointCloud(
    const std::string& group, 
    const std::string& name) const
{
    return load(group, name);
}

template<typename BaseIO>
PointBufferPtr PointCloudIO<BaseIO>::loadPointCloud(
    const std::string& group) const
{
    return load(group);
}

template<typename BaseIO>
PointBufferPtr PointCloudIO<BaseIO>::loadPointCloud( 
    const std::string& group,
    const std::string& container, 
    ReductionAlgorithmPtr reduction) const
{
    return load(group, container, reduction);
}

} // namespace scanio

} // namespace lvr2 