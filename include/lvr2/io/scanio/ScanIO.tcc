

namespace lvr2
{

template <typename FeatureBase>
void ScanIO<FeatureBase>::save(
    const size_t& scanPosNo,
    const size_t& sensorNo,
    const size_t& scanNo,
    ScanPtr scanPtr) const
{
    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->scan(scanPosNo, sensorNo, scanNo);

    // std::cout << "[ScanIO - save]" << std::endl;
    // std::cout << d << std::endl;

    if(!d.dataRoot)
    {
        return;
    }

    const bool data_loaded_before = scanPtr->loaded();

    if(!data_loaded_before)
    {
        scanPtr->load();
    }

    // std::cout << "ScanIO - save data " << scanPosNo << " " << sensorNo << " " << scanNo << std::endl;
    //// DATA
    if(scanPtr->points)
    {
        if(d.data)
        { 
            // std::cout << "Save Channel wise" << std::endl;
            m_pclIO->save(*d.dataRoot, *d.data, scanPtr->points);

            // save metas
            for(auto elem : *scanPtr->points)
            {
                Description dc = Dgen->scanChannel(scanPosNo, sensorNo, scanNo, elem.first);
                // Meta
                if(dc.meta)
                {
                    YAML::Node meta;
                    meta = elem.second;
                    meta["name"] = elem.first;
                    m_featureBase->m_kernel->saveMetaYAML(*dc.metaRoot, *dc.meta, meta);
                }
            }

        } else {
            // std::cout << "Save Partial" << std::endl;
            // a lot of code for the problem of capsulating SOME channels into one PLY
            // there could be other channels that do not fit in this ply

            // Scan is not a dataset: handle as group of channels
            // m_pclIO->save(scanPosNo, sensorNo, scanNo, scanPtr->points);
            std::unordered_map<std::string, PointBufferPtr> one_file_multi_channel;

            /// Data (Channel)
            for(auto elem : *scanPtr->points)
            {
                // std::cout << "Save " << elem.first << std::endl;
                Description dc = Dgen->scanChannel(scanPosNo, sensorNo, scanNo, elem.first);
                boost::filesystem::path proot(*dc.dataRoot);

                
                // std::cout << "Description of " <<  elem.first << std::endl;
                // std::cout << dc << std::endl;

                if(proot.extension() != "")
                {
                    if(one_file_multi_channel.find(proot.string()) == one_file_multi_channel.end())
                    {
                        one_file_multi_channel[proot.string()] = PointBufferPtr(new PointBuffer);
                    }
                    (*one_file_multi_channel[proot.string()])[elem.first] = elem.second;
                } else {
                    // group is no file
                    // concatenate group and dataset and store directly
                    // std::string filename = (proot / *dc.data).string();
                    // std::cout << "Store single channel " << elem.first << " to " << filename << std::endl;
                    
                    std::string group, dataset;
                    std::tie(group, dataset) = hdf5util::validateGroupDataset(proot.string(), *dc.data);

                    // std::cout << "Save " << elem.first << " to " << group << " - " << dataset << std::endl;

                    // Data
                    m_vchannel_io->save(group, dataset, elem.second);
                }
            }

            // DATA of exceptional "one_file_multi_channel". there we need to call the 
            // kernel directly, because we are now at file level
            for(auto ex_elem : one_file_multi_channel)
            {
                std::string group, name;
                std::tie(group, name) = hdf5util::validateGroupDataset("", ex_elem.first);

                // std::cout << "Save Channel Group to file: " << ex_elem.first << std::endl;
                // std::cout << *ex_elem.second << std::endl;

                m_featureBase->m_kernel->savePointBuffer(group, name, ex_elem.second);
            }

            // std::cout << "Save Channel METAs" << std::endl;

            // META
            // save meta for each channel
            for(auto elem : *scanPtr->points)
            {
                Description dc = Dgen->scanChannel(scanPosNo, sensorNo, scanNo, elem.first);
                if(dc.meta)
                {
                    YAML::Node meta;
                    meta = elem.second;
                    meta["name"] = elem.first;
                    m_featureBase->m_kernel->saveMetaYAML(*dc.metaRoot, *dc.meta, meta);
                }
            }

            // std::cout << "END channels" << std::endl;
        }
    }

    if(!data_loaded_before)
    {
        scanPtr->release();
    }

    // std::cout << "save SCAN META" << std::endl;

    //// META
    if(d.meta)
    {
        YAML::Node node;
        node = *scanPtr;

        // std::cout << "write YAML" << std::endl;
        m_featureBase->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
    
    // std::cout << "Success" << std::endl;
}

template <typename FeatureBase>
boost::optional<YAML::Node> ScanIO<FeatureBase>::loadMeta(
    const size_t& scanPosNo, 
    const size_t& sensorNo,
    const size_t& scanNo) const
{
    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->scan(scanPosNo, sensorNo, scanNo);
    return m_metaIO->load(d);
}

template <typename FeatureBase>
ScanPtr ScanIO<FeatureBase>::load(
    const size_t& scanPosNo, 
    const size_t& sensorNo,
    const size_t& scanNo) const
{
    ScanPtr ret;

    // Get Description of Scan Location

    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->scan(scanPosNo, sensorNo, scanNo);

    if(!d.dataRoot)
    {
        return ret;
    }

    if(!m_featureBase->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }

    // std::cout << "[ScanIO - load] Description:" << std::endl;
    // std::cout << d << std::endl;

    /// META
    if(d.meta)
    {
        YAML::Node meta;
        if(!m_featureBase->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }
        ret = std::make_shared<Scan>(meta.as<Scan>());
    } else {
        // for schemas without meta information
        ret.reset(new Scan);
    }


    /// Load each channel
    /// We need to load each channel here, because the channel list is in the scan meta file
    /// if you want to make it another way you need to change this first

    std::function<PointBufferPtr()> points_loader;

    if(d.data)
    {
        points_loader = [this, d]() {
            return this->m_pclIO->load(*d.dataRoot, *d.data);
        };
    } else {

        points_loader = [this, d, Dgen, scanPosNo, sensorNo, scanNo]() 
        {
            PointBufferPtr points;

            // Try to find some meta information first
            // key: channel_name, value: meta information
            auto channel_metas = loadChannelMetas(scanPosNo, sensorNo, scanNo);

            if(!channel_metas.empty())
            {
                // std::cout << "Found channel metas " << std::endl;
                for(auto elem : channel_metas)
                {
                    // check if element was added already
                    if(!points || points->find(elem.first) == points->end() )
                    {
                        Description dc = Dgen->scanChannel(scanPosNo, sensorNo, scanNo, elem.first);

                        // data is at dc.dataRoot / dc.data

                        boost::filesystem::path proot(*dc.dataRoot);

                        if(proot.extension() != "")
                        {
                            // channels in file
                            std::string group, name;
                            std::tie(group, name) = hdf5util::validateGroupDataset("", proot.string());
                            PointBufferPtr points_ = m_featureBase->m_kernel->loadPointBuffer(group, name);

                            // merge to complete map
                            if(!points)
                            {
                                points = points_;
                            } else {
                                for(auto elem : *points_)
                                {
                                    (*points)[elem.first] = elem.second;
                                }
                            }
                        } else {
                            // channels in folder
                            auto vo = m_vchannel_io->template load<typename PointBuffer::val_type>(*dc.dataRoot, *dc.data);
                            if(vo)
                            {
                                if(!points)
                                {
                                    points.reset(new PointBuffer);
                                }
                                (*points)[elem.first] = *vo;
                            }
                        }

                    }
                }

            } else {

                // std::cout << "Could not get channel metas" << std::endl;

                // no meta information about channels
                // could be in case of datasets cannot be 

                // but we know that points must be there
                Description dc = Dgen->scanChannel(scanPosNo, sensorNo, scanNo, "points");

                // search for data root
                boost::filesystem::path proot(*dc.dataRoot);

                if(proot.extension() != "")
                {
                    std::string group, dataset;
                    std::tie(group, dataset) = hdf5util::validateGroupDataset("", proot.string());
                    points = m_featureBase->m_kernel->loadPointBuffer(group, dataset);
                } else {
                    // search
                    if(dc.data)
                    {
                        boost::filesystem::path pdata = *dc.data;
                        if(pdata.extension() != "")
                        {
                            // found potential file to filter for
                            for(auto name : m_featureBase->m_kernel->listDatasets(proot.string()) )
                            {
                                PointBufferPtr points_ = m_featureBase->m_kernel->loadPointBuffer(proot.string(), name);

                                if(!points)
                                {
                                    points = points_;
                                } else if(points_) {
                                    for(auto elem : *points_)
                                    {
                                        (*points)[elem.first] = elem.second;
                                    }
                                }
                            }

                        } else {
                            // situation:
                            // no extension of group and no extension of dataset
                            // no meta data

                            // there are two options what happend here
                            // 1. Used Hdf5 schema and did not find any meta data
                            //    - this should not happen. meta data must be available
                            // 2. Used directory schema and stored binary channels
                            //    - this should not happen. binary channels must have an meta file

                            // std::cout << dc << std::endl;

                            throw std::runtime_error("[ScanIO - Panic. Something orrured that should not happen]");
                        }
                    }
                }
            }

            return points;
        }; // points_loader lambda
    }   

    // add reduced version
    std::function<PointBufferPtr(ReductionAlgorithmPtr)> points_loader_reduced = [points_loader](ReductionAlgorithmPtr red) {
        PointBufferPtr points = points_loader();

        if(points)
        {
            red->setPointBuffer(points);
            points = red->getReducedPoints();
        }

        return points;
    };

    // load data here?
    // TODO: add points_loader and points_loader_reduced to struct instead
    // Old:


    // ret->points = points_loader();
    // New:
    
    if(m_featureBase->m_load_data)
    {
        ret->points = points_loader();
    }
    
    ret->points_loader = points_loader;
    ret->points_loader_lazy = points_loader_reduced;

    return ret;
}


template <typename FeatureBase>
std::unordered_map<std::string, YAML::Node> ScanIO<FeatureBase>::loadChannelMetas(
    const size_t& scanPosNo, 
    const size_t& sensorNo,
    const size_t& scanNo) const
{
    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->scan(scanPosNo, sensorNo, scanNo);

    std::unordered_map<std::string, YAML::Node> channel_metas;

    // first hint: channels tag of scan meta
    if(d.meta)
    {
        YAML::Node meta;
        m_featureBase->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta);

        // std::cout << "loadChannelMetas - Loaded Meta: " << std::endl;
        // std::cout << meta << std::endl;

        if(meta["channels"])
        {
            for(auto it = meta["channels"].begin(); it != meta["channels"].end(); ++it)
            {
                std::string channel_name = it->as<std::string>();
                Description dc = Dgen->scanChannel(scanPosNo, sensorNo, scanNo, channel_name);

                // META
                if(dc.meta)
                {
                    YAML::Node cmeta;
                    m_featureBase->m_kernel->loadMetaYAML(*dc.metaRoot, *dc.meta, cmeta);


                    if(cmeta["name"])
                    {
                        channel_name = cmeta["name"].as<std::string>();
                    }

                    // std::cout << "First Hint found: " << channel_name << std::endl;
                    channel_metas[channel_name] = cmeta;
                }
            }
        }
    }
    
    // second hint: parse directory for channel metas
    Description dc = Dgen->scanChannel(scanPosNo, sensorNo, scanNo, "test");
    if(dc.meta)
    {
        std::string metaGroup = *dc.metaRoot;
        std::string metaFile = *dc.meta;
        std::tie(metaGroup, metaFile) = hdf5util::validateGroupDataset(metaGroup, metaFile);

        // std::cout << "Search for meta files in " << metaGroup << std::endl;

        for(auto meta : m_featureBase->m_kernel->metas(metaGroup, "channel"))
        {
            std::string channel_name = meta.first;

            if(meta.second["name"])
            {
                channel_name = meta.second["name"].template as<std::string>();
            }

            if(channel_metas.find(channel_name) == channel_metas.end())
            {
                // new channel found
                channel_metas[channel_name] = meta.second;
            }
        }
    }

    return channel_metas;
}   

template <typename FeatureBase>
void ScanIO<FeatureBase>::saveScan(
    const size_t& scanPosNo,
    const size_t& sensorNo,
    const size_t& scanNo,
    ScanPtr scanPtr) const
{
    save(scanPosNo, scanNo, scanPtr);
}

template <typename FeatureBase>
ScanPtr ScanIO<FeatureBase>::loadScan(
    const size_t& scanPosNo,
    const size_t& sensorNo,
    const size_t& scanNo) const
{
    return load(scanPosNo, sensorNo, scanNo);
}

template <typename FeatureBase>
ScanPtr ScanIO<FeatureBase>::loadScan(
    const size_t& scanPosNo, 
    const size_t& sensorNo,
    const size_t& scanNo, 
    ReductionAlgorithmPtr reduction) const
{
    ScanPtr ret = loadScan(scanPosNo, sensorNo, scanNo);

    if(ret)
    {
        // ret->points = ret->points_loader_reduced(reduction);
        if(ret->points)
        {
            reduction->setPointBuffer(ret->points);
            ret->points = reduction->getReducedPoints();
        }
    }

    return ret;
}

} // namespace lvr2
