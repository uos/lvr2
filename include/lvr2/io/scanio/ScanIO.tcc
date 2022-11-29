namespace lvr2
{

namespace scanio
{

template <typename BaseIO>
void ScanIO<BaseIO>::save(
    const size_t &scanPosNo,
    const size_t &sensorNo,
    const size_t &scanNo,
    ScanPtr scanPtr) const
{
    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->scan(scanPosNo, sensorNo, scanNo);

    // lvr2::logout::get() << "[ScanIO - save]" << lvr2::endl;
    // lvr2::logout::get() << d << lvr2::endl;

    if (!d.dataRoot)
    {
        return;
    }

    const bool data_loaded_before = scanPtr->loaded();

    if (!data_loaded_before)
    {
        scanPtr->load();
    }

    // lvr2::logout::get() << "ScanIO - save data " << scanPosNo << " " << sensorNo << " " << scanNo << lvr2::endl;
    //// DATA
    if (scanPtr->points)
    {
        if (d.data)
        {
            // lvr2::logout::get() << "Save Channel wise" << lvr2::endl;
            m_pclIO->save(*d.dataRoot, *d.data, scanPtr->points);

            // save metas
            for (auto elem : *scanPtr->points)
            {
                Description dc = Dgen->scanChannel(scanPosNo, sensorNo, scanNo, elem.first);
                // Meta
                if (dc.meta)
                {
                    YAML::Node meta;
                    meta = elem.second;
                    meta["name"] = elem.first;
                    m_baseIO->m_kernel->saveMetaYAML(*dc.metaRoot, *dc.meta, meta);
                }
            }
        }
        else
        {
            // lvr2::logout::get() << "Save Partial" << lvr2::endl;
            // a lot of code for the problem of capsulating SOME channels into one PLY
            // there could be other channels that do not fit in this ply

            // Scan is not a dataset: handle as group of channels
            // m_pclIO->save(scanPosNo, sensorNo, scanNo, scanPtr->points);
            std::unordered_map<std::string, PointBufferPtr> one_file_multi_channel;

            /// Data (Channel)
            for (auto elem : *scanPtr->points)
            {
                // lvr2::logout::get() << "Save " << elem.first << lvr2::endl;
                Description dc = Dgen->scanChannel(scanPosNo, sensorNo, scanNo, elem.first);
                boost::filesystem::path proot(*dc.dataRoot);

                // lvr2::logout::get() << "Description of " <<  elem.first << lvr2::endl;
                // lvr2::logout::get() << dc << lvr2::endl;

                if (proot.extension() != "")
                {
                    if (one_file_multi_channel.find(proot.string()) == one_file_multi_channel.end())
                    {
                        one_file_multi_channel[proot.string()] = PointBufferPtr(new PointBuffer);
                    }
                    (*one_file_multi_channel[proot.string()])[elem.first] = elem.second;
                }
                else
                {
                    // group is no file
                    // concatenate group and dataset and store directly
                    // std::string filename = (proot / *dc.data).string();
                    // lvr2::logout::get() << "Store single channel " << elem.first << " to " << filename << lvr2::endl;

                    std::string group, dataset;
                    std::tie(group, dataset) = hdf5util::validateGroupDataset(proot.string(), *dc.data);

                    // lvr2::logout::get() << "Save " << elem.first << " to " << group << " - " << dataset << lvr2::endl;

                    // Data
                    m_vchannel_io->save(group, dataset, elem.second);
                }
            }

            // DATA of exceptional "one_file_multi_channel". there we need to call the
            // kernel directly, because we are now at file level
            for (auto ex_elem : one_file_multi_channel)
            {
                std::string group, name;
                std::tie(group, name) = hdf5util::validateGroupDataset("", ex_elem.first);

                // lvr2::logout::get() << "Save Channel Group to file: " << ex_elem.first << lvr2::endl;
                // lvr2::logout::get() << *ex_elem.second << lvr2::endl;

                m_baseIO->m_kernel->savePointBuffer(group, name, ex_elem.second);
            }

            // lvr2::logout::get() << "Save Channel METAs" << lvr2::endl;

            // META
            // save meta for each channel
            for (auto elem : *scanPtr->points)
            {
                Description dc = Dgen->scanChannel(scanPosNo, sensorNo, scanNo, elem.first);
                if (dc.meta)
                {
                    YAML::Node meta;
                    meta = elem.second;
                    meta["name"] = elem.first;
                    m_baseIO->m_kernel->saveMetaYAML(*dc.metaRoot, *dc.meta, meta);
                }
            }

            // lvr2::logout::get() << "END channels" << lvr2::endl;
        }
    }

    if (!data_loaded_before)
    {
        scanPtr->release();
    }

    // lvr2::logout::get() << "save SCAN META" << lvr2::endl;

    //// META
    if (d.meta)
    {
        YAML::Node node;
        node = *scanPtr;

        // lvr2::logout::get() << "write YAML" << lvr2::endl;
        m_baseIO->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }

    // lvr2::logout::get() << "Success" << lvr2::endl;
}

template <typename BaseIO>
boost::optional<YAML::Node> ScanIO<BaseIO>::loadMeta(
    const size_t &scanPosNo,
    const size_t &sensorNo,
    const size_t &scanNo) const
{
    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->scan(scanPosNo, sensorNo, scanNo);
    return m_metaIO->load(d);
}

template <typename BaseIO>
ScanPtr ScanIO<BaseIO>::load(
    const size_t &scanPosNo,
    const size_t &sensorNo,
    const size_t &scanNo) const
{
    ScanPtr ret;

    // Get Description of Scan Location

    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->scan(scanPosNo, sensorNo, scanNo);

    if (!d.dataRoot)
    {
        return ret;
    }
    if (!m_baseIO->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }

    // lvr2::logout::get() << "[ScanIO - load] Description:" << lvr2::endl;
    // lvr2::logout::get() << d << lvr2::endl;

    /// META
    if (d.meta)
    {
        YAML::Node meta;
        if (!m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }

        try
        {
            ret = std::make_shared<Scan>(meta.as<Scan>());
        }
        catch (const YAML::TypedBadConversion<Scan> &ex)
        {
            lvr2::logout::get() << lvr2::error << "[ScanIO - load] Scan (" << scanPosNo << ", " << sensorNo << ", " << scanNo << ") : Could not decode YAML as Scan." << lvr2::endl;
            throw ex;
        }
    }
    else
    {
        // for schemas without meta information
        ret = std::make_shared<Scan>();
    }

    // lvr2::logout::get() << "[ScanIO - load] Meta loaded." << lvr2::endl;
    // lvr2::logout::get() << "- points: " << ret->numPoints << lvr2::endl;

    std::function<PointBufferPtr()> points_loader;
    std::function<void(ScanPtr)> points_saver;

    // Creating a point saver lambda with information about
    // the position of the current scan within the scan project
    points_saver = [t = m_baseIO->shared_from_this(), scanPosNo, sensorNo, scanNo](ScanPtr p)
    {
        lvr2::logout::get() << lvr2::info << "[Point Saver (2)]: Saving scan " << scanNo
                  << " of LiDAR " << sensorNo << " of scan position " << scanPosNo << lvr2::endl;
        t->ScanIO<BaseIO>::save(scanPosNo, sensorNo, scanNo, p);
    };

    if (d.data)
    {
        // Here we need to keep the actual instance of the base io
        // alive to assure that the original source paths are used
        // to load the data. Even if the original loader goes out of
        // scope the shared_from_this will keep it alive until the
        // points_loaded function is freed
        points_loader = [t = m_baseIO->shared_from_this(), d]()
        {
            return t->PointCloudIO<BaseIO>::load(*d.dataRoot, *d.data);
        };
    }
    else
    {
        points_loader = [schema = m_baseIO->m_description,
                            kernel = m_baseIO->m_kernel,
                            scanPosNo,
                            sensorNo,
                            scanNo]()
        {
            PointBufferPtr points;

            if (!schema)
            {
                lvr2::logout::get() << lvr2::warning << "[Point Loader]: Schema empty" << lvr2::endl;
            }

            if (!kernel)
            {
                lvr2::logout::get() << lvr2::warning << "[Point Loader]: Kernel empty" << lvr2::endl;
            }

            FeatureBuild<ScanIO> io(kernel, schema, false);

            auto channel_metas = io.loadChannelMetas(scanPosNo, sensorNo, scanNo);

            if (!channel_metas.empty())
            {
                for (auto elem : channel_metas)
                {
                    // check if element was added already
                    if (!points || points->find(elem.first) == points->end())
                    {
                        Description dc = schema->scanChannel(scanPosNo, sensorNo, scanNo, elem.first);

                        // data is at dc.dataRoot / dc.data

                        boost::filesystem::path proot(*dc.dataRoot);

                        if (proot.extension() != "")
                        {
                            // channels in file
                            std::string group, name;
                            std::tie(group, name) = hdf5util::validateGroupDataset("", proot.string());
                            PointBufferPtr points_ = kernel->loadPointBuffer(group, name);

                            // merge to complete map
                            if (!points)
                            {
                                points = points_;
                            }
                            else
                            {
                                for (auto elem : *points_)
                                {
                                    (*points)[elem.first] = elem.second;
                                }
                            }
                        }
                        else
                        {
                            // channels in folder
                            auto vo = io.template loadVariantChannel<typename PointBuffer::val_type>(*dc.dataRoot, *dc.data);
                            if (vo)
                            {
                                if (!points)
                                {
                                    points.reset(new PointBuffer);
                                }
                                (*points)[elem.first] = *vo;
                            }
                        }
                    }
                }
            }
            else
            {

                // lvr2::logout::get() << "Could not get channel metas" << lvr2::endl;

                // no meta information about channels
                // could be in case of datasets cannot be

                // but we know that points must be there
                Description dc = schema->scanChannel(scanPosNo, sensorNo, scanNo, "points");

                // search for data root
                boost::filesystem::path proot(*dc.dataRoot);

                if (proot.extension() != "")
                {
                    std::string group, dataset;
                    std::tie(group, dataset) = hdf5util::validateGroupDataset("", proot.string());
                    points = kernel->loadPointBuffer(group, dataset);
                }
                else
                {
                    // search
                    if (dc.data)
                    {
                        boost::filesystem::path pdata = *dc.data;
                        if (pdata.extension() != "")
                        {
                            // found potential file to filter for
                            for (auto name : kernel->listDatasets(proot.string()))
                            {
                                PointBufferPtr points_ = kernel->loadPointBuffer(proot.string(), name);

                                if (!points)
                                {
                                    points = points_;
                                }
                                else if (points_)
                                {
                                    for (auto elem : *points_)
                                    {
                                        (*points)[elem.first] = elem.second;
                                    }
                                }
                            }
                        }
                        else
                        {
                            // situation:
                            // no extension of group and no extension of dataset
                            // no meta data
                            // there are two options what happend here
                            // 1. Used Hdf5 schema and did not find any meta data
                            //    - this should not happen. meta data must be available
                            // 2. Used directory schema and stored binary channels
                            //    - this should not happen. binary channels must have an meta file

                            lvr2::logout::get() << lvr2::error << "[ScanIO - load] Could not load file by description: " << lvr2::endl;
                            lvr2::logout::get() << lvr2::error << dc << lvr2::endl;

                            throw std::runtime_error("[ScanIO - Panic. Something orrured that should not happen]");
                        }
                    }
                }
            }

            return points;
        }; // points_loader lambda
    }

    // add reduced version
    std::function<PointBufferPtr(ReductionAlgorithmPtr)> points_loader_reduced = [points_loader](ReductionAlgorithmPtr red)
    {
        PointBufferPtr points = points_loader();

        if (points)
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

    if (m_baseIO->m_load_data)
    {
        ret->points = points_loader();
    }

    ret->points_saver = points_saver;
    ret->points_loader = points_loader;
    ret->points_loader_reduced = points_loader_reduced;

    return ret;
}

template <typename BaseIO>
std::unordered_map<std::string, YAML::Node> ScanIO<BaseIO>::loadChannelMetas(
    const size_t &scanPosNo,
    const size_t &sensorNo,
    const size_t &scanNo) const
{
    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->scan(scanPosNo, sensorNo, scanNo);

    // lvr2::logout::get() << "loadChannelMetas from description" << lvr2::endl;
    // lvr2::logout::get() << d << lvr2::endl;

    std::unordered_map<std::string, YAML::Node> channel_metas;

    // first hint: channels tag of scan meta
    if (d.meta)
    {
        YAML::Node meta;

        // cout << m_baseIO << endl;
        // cout << m_baseIO->m_kernel << endl;
        // cout << m_baseIO->m_kernel->fileResource() << endl;

        m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta);

        // lvr2::logout::get() << "loadChannelMetas - Loaded Meta: " << lvr2::endl;
        // lvr2::logout::get() << meta << lvr2::endl;

        if (meta["channels"])
        {
            for (auto it = meta["channels"].begin(); it != meta["channels"].end(); ++it)
            {
                std::string channel_name = it->as<std::string>();
                Description dc = Dgen->scanChannel(scanPosNo, sensorNo, scanNo, channel_name);

                // META
                if (dc.meta)
                {
                    YAML::Node cmeta;
                    m_baseIO->m_kernel->loadMetaYAML(*dc.metaRoot, *dc.meta, cmeta);

                    if (cmeta["name"])
                    {
                        channel_name = cmeta["name"].as<std::string>();
                    }

                    // lvr2::logout::get() << "First Hint found: " << channel_name << lvr2::endl;
                    channel_metas[channel_name] = cmeta;
                }
            }
        }
    }

    // second hint: parse directory for channel metas
    Description dc = Dgen->scanChannel(scanPosNo, sensorNo, scanNo, "test");
    if (dc.meta)
    {
        std::string metaGroup = *dc.metaRoot;
        std::string metaFile = *dc.meta;
        std::tie(metaGroup, metaFile) = hdf5util::validateGroupDataset(metaGroup, metaFile);

        // lvr2::logout::get() << "Search for meta files in " << metaGroup << lvr2::endl;
        for (auto meta : m_baseIO->m_kernel->metas(metaGroup, "channel"))
        {
            std::string channel_name = meta.first;

            if (meta.second["name"])
            {
                channel_name = meta.second["name"].template as<std::string>();
            }

            if (channel_metas.find(channel_name) == channel_metas.end())
            {
                // new channel found
                channel_metas[channel_name] = meta.second;
            }
        }
    }

    return channel_metas;
}

template <typename BaseIO>
void ScanIO<BaseIO>::saveScan(
    const size_t &scanPosNo,
    const size_t &sensorNo,
    const size_t &scanNo,
    ScanPtr scanPtr) const
{
    save(scanPosNo, scanNo, scanPtr);
}

template <typename BaseIO>
ScanPtr ScanIO<BaseIO>::loadScan(
    const size_t &scanPosNo,
    const size_t &sensorNo,
    const size_t &scanNo) const
{
    return load(scanPosNo, sensorNo, scanNo);
}

template <typename BaseIO>
ScanPtr ScanIO<BaseIO>::loadScan(
    const size_t &scanPosNo,
    const size_t &sensorNo,
    const size_t &scanNo,
    ReductionAlgorithmPtr reduction) const
{
    ScanPtr ret = loadScan(scanPosNo, sensorNo, scanNo);

    if (ret)
    {
        // ret->points = ret->points_loader_reduced(reduction);

        if (ret->points)
        {
            reduction->setPointBuffer(ret->points);
            ret->points = reduction->getReducedPoints();
        }
        else if (ret->points_loader_reduced)
        {
            ret->points = ret->points_loader_reduced(reduction);
        }
        else if (ret->points_loader)
        {
            ret->load();
            if (ret->points)
            {
                reduction->setPointBuffer(ret->points);
                ret->points = reduction->getReducedPoints();
            }
        }
    }

    return ret;
    }

} // namespace scanio

} // namespace lvr2
