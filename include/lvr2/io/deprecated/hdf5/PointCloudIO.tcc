namespace lvr2 {

namespace hdf5features {

template<typename Derived>
void PointCloudIO<Derived>::save(std::string name, const PointBufferPtr& buffer)
{
    HighFive::Group g = hdf5util::getGroup(
        m_file_access->m_hdf5_file,
        name,
        true
    );

    save(g, buffer);
}

template<typename Derived>
void PointCloudIO<Derived>::save(HighFive::Group& group, const PointBufferPtr& buffer)
{
    std::string id(PointCloudIO<Derived>::ID);
    std::string obj(PointCloudIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    for(auto elem : *buffer)
    {
        m_vchannel_io->save(group, elem.first, elem.second);
    }
}


template<typename Derived>
PointBufferPtr PointCloudIO<Derived>::load(std::string name)
{
    PointBufferPtr ret;

    if(hdf5util::exist(m_file_access->m_hdf5_file, name))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, name, false);
        ret = load(g);
    } 

    return ret;
}

template<typename Derived>
PointBufferPtr PointCloudIO<Derived>::loadPointCloud(std::string name)
{
    return load(name);
}

template<typename Derived>
PointBufferPtr PointCloudIO<Derived>::load(HighFive::Group& group)
{
    PointBufferPtr ret;

    // check if flags are correct
    if(!isPointCloud(group) )
    {
        std::cout << "[Hdf5IO - PointCloudIO] WARNING: flags of " << group.getId() << " are not correct." << std::endl;
        return ret;
    }

    for(auto name : group.listObjectNames() )
    {
        std::unique_ptr<HighFive::DataSet> dataset;

        try {
            dataset = std::make_unique<HighFive::DataSet>(
                group.getDataSet(name)
            );
        } catch(HighFive::DataSetException& ex) {

        }

        if(dataset)
        {
            // name is dataset
            boost::optional<PointBuffer::val_type> opt_vchannel
                 = m_vchannel_io->template load<PointBuffer::val_type>(group, name);
            
            if(opt_vchannel)
            {
                if(!ret)
                {
                    ret.reset(new PointBuffer);
                }
                ret->insert({
                    name,
                    *opt_vchannel
                });
            }
            
        }

    }

    return ret;
}

template<typename Derived>
bool PointCloudIO<Derived>::isPointCloud(
    HighFive::Group& group)
{
    // std::string id(PointCloudIO<Derived>::ID);
    // std::string obj(PointCloudIO<Derived>::OBJID);
    // return hdf5util::checkAttribute(group, "IO", id)
    //     && hdf5util::checkAttribute(group, "CLASS", obj);

    return true;
}

} // hdf5features

} // namespace lvr2 