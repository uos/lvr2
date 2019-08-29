namespace lvr2 {

namespace hdf5features {

template<typename Derived>
void MeshIO<Derived>::save(std::string name, const MeshBufferPtr& buffer)
{
    HighFive::Group g = hdf5util::getGroup(
        m_file_access->m_hdf5_file,
        name,
        true
    );

    save(g, buffer);
}

template<typename Derived>
void MeshIO<Derived>::save(HighFive::Group& group, const MeshBufferPtr& buffer)
{
    for(auto elem : *buffer)
    {
        m_vchannel_io->save(group, elem.first, elem.second);
    }
}


template<typename Derived>
MeshBufferPtr MeshIO<Derived>::load(std::string name)
{
    MeshBufferPtr ret;

    if(hdf5util::exist(m_file_access->m_hdf5_file, name))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, name, false);
        ret = load(g);
    }

    return ret;
}

template<typename Derived>
MeshBufferPtr MeshIO<Derived>::loadMesh(std::string name)
{
    return load(name);
}

template<typename Derived>
MeshBufferPtr MeshIO<Derived>::load(HighFive::Group& group)
{
    MeshBufferPtr ret;

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
            boost::optional<MeshBuffer::val_type> opt_vchannel
                 = m_vchannel_io->template load<MeshBuffer::val_type>(group, name);

            if(opt_vchannel)
            {
                if(!ret)
                {
                    ret.reset(new MeshBuffer);
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

} // hdf5features

} // namespace lvr2
