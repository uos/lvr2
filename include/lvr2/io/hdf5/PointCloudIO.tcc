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
    std::cout << "[Hdf5IO - PointCloudIO] save" << std::endl;

    for(auto it = buffer->typedBegin< char >(); it != buffer->end(); ++it) {
        m_channel_io->save(group, it->first, it->second);
    }

    for(auto it = buffer->typedBegin< unsigned char >(); it != buffer->end(); ++it){
        m_channel_io->save(group, it->first, it->second);
    }


    for(auto it = buffer->typedBegin< short >(); it != buffer->end(); ++it){
        m_channel_io->save(group, it->first, it->second);
    }


    for(auto it = buffer->typedBegin< unsigned short >(); it != buffer->end(); ++it){
        m_channel_io->save(group, it->first, it->second);
    }


    for(auto it = buffer->typedBegin< int >(); it != buffer->end(); ++it){
        m_channel_io->save(group, it->first, it->second);
    }

    for(auto it = buffer->typedBegin< unsigned int >(); it != buffer->end(); ++it){
        m_channel_io->save(group, it->first, it->second);
    }

    for(auto it = buffer->typedBegin< float >(); it != buffer->end(); ++it){
        m_channel_io->save(group, it->first, it->second);
    }

    for(auto it = buffer->typedBegin< double >(); it != buffer->end(); ++it){
        m_channel_io->save(group, it->first, it->second);
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

// R == 0
template<typename Derived, int R, typename std::enable_if<R == 0, void>::type* = nullptr>
PointBuffer::val_type loadChannel(
    HighFive::DataType dtype,
    ChannelIO<Derived>* channel_io,
    HighFive::Group& group,
    std::string name)
{
    if(dtype == HighFive::AtomicType<PointBuffer::type_of_index<R> >())
    {
        PointBuffer::val_type ret;
        ret = *channel_io->template load<PointBuffer::type_of_index<R> >(group, name);
        return ret;
    } else {
        return PointBuffer::val_type();
    }
}

// R != 0
template<typename Derived, int R, typename std::enable_if<R != 0, void>::type* = nullptr>
PointBuffer::val_type loadChannel(
    HighFive::DataType dtype,
    ChannelIO<Derived>* channel_io,
    HighFive::Group& group,
    std::string name)
{
    if(dtype == HighFive::AtomicType<PointBuffer::type_of_index<R> >())
    {
        PointBuffer::val_type ret;
        ret = *channel_io->template load<PointBuffer::type_of_index<R> >(group, name);
        return ret;
    } else {
        return loadChannel<Derived, R-1>(dtype, channel_io, group, name);
    }
}


template<typename Derived>
PointBuffer::val_type PointCloudIO<Derived>::loadDynamic(
    HighFive::DataType dtype,
    HighFive::Group& group,
    std::string name)
{
    return loadChannel<Derived, PointBuffer::num_types-1>(dtype, m_channel_io, group, name);
}

template<typename Derived>
PointBufferPtr PointCloudIO<Derived>::load(HighFive::Group& group)
{
    std::cout << "[Hdf5IO - PointCloudIO] load" << std::endl;
    PointBufferPtr ret;


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
            if(!ret)
            {
                ret.reset(new PointBuffer);
            }
            // name is dataset
            ret->insert({
                name,
                loadDynamic(dataset->getDataType(), group, name)
            });
        }

    }

    // TODO
    return ret;
}

} // hdf5features

} // namespace lvr2 