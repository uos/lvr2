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
        ret = load(g, name);
    } 

    // TODO
    return ret;
}

template<typename Derived>
PointBufferPtr PointCloudIO<Derived>::load(HighFive::Group& group)
{
    std::cout << "[Hdf5IO - PointCloudIO] load" << std::endl;
    PointBufferPtr ret;



    // TODO
    return ret;
}

} // hdf5features

} // namespace lvr2 