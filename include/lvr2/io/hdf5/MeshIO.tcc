namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void MeshIO<Derived>::save(std::string name, const MeshBufferPtr& buffer)
{
    HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, name, true);

    save(g, buffer);
}

template <typename Derived>
void MeshIO<Derived>::save(HighFive::Group& group, const MeshBufferPtr& buffer)
{
    std::string id(MeshIO<Derived>::ID);
    std::string obj(MeshIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    if (!group.exist("channels"))
    {
        group.createGroup("channels");
    }
    HighFive::Group channelsGroup = group.getGroup("channels");

    for (auto elem : *buffer)
    {
        m_vchannel_io->save(channelsGroup, elem.first, elem.second);
    }

    if (!group.exist("textures"))
    {
        group.createGroup("textures");
    }
    HighFive::Group texturesGroup = group.getGroup("textures");

    for (auto texture : buffer->getTextures())
    {
        auto dims   = std::vector<size_t>{texture.m_width, texture.m_height, texture.m_numChannels};
        auto chunks = std::vector<hsize_t>{dims[0], dims[1], dims[2]};
        auto data
            = boost::shared_array<unsigned char>(new unsigned char[dims[0] * dims[1] * dims[2]]);
        memcpy(data.get(), texture.m_data, dims[0] * dims[1] * dims[2]);
        m_array_io->template save<unsigned char>(
            texturesGroup, std::to_string(texture.m_index), dims, chunks, data);
    }
}

template <typename Derived>
MeshBufferPtr MeshIO<Derived>::load(std::string name)
{
    MeshBufferPtr ret;

    if (hdf5util::exist(m_file_access->m_hdf5_file, name))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, name, false);
        ret               = load(g);
    }

    return ret;
}

template <typename Derived>
MeshBufferPtr MeshIO<Derived>::loadMesh(std::string name)
{
    return load(name);
}

template <typename Derived>
MeshBufferPtr MeshIO<Derived>::load(HighFive::Group& group)
{
    MeshBufferPtr ret;

    if (!isMesh(group))
    {
        std::cout << "[Hdf5IO - MeshIO] WARNING: flags of " << group.getId() << " are not correct."
                  << std::endl;
        return ret;
    }

    if (group.exist("channels"))
    {
        HighFive::Group channelsGroup = group.getGroup("channels");

        for (auto name : channelsGroup.listObjectNames())
        {
            std::unique_ptr<HighFive::DataSet> dataset;

            try
            {
                dataset = std::make_unique<HighFive::DataSet>(channelsGroup.getDataSet(name));
            }
            catch (HighFive::DataSetException& ex)
            {
            }

            if (dataset)
            {
                // name is dataset
                boost::optional<MeshBuffer::val_type> opt_vchannel
                    = m_vchannel_io->template load<MeshBuffer::val_type>(channelsGroup, name);

                if (opt_vchannel)
                {
                    if (!ret)
                    {
                        ret.reset(new MeshBuffer);
                    }
                    ret->insert({name, *opt_vchannel});
                }
            }
        }
    }

    if (group.exist("textures"))
    {
        HighFive::Group texturesGroup = group.getGroup("textures");

        std::vector<lvr2::Texture> textures;

        for (auto name : texturesGroup.listObjectNames())
        {
            // dimensions of texture: width, height, numChannels
            std::vector<size_t> dimensions;

            // name is texture id
            boost::shared_array<unsigned char> textureData
                = m_array_io->template load<unsigned char>(texturesGroup, name, dimensions);

            if (dimensions.size() == 3) {
                textures.push_back(Texture(stoi(name),
                                           dimensions[0],
                                           dimensions[1],
                                           dimensions[2],
                                           1,
                                           1.0,
                                           textureData.get()));
            }
        }

        ret->setTextures(textures);
    }

    return ret;
}

template <typename Derived>
bool MeshIO<Derived>::isMesh(
    HighFive::Group& group)
{
    std::string id(MeshIO<Derived>::ID);
    std::string obj(MeshIO<Derived>::OBJID);
    return hdf5util::checkAttribute(group, "IO", id)
        && hdf5util::checkAttribute(group, "CLASS", obj);
}

} // namespace hdf5features

} // namespace lvr2
