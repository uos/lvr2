#include <boost/filesystem.hpp>

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
void MeshIO<Derived>::save(
  HighFive::Group& group,
  const MeshBufferPtr& buffer)
{
    std::string id(MeshIO<Derived>::ID);
    std::string obj(MeshIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    if (!group.exist(m_geometry_group))
    {
        group.createGroup(m_geometry_group);
    }
    HighFive::Group channelsGroup = group.getGroup(m_geometry_group);
    for (auto elem : *buffer)
    {
        m_vchannel_io->save(channelsGroup, elem.first, elem.second);
    }

    if(!buffer->getTextures().empty())
    {
        if (!group.exist(m_textures_group))
        {
            group.createGroup(m_textures_group);
        }
        HighFive::Group texturesGroup = group.getGroup(m_textures_group);

        for (const Texture& texture : buffer->getTextures())
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
    
    if(!buffer->getMaterials().empty())
    {
        if (!group.exist(m_materials_group))
        {
            group.createGroup(m_materials_group);
        }
        HighFive::Group materialsGroup = group.getGroup(m_materials_group);


        size_t numMaterials = buffer->getMaterials().size();
        boost::shared_array<int> textureHandles(new int[numMaterials]);

        boost::shared_array<int16_t> RGB8Color(new int16_t[numMaterials * 3]);
        lvr2::Material material;
        for (int i = 0; i < numMaterials; i++)
        {
            material = buffer->getMaterials().at(i);

            // Both material fields are optional. We save -1, if they are not initialized.
            textureHandles.get()[i] = (material.m_texture) ? (int) material.m_texture->idx() : -1;;

            if (material.m_color)
            {
                RGB8Color.get()[3 * i    ] = static_cast<int16_t>(material.m_color.get()[0]);
                RGB8Color.get()[3 * i + 1] = static_cast<int16_t>(material.m_color.get()[1]);
                RGB8Color.get()[3 * i + 2] = static_cast<int16_t>(material.m_color.get()[2]);
            }
            else
            {
                RGB8Color.get()[3 * i    ] = -1;
                RGB8Color.get()[3 * i + 1] = -1;
                RGB8Color.get()[3 * i + 2] = -1;
            }
        }

        std::vector <size_t> numMat{numMaterials}; // TODO: what is this supposed to do? Storing empty buffers?
        std::vector <hsize_t> chunkMat{numMaterials};
        std::vector <size_t> dimensionsRgb{numMaterials, 3};
        std::vector <hsize_t> chunkRgb{numMaterials, 3};
        m_array_io->save(materialsGroup, "texture_handles", numMat, chunkMat, textureHandles);
        m_array_io->save(materialsGroup, "rgb_color", dimensionsRgb, chunkRgb, RGB8Color);
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
    m_mesh_name = name;
    return load(name);
}

template <typename Derived>
MeshBufferPtr MeshIO<Derived>::load(HighFive::Group& group)
{
    MeshBufferPtr ret = nullptr;

    if (!isMesh(group))
    {
        std::cout << "[Hdf5IO - MeshIO] WARNING: flags of " << group.getId() << " are not correct."
                  << std::endl;
        return ret;
    }

    if (group.exist(m_geometry_group))
    {
        HighFive::Group channelsGroup = group.getGroup(m_geometry_group);

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

    if (ret == nullptr)
    {
        return nullptr;
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

    if (group.exist(m_geometry_group))
    {
        HighFive::Group materialsGroup = group.getGroup(m_geometry_group);

        std::vector<lvr2::Material> materials;
        std::vector<size_t> dimensionColor;
        std::vector<size_t> dimensionTextureHandle;
        boost::shared_array<int16_t> materialColor;
        boost::shared_array<int> materialTexture;
        if(materialsGroup.exist("rgb_color"))
        {
            materialColor = m_array_io->template load<int16_t>(materialsGroup, "rgb_color", dimensionColor);
        }
        if(materialsGroup.exist("texture_handles"))
        {
            materialTexture = m_array_io->template load<int>(materialsGroup, "texture_handles", dimensionTextureHandle);
        }
        if(materialColor && materialTexture)
        {
            if(dimensionColor.at(0) != dimensionTextureHandle.at(0) || dimensionColor.at(1) != 3)
            {
                std::cout << "[Hdf5IO - MeshIO] WARNING: Wrong material dimensions. Materials will not be loaded." << std::endl;
            }
            else
            {
                for(int i = 0; i < dimensionTextureHandle.at(0); i++)
                {
                    lvr2::Material nextMat;
                    if(materialColor.get()[i * 3] != -1)
                    {
                        nextMat.m_color = boost::optional<lvr2::RGB8Color>(
                                {static_cast<uint8_t>(materialColor.get()[i * 3    ]),
                                 static_cast<uint8_t>(materialColor.get()[i * 3 + 1]),
                                 static_cast<uint8_t>(materialColor.get()[i * 3 + 2])});
                    }
                    if(materialTexture.get()[i] != -1)
                    {
                        nextMat.m_texture = boost::optional<lvr2::TextureHandle>(materialTexture.get()[i]);
                    }
                    materials.push_back(nextMat);
                }
                ret->setMaterials(materials);
            }
        }
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

template <typename Derived>
void MeshIO<Derived>::setMeshName(std::string meshName)
{
    m_mesh_name = meshName;
}

template <typename Derived>
FloatChannelOptional MeshIO<Derived>::getVertices()
{
    namespace bfs = boost::filesystem;
    bfs::path p = bfs::path(m_mesh_name) / m_geometry_group;
    return m_channel_io->template load<float>(p.string(), "vertices");
}

template <typename Derived>
IndexChannelOptional MeshIO<Derived>::getIndices()
{
    namespace bfs = boost::filesystem;
    bfs::path p = bfs::path(m_mesh_name) / m_geometry_group;
    return m_channel_io->template load<unsigned int>(p.string(), "face_indices");
}

template <typename Derived>
bool MeshIO<Derived>::addVertices(const FloatChannel& channel)
{
    // create new mesh group if noone is existing
    if(!hdf5util::exist(m_file_access->m_hdf5_file, m_mesh_name))
    {
        // create group if not existing
        HighFive::Group mesh_group = hdf5util::getGroup(m_file_access->m_hdf5_file, m_mesh_name, true);
        std::string id(MeshIO<Derived>::ID);
        std::string obj(MeshIO<Derived>::OBJID);
        hdf5util::setAttribute(mesh_group, "IO", id);
        hdf5util::setAttribute(mesh_group, "CLASS", obj);
    }

    // use existing mesh group
    HighFive::Group mesh_group = hdf5util::getGroup(m_file_access->m_hdf5_file, m_mesh_name, false);
    if(!isMesh(mesh_group))
    {
        throw std::runtime_error("[MeshIO] tried to save vertices into a none mesh group!");
    }

    // and save the face indices to it
    namespace bfs = boost::filesystem;
    bfs::path p = bfs::path(m_mesh_name) / m_geometry_group;
    m_channel_io->save(p.string(), "vertices", channel);
    return true;
}

template <typename Derived>
bool MeshIO<Derived>::addIndices(const IndexChannel& channel)
{
    // create new mesh group if noone is existing
    if(!hdf5util::exist(m_file_access->m_hdf5_file, m_mesh_name))
    {
        // create group if not existing
        HighFive::Group mesh_group = hdf5util::getGroup(m_file_access->m_hdf5_file, m_mesh_name, true);
        std::string id(MeshIO<Derived>::ID);
        std::string obj(MeshIO<Derived>::OBJID);
        hdf5util::setAttribute(mesh_group, "IO", id);
        hdf5util::setAttribute(mesh_group, "CLASS", obj);
    }

    // use existing mesh group
    HighFive::Group mesh_group = hdf5util::getGroup(m_file_access->m_hdf5_file, m_mesh_name, false);
    if(!isMesh(mesh_group))
    {
        throw std::runtime_error("[MeshIO] tried to save face_indices into a none mesh group!");
    }

    // and save the face indices to it
    namespace bfs = boost::filesystem;
    bfs::path p = bfs::path(m_mesh_name) / m_geometry_group;
    m_channel_io->save(p.string(), "face_indices", channel);
    return true;
}

template<typename Derived>
template <typename T>
bool MeshIO<Derived>::getChannel(
  const std::string group,
  const std::string name,
  boost::optional<AttributeChannel<T> >& channel)
{
    namespace bfs = boost::filesystem;
    bfs::path p = bfs::path(m_mesh_name) / group;
    channel = m_channel_io->template load<T>(p.string(), name);
    return (bool)channel;
}

template<typename Derived>
template <typename T>
bool MeshIO<Derived>::addChannel(const std::string group, const std::string name, const AttributeChannel<T>& channel)
{
    namespace bfs = boost::filesystem;
    bfs::path p = bfs::path(m_mesh_name) / group;
    m_channel_io->save(p.string(), name, channel);
    return true;
}

template<typename Derived>
bool MeshIO<Derived>::getChannel(const std::string group, const std::string name, FloatChannelOptional& channel)
{
    return getChannel<float>(group, name, channel);
}

template<typename Derived>
bool MeshIO<Derived>::getChannel(const std::string group, const std::string name, IndexChannelOptional& channel)
{
    return getChannel<unsigned int>(group, name, channel);
}

template<typename Derived>
bool MeshIO<Derived>::getChannel(const std::string group, const std::string name, UCharChannelOptional& channel)
{
    return getChannel<unsigned char>(group, name, channel);
}

template<typename Derived>
bool MeshIO<Derived>::addChannel(const std::string group, const std::string name, const FloatChannel& channel)
{
    return addChannel<float>(group, name, channel);
}

template<typename Derived>
bool MeshIO<Derived>::addChannel(const std::string group, const std::string name, const IndexChannel& channel)
{
    return addChannel<unsigned int>(group, name, channel);
}

template<typename Derived>
bool MeshIO<Derived>::addChannel(const std::string group, const std::string name, const UCharChannel& channel)
{
    return addChannel<unsigned char>(group, name, channel);
}

} // namespace hdf5features

} // namespace lvr2
