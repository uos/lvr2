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

    if(!buffer->getTextures().empty())
    {
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
    
    if(!buffer->getMaterials().empty())
    {
        if (!group.exist("materials"))
        {
            group.createGroup("materials");
        }
        HighFive::Group materialsGroup = group.getGroup("materials");


        size_t numMaterials = buffer->getMaterials().size();
        boost::shared_array<int> textureHandles(new int[numMaterials]);

        boost::shared_array<int16_t> rgb8Color(new int16_t[numMaterials * 3]);
        lvr2::Material material;
        for (int i = 0; i < numMaterials; i++)
        {
            material = buffer->getMaterials().at(i);

            // Both material fields are optional. We save -1, if they are not initialized.
            textureHandles.get()[i] = (material.m_texture) ? (int) material.m_texture->idx() : -1;;

            if (material.m_color)
            {
                rgb8Color.get()[3 * i    ] = static_cast<int16_t>(material.m_color.get()[0]);
                rgb8Color.get()[3 * i + 1] = static_cast<int16_t>(material.m_color.get()[1]);
                rgb8Color.get()[3 * i + 2] = static_cast<int16_t>(material.m_color.get()[2]);
            }
            else
            {
                rgb8Color.get()[3 * i    ] = -1;
                rgb8Color.get()[3 * i + 1] = -1;
                rgb8Color.get()[3 * i + 2] = -1;
            }
        }

        std::vector <size_t> numMat{numMaterials};
        std::vector <hsize_t> chunkMat{numMaterials};
        std::vector <size_t> dimensionsRgb{numMaterials, 3};
        std::vector <hsize_t> chunkRgb{numMaterials, 3};
        m_array_io->save(materialsGroup, "texture_handles", numMat, chunkMat, textureHandles);
        m_array_io->save(materialsGroup, "rgb_color", dimensionsRgb, chunkRgb, rgb8Color);
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

    if (group.exist("materials"))
    {
        HighFive::Group materialsGroup = group.getGroup("materials");

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
                        nextMat.m_color = boost::optional<lvr2::Rgb8Color>(
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
    if (!hdf5util::exist(m_file_access->m_hdf5_file, m_mesh_name))
    {
        return boost::none;
    }
    HighFive::Group group = hdf5util::getGroup(m_file_access->m_hdf5_file, m_mesh_name, false);

    if (!isMesh(group))
    {
        std::cout << "[Hdf5IO - MeshIO] WARNING: flags of " << group.getId() << " are not correct."
                  << std::endl;
        return boost::none;
    }

    if (group.exist("channels"))
    {
        HighFive::Group channelsGroup = group.getGroup("channels");
        std::unique_ptr<HighFive::DataSet> dataset = std::make_unique<HighFive::DataSet>(
                channelsGroup.getDataSet("vertices"));
        std::vector<size_t> dim = dataset->getSpace().getDimensions();
        FloatChannel channel(dim[0], dim[1]);
        dataset->read(channel.dataPtr().get());
        return channel;
    }

    // If all fails return none
    return boost::none;
}

template <typename Derived>
IndexChannelOptional MeshIO<Derived>::getIndices()
{
    if (!hdf5util::exist(m_file_access->m_hdf5_file, m_mesh_name))
    {
        return boost::none;
    }
    HighFive::Group group = hdf5util::getGroup(m_file_access->m_hdf5_file, m_mesh_name, false);

    if (!isMesh(group))
    {
        std::cout << "[Hdf5IO - MeshIO] WARNING: flags of " << group.getId() << " are not correct."
                  << std::endl;
        return boost::none;
    }

    if (group.exist("channels"))
    {
        HighFive::Group channelsGroup = group.getGroup("channels");
        std::unique_ptr<HighFive::DataSet> dataset = std::make_unique<HighFive::DataSet>(
                channelsGroup.getDataSet("face_indices"));
        std::vector<size_t> dim = dataset->getSpace().getDimensions();
        IndexChannel channel(dim[0], dim[1]);
        dataset->read(channel.dataPtr().get());
        return channel;
    }

    // If all fails return none
    return boost::none;
}

template <typename Derived>
bool MeshIO<Derived>::addVertices(const FloatChannel& channel)
{
    HighFive::Group group = hdf5util::getGroup(m_file_access->m_hdf5_file, m_mesh_name, true);
    if (!hdf5util::exist(m_file_access->m_hdf5_file, m_mesh_name)) {
        return false;
    }

    std::string id(MeshIO<Derived>::ID);
    std::string obj(MeshIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    if (!group.exist("channels"))
    {
        group.createGroup("channels");
    }
    HighFive::Group channelsGroup = group.getGroup("channels");
    m_vchannel_io->save(channelsGroup, "vertices", VariantChannel<float>(channel));
    return true;
}

template <typename Derived>
bool MeshIO<Derived>::addIndices(const IndexChannel& channel)
{
    HighFive::Group group = hdf5util::getGroup(m_file_access->m_hdf5_file, m_mesh_name, true);
    if (!hdf5util::exist(m_file_access->m_hdf5_file, m_mesh_name)) {
        return false;
    }

    std::string id(MeshIO<Derived>::ID);
    std::string obj(MeshIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    if (!group.exist("channels"))
    {
        group.createGroup("channels");
    }
    HighFive::Group channelsGroup = group.getGroup("channels");
    m_vchannel_io->save(channelsGroup, "face_indices", VariantChannel<unsigned int>(channel));
    return true;
}

template<typename Derived>
template <typename T>
bool MeshIO<Derived>::getChannel(const std::string group, const std::string name, boost::optional<AttributeChannel<T>>& channel)
{
    if(m_file_access->m_hdf5_file && m_file_access->m_hdf5_file->isValid())
    {
        HighFive::Group meshGroup = hdf5util::getGroup(m_file_access->m_hdf5_file, m_mesh_name, false);

        // TODO check group for vertex / face attribute and set flag in hdf5 channel
        HighFive::Group g = meshGroup.getGroup("channels");

        if(g.exist(name))
        {
            HighFive::DataSet dataset = g.getDataSet(name);
            std::vector<size_t> dim = dataset.getSpace().getDimensions();

            size_t elementCount = 1;
            for (auto e : dim)
            elementCount *= e;

            if(elementCount)
            {
                channel = Channel<T>(dim[0], dim[1]);
                dataset.read(channel->dataPtr().get());
            }
        }
    }
    else
    {
        throw std::runtime_error("[Hdf5 - ChannelIO]: Hdf5 file not open.");
    }
    return true;
}

template<typename Derived>
template <typename T>
bool MeshIO<Derived>::addChannel(const std::string group, const std::string name, const AttributeChannel<T>& channel)
{
    if(m_file_access->m_hdf5_file && m_file_access->m_hdf5_file->isValid())
    {
        HighFive::DataSpace dataSpace({channel.numElements(), channel.width()});
        HighFive::DataSetCreateProps properties;

        if(m_file_access->m_chunkSize)
        {
            properties.add(HighFive::Chunking({channel.numElements(), channel.width()}));
        }
        if(m_file_access->m_compress)
        {
            //properties.add(HighFive::Shuffle());
            properties.add(HighFive::Deflate(9));
        }

        HighFive::Group meshGroup = hdf5util::getGroup(m_file_access->m_hdf5_file, m_mesh_name, true);
        if (!meshGroup.exist("channels"))
        {
            meshGroup.createGroup("channels");
        }

        // TODO check group for vertex / face attribute and set flag in hdf5 channel
        HighFive::Group g = meshGroup.getGroup("channels");

        std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<T>(
                g, name, dataSpace, properties);

        const T* ptr = channel.dataPtr().get();
        dataset->write(ptr);
        m_file_access->m_hdf5_file->flush();
        std::cout << timestamp << " Added attribute \"" << name << "\" to group \"" << group
                  << "\" to the given HDF5 file!" << std::endl;
    } else {
        throw std::runtime_error("[Hdf5IO - ChannelIO]: Hdf5 file not open.");
    }
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
