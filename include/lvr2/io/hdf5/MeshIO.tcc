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

    if (!group.exist("materials"))
    {
        group.createGroup("materials");
    }
    HighFive::Group materialsGroup = group.getGroup("materials");
    if(!buffer->getMaterials().empty())
    {
        size_t numMaterials = buffer->getMaterials().size();
        boost::shared_array<int> textureHandles(new int[numMaterials]);

        boost::shared_array<int16_t> rgb8Color(new int16_t[numMaterials * 3]);
        lvr2::Material material;
        for (int i = 0; i < numMaterials; i++)
        {
            material = buffer->getMaterials().at(i);

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

} // namespace hdf5features

} // namespace lvr2
