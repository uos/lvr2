namespace lvr2 {

template<typename T>
void HDF5IO::addArray(HighFive::Group& g,
        std::string datasetName,
        std::vector<size_t>& dim,
        std::vector<hsize_t>& chunkSizes,
        boost::shared_array<T>& data)
{
    HighFive::DataSpace dataSpace(dim);
    HighFive::DataSetCreateProps properties;

    if(m_chunkSize)
    {
        // We habe to check explicitly if chunk size
        // is < dimensionality to avoid errors from
        // the HDF5 lib
        for(size_t i = 0; i < chunkSizes.size(); i++)
        {
            if(chunkSizes[i] > dim[i])
            {

                chunkSizes[i] = dim[i];
            }
        }
        properties.add(HighFive::Chunking(chunkSizes));
    }
    if(m_compress)
    {
        //properties.add(HighFive::Shuffle());
        properties.add(HighFive::Deflate(9));
    }
    HighFive::DataSet dataset = g.createDataSet<T>(datasetName, dataSpace, properties);
    const T* ptr = data.get();
    dataset.write(ptr);
}

template<typename T>
void HDF5IO::addArray(
        std::string groupName,
        std::string datasetName,
        std::vector<size_t>& dimensions,
        std::vector<hsize_t>& chunkSize, boost::shared_array<T> data)
{
    HighFive::Group g = getGroup(groupName);
    addArray(g, datasetName, dimensions, chunkSize, data);
}

template<typename T>
void HDF5IO::addArray(
        std::string groupName, std::string datasetName,
        std::vector<size_t>& dimensions, boost::shared_array<T> data)
{
    HighFive::Group g = getGroup(groupName);

    // Compute chunk size vector, i.e., set the chunk size in
    // each dimension to default size. Add float array will
    // trim this values if chunkSize > dim.
    std::vector<hsize_t> chunks;
    for(auto i: dimensions)
    {
            chunks.push_back(i);
    }
    addArray(g, datasetName, dimensions, chunks, data);
}

template<typename T>
void HDF5IO::addArray(
        std::string group, std::string name,
        unsigned int size, boost::shared_array<T> data)
{
    if(m_hdf5_file)
    {
        std::vector<size_t> dim = {size, 1};
        std::vector<hsize_t> chunks {m_chunkSize, 1};
        HighFive::Group g = getGroup(group);
        addArray(g, name, dim, chunks, data);
    }
}

template<typename T>
boost::shared_array<T> HDF5IO::getArray(
        HighFive::Group& g, std::string datasetName,
        std::vector<size_t>& dim)
{
    boost::shared_array<T> ret;

    if(m_hdf5_file)
    {
        if (g.exist(datasetName))
        {
            HighFive::DataSet dataset = g.getDataSet(datasetName);
            dim = dataset.getSpace().getDimensions();

            size_t elementCount = 1;
            for (auto e : dim)
                elementCount *= e;

            if(elementCount)
            {
                ret = boost::shared_array<T>(new T[elementCount]);

                dataset.read(ret.get());
            }
        }
    }

    return ret;
}

template<typename T>
boost::shared_array<T> HDF5IO::getArray(
        std::string groupName, std::string datasetName,
        std::vector<size_t>& dim)
{
    boost::shared_array<T> ret;

    if(m_hdf5_file)
    {
        if (exist(groupName))
        {
            HighFive::Group g = getGroup(groupName, false);
            ret = getArray<T>(g, datasetName, dim);
        }
    }

    return ret;
}

template<typename T>
boost::shared_array<T> HDF5IO::getArray(
        std::string groupName, std::string datasetName,
        unsigned int& size)
{
    boost::shared_array<T> ret;

    if(m_hdf5_file)
    {
        if (exist(groupName))
        {
            HighFive::Group g = getGroup(groupName, false);
            std::vector<size_t> dim;
            ret = getArray<T>(g, datasetName, dim);

            size = 1;

            // if you use this function, you expect a one dimensional array
            // and therefore we calculate the toal amount of elements
            for (auto cur : dim)
                size *= cur;
        }
    }

    return ret;
}

template <typename T>
boost::shared_array<T> HDF5IO::reduceData(boost::shared_array<T> data, size_t dataCount, size_t dataWidth, unsigned int reductionFactor, size_t *reducedDataCount)
{
    *reducedDataCount = dataCount / reductionFactor + 1;
    boost::shared_array<T> reducedData = boost::shared_array<T>( new T[(*reducedDataCount) * dataWidth] );

    size_t reducedDataIdx = 0;
    for (size_t i = 0; i < dataCount; i++)
    {
        if (i % reductionFactor == 0)
        {
            std::copy(data.get() + i*dataWidth,
                    data.get() + (i+1)*dataWidth,
                    reducedData.get() + reducedDataIdx*dataWidth);

            reducedDataIdx++;
        }
    }

    return reducedData;
}

template <typename T>
bool HDF5IO::getChannel(const std::string group, const std::string name, boost::optional<AttributeChannel<T>>& channel){
    auto mesh_opt = getMeshGroup();
    if(!mesh_opt) return false;
    auto mesh = mesh_opt.get();
    if(!mesh.exist(group))
    {
        std::cout << timestamp << " Could not find mesh attribute group \"" << group << "\" in the given HDF5 file!"
        << std::endl;
        return false;
    }
    auto attribute_group = mesh.getGroup(group);
    if(!attribute_group.exist(name))
    {
        std::cout << timestamp << " Could not find mesh attribute \"" << name << "\" in group \"" << group
        << "\" in the given HDF5 file!" << std::endl;
        return false;
    }

    std::vector<size_t >dims;
    auto values = getArray<T>(attribute_group, name, dims);
    channel = AttributeChannel<T>(dims[0], dims[1], values);
    return true;
}

template <typename T>
bool HDF5IO::addChannel(const std::string group, const std::string name, const AttributeChannel<T>& channel)
{
    std::vector<size_t > dims = {channel.numElements(), channel.width()};
    addArray<T>(m_mesh_path + "/" + group , name, dims, channel.dataPtr());
    return true;
}

} // namespace lvr2
