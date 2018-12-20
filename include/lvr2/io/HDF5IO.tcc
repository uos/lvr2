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

} // namespace lvr2
