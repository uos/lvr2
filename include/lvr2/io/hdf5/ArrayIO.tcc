namespace lvr2 {

namespace hdf5features {

template<typename Derived>
template<typename T>
boost::shared_array<T> ArrayIO<Derived>::loadArray(
    std::string groupName,
    std::string datasetName,
    size_t& size)
{
    return load<T>(groupName, datasetName, size);
}

template<typename Derived>
template<typename T>
boost::shared_array<T> ArrayIO<Derived>::load(
    std::string groupName,
    std::string datasetName,
    size_t& size)
{
    boost::shared_array<T> ret;

    HighFive::Group g = hdf5util::getGroup(
        m_file_access->m_hdf5_file,
        groupName,
        false
    );

    std::vector<size_t> dim;
    ret = load<T>(g, datasetName, dim);

    size = 1;
    for (auto cur : dim)
        size *= cur;

    return ret;
}

template<typename Derived>
template<typename T>
boost::shared_array<T> ArrayIO<Derived>::load(
    std::string groupName,
    std::string datasetName,
    std::vector<size_t>& dim)
{
    HighFive::Group g = hdf5util::getGroup(
        m_file_access->m_hdf5_file,
        groupName,
        false
    );

    return load<T>(g, datasetName, dim);
}

template<typename Derived>
template<typename T>
boost::shared_array<T> ArrayIO<Derived>::load(
    HighFive::Group& g,
    std::string datasetName,
    std::vector<size_t>& dim)
{
    boost::shared_array<T> ret;
    
    if(m_file_access->m_hdf5_file && m_file_access->m_hdf5_file->isValid())
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
    } else {
        throw std::runtime_error("[Hdf5 - ArrayIO]: Hdf5 file not open.");
    }

    return ret;
}

template<typename Derived>
template<typename T>
void ArrayIO<Derived>::save(
    std::string groupName,
    std::string datasetName,
    size_t size,
    boost::shared_array<T> data)
{
    std::vector<size_t> dim = {size, 1};
    std::vector<hsize_t> chunks {m_file_access->m_chunkSize, 1};
    HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, groupName);
    save(g, datasetName, dim, chunks, data);
}

template<typename Derived>
template<typename T>
void ArrayIO<Derived>::save(
        std::string groupName,
        std::string datasetName,
        std::vector<size_t>& dimensions,
        boost::shared_array<T> data)
{
    HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, groupName);

    // Compute chunk size vector, i.e., set the chunk size in
    // each dimension to default size. Add float array will
    // trim this values if chunkSize > dim.
    std::vector<hsize_t> chunks;
    for(auto i: dimensions)
    {
            chunks.push_back(i);
    }
    save(g, datasetName, dimensions, chunks, data);
}

template<typename Derived>
template<typename T>
void ArrayIO<Derived>::save(
        std::string groupName,
        std::string datasetName,
        std::vector<size_t>& dimensions,
        std::vector<hsize_t>& chunkSize,
        boost::shared_array<T> data)
{
    HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, groupName);
    save(g, datasetName, dimensions, chunkSize, data);
}

template<typename Derived>
template<typename T>
void ArrayIO<Derived>::save(HighFive::Group& g,
        std::string datasetName,
        std::vector<size_t>& dim,
        std::vector<hsize_t>& chunkSizes,
        boost::shared_array<T>& data)
{
    if(m_file_access->m_hdf5_file && m_file_access->m_hdf5_file->isValid())
    {

        HighFive::DataSpace dataSpace(dim);
        HighFive::DataSetCreateProps properties;

        if(m_file_access->m_chunkSize)
        {
            // We have to check explicitly if chunk size
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
        if(m_file_access->m_compress)
        {
            //properties.add(HighFive::Shuffle());
            properties.add(HighFive::Deflate(9));
        }
        
        std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<T>(
            g, datasetName, dataSpace, properties
        );

        const T* ptr = data.get();
        dataset->write(ptr);
        m_file_access->m_hdf5_file->flush();
    } else {
        throw std::runtime_error("[Hdf5 - ArrayIO]: Hdf5 file not open.");
    }
}

} // namespace hdf5_features

} // namespace lvr2