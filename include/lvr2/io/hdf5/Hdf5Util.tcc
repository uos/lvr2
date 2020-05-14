namespace lvr2
{

namespace hdf5util
{
    
template<typename T>
void addArray(HighFive::Group& g,
    const std::string datasetName,
    std::vector<size_t>& dim,
    boost::shared_array<T>& data)
{
    HighFive::DataSpace dataSpace(dim);
    HighFive::DataSetCreateProps properties;

    // if(m_chunkSize)
    // {
    //     // We have to check explicitly if chunk size
    //     // is < dimensionality to avoid errors from
    //     // the HDF5 lib
    //     for(size_t i = 0; i < chunkSizes.size(); i++)
    //     {
    //         if(chunkSizes[i] > dim[i])
    //         {
    //             chunkSizes[i] = dim[i];
    //         }
    //     }
    //     properties.add(HighFive::Chunking(chunkSizes));
    // }
    // if(m_compress)
    // {
    //     //properties.add(HighFive::Shuffle());
    //     properties.add(HighFive::Deflate(9));
    // }
    HighFive::DataSet dataset = g.createDataSet<T>(datasetName, dataSpace, properties);
    const T* ptr = data.get();
    dataset.write(ptr);

    //std::cout << timestamp << " Wrote " << datasetName << " to HDF5 file." << std::endl;
}

template<typename T>
void addArray(
    HighFive::Group& g, 
    const std::string datasetName, 
    const size_t& length, 
    boost::shared_array<T>& data)
{
    std::vector<size_t> dim = {length, 1};
    addArray(g, datasetName, dim, data);
}

template<typename T>
boost::shared_array<T> getArray(
    const HighFive::Group& g, 
    const std::string& datasetName,
    std::vector<size_t>& dim)
{
    boost::shared_array<T> ret;

    if (g.exist(datasetName))
    {
        HighFive::DataSet dataset = g.getDataSet(datasetName);
        dim = dataset.getSpace().getDimensions();

        size_t elementCount = 1;
        for (auto e : dim)
            elementCount *= e;

        if (elementCount)
        {
            ret = boost::shared_array<T>(new T[elementCount]);

            dataset.read(ret.get());
        }
    }

    return ret;
}

template<typename T>
std::vector<size_t> getDimensions(
    const HighFive::Group& g, 
    const std::string& datasetName)
{
    if (g.exist(datasetName))
    {
        HighFive::DataSet dataset = g.getDataSet(datasetName);
        return dataset.getSpace().getDimensions();
    }

    return std::vector<size_t>(0);
}

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void addMatrix(HighFive::Group& group,
    std::string datasetName,
    const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat)
{
    if(group.isValid())
    {
        std::vector<hsize_t> chunkSizes = {_Rows, _Cols};
        std::vector<size_t > dims = {_Rows, _Cols};
        HighFive::DataSpace dataSpace(dims);
        HighFive::DataSetCreateProps properties;

        // if(m_file_access->m_chunkSize)
        // {
        //     for(size_t i = 0; i < chunkSizes.size(); i++)
        //     {
        //         if(chunkSizes[i] > dims[i])
        //         {
        //             chunkSizes[i] = dims[i];
        //         }
        //     }
        //     properties.add(HighFive::Chunking(chunkSizes));
        // }
        // if(m_file_access->m_compress)
        // {
        //     //properties.add(HighFive::Shuffle());
        //     properties.add(HighFive::Deflate(9));
        // }

        std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<_Scalar>(
            group, datasetName, dataSpace, properties
        );

        const _Scalar* ptr = mat.data();
        dataset->write(ptr);
        
    } 
    else 
    {
        throw std::runtime_error("[Hdf5IO - ChannelIO]: Hdf5 file not open.");
    }
}

template<typename MatrixT>
boost::optional<MatrixT> getMatrix(const HighFive::Group& g, const std::string& datasetName)
{
    boost::optional<MatrixT> ret;

    if(g.isValid())
    {
        if(g.exist(datasetName))
        {
            HighFive::DataSet dataset = g.getDataSet(datasetName);
            std::vector<size_t> dim = dataset.getSpace().getDimensions();

            size_t elementCount = 1;
            for (auto e : dim)
                elementCount *= e;

            MatrixT mat;
            dataset.read(mat.data());
            ret = mat;
        }
    } 
    else 
    {
        throw std::runtime_error("[Hdf5 - MatrixIO]: Hdf5 file not open.");
    }

    return ret;
}

template <typename T>
std::unique_ptr<HighFive::DataSet> createDataset(HighFive::Group& g,
                                                 std::string datasetName,
                                                 const HighFive::DataSpace& dataSpace,
                                                 const HighFive::DataSetCreateProps& properties)
{
    std::unique_ptr<HighFive::DataSet> dataset;

    if (g.exist(datasetName))
    {
        try
        {
            dataset = std::make_unique<HighFive::DataSet>(g.getDataSet(datasetName));
        }
        catch (HighFive::DataSetException& ex)
        {
            std::cout << "[Hdf5Util - createDataset] " << datasetName << " is not a dataset"
                      << std::endl;
        }

        // check existing dimensions
        const std::vector<size_t> dims_old = dataset->getSpace().getDimensions();
        const std::vector<size_t> dims_new = dataSpace.getDimensions();

        if (dataset->getDataType() != HighFive::AtomicType<T>())
        {
            // different datatype -> delete
            int result = H5Ldelete(g.getId(), datasetName.data(), H5P_DEFAULT);
            dataset = std::make_unique<HighFive::DataSet>(
                g.createDataSet<T>(datasetName, dataSpace, properties));
        }
        else if (dims_old[0] != dims_new[0] || dims_old[1] != dims_new[1])
        {
            // same datatype but different size -> resize

            std::cout << "[Hdf5Util - createDataset] WARNING: size has changed. resizing dataset "
                      << std::endl;

            //
            try
            {
                dataset->resize(dims_new);
            }
            catch (HighFive::DataSetException& ex)
            {
                std::cout << "[Hdf5Util - createDataset] WARNING: could not resize. Generating new "
                             "space..."
                          << std::endl;
                int result = H5Ldelete(g.getId(), datasetName.data(), H5P_DEFAULT);

                dataset = std::make_unique<HighFive::DataSet>(
                    g.createDataSet<T>(datasetName, dataSpace, properties));
            }
        }
    }
    else
    {
        dataset = std::make_unique<HighFive::DataSet>(
            g.createDataSet<T>(datasetName, dataSpace, properties));
    }

    return std::move(dataset);
}

template <typename T>
void setAttribute(HighFive::Group& g, const std::string& attr_name, T& data)
{
    bool use_existing_attribute = false;
    bool overwrite = false;

    if (g.hasAttribute(attr_name))
    {
        // check if attribute is the same
        HighFive::Attribute attr = g.getAttribute(attr_name);
        if (attr.getDataType() == HighFive::AtomicType<T>())
        {
            T value;
            attr.read(value);

            use_existing_attribute = true;
            if (value != data)
            {
                overwrite = true;
            }
        }
    }

    if (!use_existing_attribute)
    {
        g.createAttribute<T>(attr_name, data);
    }
    else if (overwrite)
    {
        g.getAttribute(attr_name).write<T>(data);
    }
}

template <typename T>
bool checkAttribute(HighFive::Group& g, const std::string& attr_name, T& data)
{
    // check if attribute exists
    if (!g.hasAttribute(attr_name))
    {
        return false;
    }

    // check if attribute type is the same
    HighFive::Attribute attr = g.getAttribute(attr_name);
    if (attr.getDataType() != HighFive::AtomicType<T>())
    {
        return false;
    }

    // check if attribute value is the same
    T value;
    attr.read(value);
    if (value != data)
    {
        return false;
    }

    return true;
}

} // namespace hdf5util

} // namespace lvr2