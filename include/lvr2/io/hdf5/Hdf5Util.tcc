namespace lvr2
{

namespace hdf5util
{

template<typename T>
void addAtomic(HighFive::Group& g,
    const std::string datasetName,
    const T data)
{
    std::vector<size_t> dim = {1};
    HighFive::DataSpace dataSpace(dim);
    HighFive::DataSetCreateProps properties;
    auto dataset = createDataset<T>(g, datasetName, dataSpace, properties);
    dataset->write_raw(&data);
}

template<typename T>
void addArray(HighFive::Group& g,
    const std::string datasetName,
    std::vector<size_t>& dim,
    boost::shared_array<T>& data)
{
    HighFive::DataSpace dataSpace(dim);
    HighFive::DataSetCreateProps properties;

    auto dataset = createDataset<T>(g, datasetName, dataSpace, properties);
    if(dataset)
    {
        const T* ptr = data.get();
        dataset->write_raw(ptr);
    }
}

template<typename T>
void addArray(
    HighFive::Group& g, 
    const std::string datasetName, 
    const size_t& length, 
    boost::shared_array<T>& data)
{
    std::vector<size_t> dim = {length};
    addArray(g, datasetName, dim, data);
}

template<typename T>
void addVector(HighFive::Group& g,
    const std::string datasetName,
    const std::vector<T>& data)
{
    std::vector<size_t> dim = {data.size()};
    HighFive::DataSpace dataSpace(dim);
    HighFive::DataSetCreateProps properties;

    auto dataset = createDataset<T>(g, datasetName, dataSpace, properties);

    if(dataset)
    {
        const T* ptr = data.data();
        dataset->write_raw(ptr);
    }
}

template<typename T>
boost::optional<T> getAtomic(
    const HighFive::Group& g,
    const std::string datasetName)
{
    boost::optional<T> ret;

    if(g.isValid())
    {
        if(g.exist(datasetName))
        {
            HighFive::DataSet dataset = g.getDataSet(datasetName);
            std::vector<size_t> dims = dataset.getSpace().getDimensions();
            
            if(dims.size() == 1 && dims[0] == 1)
            {
                T data;
                dataset.read(data);
                ret = data;
            } else {
                std::cout << "[Hdf5Util - getAtomic]: " << datasetName << ", size: " << dims.size() << ": " << dims[0] << std::endl;
                throw std::runtime_error("[Hdf5Util - getAtomic]: try to load dataset of size > 1 as atomic.");
            }
        }
    } else {
        throw std::runtime_error("[Hdf5Util - getAtomic]: Hdf5 file not open.");
    }

    return ret;
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
boost::shared_array<T> getArray(
    const HighFive::Group& g, 
    const std::string& datasetName,
    size_t& dim)
{
    boost::shared_array<T> ret;

    if (g.exist(datasetName))
    {
        HighFive::DataSet dataset = g.getDataSet(datasetName);
        std::vector<size_t> dims = dataset.getSpace().getDimensions();

        if(dims.size() > 1)
        {
            return ret;
        }

        dim = dims[0];

        if(dim)
        {
            ret = boost::shared_array<T>(new T[dim]);

            dataset.read(ret.get());
        }
    }

    return ret;
}


template<typename T>
boost::optional<std::vector<T> > getVector(
    const HighFive::Group& g, 
    const std::string& datasetName)
{
    boost::optional<std::vector<T> > ret;

    if(g.isValid())
    {
        if(g.exist(datasetName))
        {
            HighFive::DataSet dataset = g.getDataSet(datasetName);
            std::vector<size_t> dim = dataset.getSpace().getDimensions();

            if(dim.size() > 1 && dim[2] > 1)
            {
                // is not a vector
                throw std::runtime_error("[Hdf5Util - getVector]: second dimension too big for a vector (>1).");
            }

            std::vector<T> data(dim[0]);
            dataset.read(&data[0]);
            ret = data;
        }
    } else {
        throw std::runtime_error("[Hdf5Util - getVector]: Hdf5 file not open.");
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

        std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<_Scalar>(
            group, datasetName, dataSpace, properties
        );

        if(dataset)
        {
            const _Scalar* ptr = mat.data();
            dataset->write_raw(ptr);
        }
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

    if constexpr (H5AllowedTypes::contains<T>())
    {
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
            else
            {
                // same datatype but different size -> resize
                // check dimensionality
                bool same_dims = true;
                if (dims_old.size() != dims_new.size())
                {
                    same_dims = false;
                } else {
                    // same sized: check entries
                    for(size_t i=0; i<dims_old.size(); i++)
                    {
                        if(dims_old[i] != dims_new[i])
                        {
                            same_dims = false;
                            break;
                        }
                    }
                }
                // same datatype but different size -> resize

                if(!same_dims)
                {
                    std::cout << "[Hdf5Util - createDataset] WARNING: size has changed. resizing dataset "
                        << datasetName << " from size " 
                        << dims_old[0] << "x" << dims_old[1] << " to " 
                        << dims_new[0] << "x" << dims_new[1] << std::endl;

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
        }
        else
        {
            dataset = std::make_unique<HighFive::DataSet>(
                g.createDataSet<T>(datasetName, dataSpace, properties));
        }
    } else {
        std::cout << "[Hdf5Util - createDataset] WARNING: could not create dataset ' << " << datasetName << "'. Data Type not allowed by H5" << std::endl;
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
void setAttribute(
    HighFive::DataSet& d,
    const std::string& attr_name, 
    T& data)
{
    bool use_existing_attribute = false;
    bool overwrite = false;

    if (d.hasAttribute(attr_name))
    {
        // check if attribute is the same
        HighFive::Attribute attr = d.getAttribute(attr_name);
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
        d.createAttribute<T>(attr_name, data);
    }
    else if (overwrite)
    {
        d.getAttribute(attr_name).write<T>(data);
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

template <typename T>
bool checkAttribute(HighFive::DataSet& d, const std::string& attr_name, T& data)
{
    // check if attribute exists
    if (!d.hasAttribute(attr_name))
    {
        return false;
    }

    // check if attribute type is the same
    HighFive::Attribute attr = d.getAttribute(attr_name);
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

template <typename T>
boost::optional<T> getAttribute(const HighFive::Group& g, const std::string& attr_name)
{
    boost::optional<T> ret;

    if(g.hasAttribute(attr_name))
    {
        T data;
        g.getAttribute(attr_name).read(data);
        ret = data;
    }

    return ret;
}

template <typename T>
boost::optional<T> getAttribute(const HighFive::DataSet& d, const std::string& attr_name)
{
    boost::optional<T> ret;

    if(d.hasAttribute(attr_name))
    {
        T data;
        d.getAttribute(attr_name).read(data);
        ret = data;
    }

    return ret;
}

} // namespace hdf5util

} // namespace lvr2