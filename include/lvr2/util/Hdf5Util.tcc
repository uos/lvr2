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

template <typename T, typename HT>
void setAttribute(
    HT& g, 
    const std::string& attr_name, 
    const T& data)
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
        g.template createAttribute<T>(attr_name, data);
    }
    else if (overwrite)
    {
        g.getAttribute(attr_name).template write<T>(data);
    }
}

template<typename T, typename HT>
void setAttributeVector(
    HT& g,
    const std::string& attr_name,
    const std::vector<T>& vec)
{
    std::vector<size_t> dims = {vec.size()};
    HighFive::DataSpace ds(dims);
    std::unique_ptr<HighFive::Attribute> h5att;

    if(g.hasAttribute(attr_name))
    {
        h5att = std::make_unique<HighFive::Attribute>(
            g.getAttribute(attr_name)
        );

        bool reset = false;

        if(h5att->getDataType() == HighFive::AtomicType<T>())
        {
            // Type is the same
            std::vector<size_t> dims_old = h5att->getSpace().getDimensions();

            if(dims_old.size() == dims.size())
            {
                for(size_t i=0; i<dims.size(); i++)
                {
                    if(dims_old[i] != dims[i])
                    {
                        reset = true;
                        break;
                    }
                }
            } else {
                reset = true;
            }
        } else {
            reset = true;
        }

        if(reset)
        {
            h5att.reset();
            g.deleteAttribute(attr_name);
        }
    }

    if(!h5att)
    {
        h5att = std::make_unique<HighFive::Attribute>(
            g.template createAttribute<T>(attr_name, ds)
        );
    }
    h5att->write(vec);
}

template<typename T, typename HT>
void setAttributeArray(
    HT& g,
    const std::string& attr_name,
    boost::shared_array<T> data,
    size_t size)
{
    std::vector<size_t> dims = {size};
    HighFive::DataSpace ds(dims);
    std::unique_ptr<HighFive::Attribute> h5att;

    if(g.hasAttribute(attr_name))
    {
        h5att = std::make_unique<HighFive::Attribute>(
            g.getAttribute(attr_name)
        );

        bool reset = false;

        if(h5att->getDataType() == HighFive::AtomicType<T>())
        {
            // Type is the same
            std::vector<size_t> dims_old = h5att->getSpace().getDimensions();

            if(dims_old.size() == dims.size())
            {
                for(size_t i=0; i<dims.size(); i++)
                {
                    if(dims_old[i] != dims[i])
                    {
                        reset = true;
                        break;
                    }
                }
            } else {
                reset = true;
            }
        } else {
            reset = true;
        }

        if(reset)
        {
            h5att.reset();
            g.deleteAttribute(attr_name);
        }
    }

    if(!h5att)
    {
        h5att = std::make_unique<HighFive::Attribute>(
            g.template createAttribute<T>(attr_name, ds)
        );
    }

    h5att->write(data.get());
}

template<typename HT>
void setAttributeMatrix(
    HT& g, 
    const std::string& attr_name,
    const Eigen::MatrixXd& mat)
{
    std::vector<size_t> dims = {
        static_cast<size_t>(mat.cols()), 
        static_cast<size_t>(mat.rows())};

    HighFive::DataSpace ds(dims);

    std::unique_ptr<HighFive::Attribute> h5att;

    if(g.hasAttribute(attr_name))
    {
        h5att = std::make_unique<HighFive::Attribute>(
            g.getAttribute(attr_name)
        );

        bool reset = false;

        if(h5att->getDataType() == HighFive::AtomicType<double>())
        {
            // Type is the same
            std::vector<size_t> dims_old = h5att->getSpace().getDimensions();

            if(dims_old.size() == dims.size())
            {
                for(size_t i=0; i<dims.size(); i++)
                {
                    if(dims_old[i] != dims[i])
                    {
                        reset = true;
                        break;
                    }
                }
            } else {
                reset = true;
            }
        } else {
            reset = true;
        }

        if(reset)
        {
            h5att.reset();
            g.deleteAttribute(attr_name);
        }
    }

    if(!h5att)
    {
        h5att = std::make_unique<HighFive::Attribute>(
            g.template createAttribute<double>(attr_name, ds)
        );
    }

    h5att->write_raw(mat.data());
}

template <typename HT, typename T>
bool checkAttribute(HT& g, const std::string& attr_name, T& data)
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

template <typename T, typename HT>
boost::optional<T> getAttribute(const HT& g, const std::string& attr_name)
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

template<typename T, typename HT>
boost::optional<std::vector<T> > getAttributeVector(
    const HT& g,
    const std::string& attr_name)
{
    boost::optional<std::vector<T> > ret;

    if(g.hasAttribute(attr_name))
    {
        HighFive::Attribute h5attr = g.getAttribute(attr_name);
        if(h5attr.getDataType() != HighFive::AtomicType<T>())
        {
            return ret;
        }

        std::vector<size_t> dims = h5attr.getSpace().getDimensions();
        
        if(dims.size() != 1)
        {
            return ret;
        }

        std::vector<T> data(dims[0]);
        h5attr.read(data);
        ret = data;
    }

    return ret;
}

template<typename HT>
boost::optional<Eigen::MatrixXd> getAttributeMatrix(
    const HT& g,
    const std::string& attr_name)
{
    boost::optional<Eigen::MatrixXd> ret;

    if(g.hasAttribute(attr_name))
    {
        HighFive::Attribute h5attr = g.getAttribute(attr_name);
        if(h5attr.getDataType() != HighFive::AtomicType<double>())
        {
            return ret;
        }

        std::vector<size_t> dims = h5attr.getSpace().getDimensions();
        
        if(dims.size() != 2)
        {
            return ret;
        }

        Eigen::MatrixXd mat(dims[0], dims[1]);

        h5attr.read(mat.data());

        ret = mat;
    }

    return ret;
}

template<typename HT>
void setAttributeMeta(
    HT& g,
    YAML::Node meta,
    std::string prefix)
{
    for(auto it = meta.begin(); it != meta.end(); ++it) 
    {   
        std::string key = it->first.as<std::string>();
        YAML::Node value = it->second;

        // attributeName of hdf5
        std::string attributeName = key;
        
        // add prefix to key
        if(prefix != ""){ attributeName = prefix + "/" + attributeName; }

        if(value.Type() == YAML::NodeType::Scalar)
        {
            // Write Scalar
            // std::cout << attributeName << ": Scalar" << std::endl;

            // get scalar type
            long int lint;
            double dbl;
            bool bl;
            std::string str;

            if(YAML::convert<long int>::decode(value, lint))
            {
                setAttribute(g, attributeName, lint);
            } 
            else if(YAML::convert<double>::decode(value, dbl)) 
            {
                setAttribute(g, attributeName, dbl);
            } 
            else if(YAML::convert<bool>::decode(value, bl))
            {
                setAttribute(g, attributeName, bl);
            } 
            else if(YAML::convert<std::string>::decode(value, str))
            {
                setAttribute(g, attributeName, str);
            }
            else
            {
                std::cout << "ERROR: UNKNOWN TYPE of value " << value << std::endl;
            }

            // std::cout << attributeName << ": written." << std::endl;
        } 
        else if(value.Type() == YAML::NodeType::Sequence) 
        {

            // check if matrix
            if(value.begin()->IsSequence() )
            {
                // Double list
                // std::cout << "IS MATRIX" << std::endl;
                if(YAML::isMatrix(value))
                {
                    // list of lists is a matrix
                    // TODO: hard coded type double
                    Eigen::MatrixXd mat;
                    if(YAML::convert<Eigen::MatrixXd>::decode(value, mat))
                    {
                        setAttributeMatrix(g, attributeName, mat);
                    } else {
                        std::cout << "ERROR matrix" << std::endl;
                    }
                } else {
                    // error
                    std::cout << "ERROR cannot parse list of lists" << std::endl;
                }
            } else {
                // Simple list
                // std::cout << attributeName << ": Sequence" << std::endl;
                // check the type with all elements
                bool is_int = true;
                bool is_double = true;
                bool is_bool = true;
                bool is_string = true;

                size_t nelements = 0;

                for(auto seq_it = value.begin(); seq_it != value.end(); seq_it++)
                {
                    long int lint;
                    double dbl;
                    bool bl;
                    std::string str;

                    // std::cout << "Id: " << nelements << std::endl;

                    // std::cout << "Integer check" << std::endl;
                    if(!YAML::convert<long int>::decode(*seq_it, lint))
                    {
                        is_int = false;
                    }

                    // std::cout << "Double check" << std::endl;
                    if(!YAML::convert<double>::decode(*seq_it, dbl))
                    {
                        is_double = false;
                    }

                    // std::cout << "Bool check" << std::endl;
                    if(!YAML::convert<bool>::decode(*seq_it, bl))
                    {
                        is_bool = false;
                    }

                    if(!YAML::convert<std::string>::decode(*seq_it, str))
                    {
                        is_string = false;
                    }

                    nelements++;
                }
                if(is_int)
                {
                    std::vector<long int> data;
                    for(auto seq_it = value.begin(); seq_it != value.end(); seq_it++)
                    {
                        data.push_back(seq_it->as<long int>());
                    }
                    setAttributeVector(g, attributeName, data);
                }
                else if(is_double)
                {
                    std::vector<double> data;
                    for(auto seq_it = value.begin(); seq_it != value.end(); seq_it++)
                    {
                        data.push_back(seq_it->as<double>());
                    }
                    setAttributeVector(g, attributeName, data);
                }
                else if(is_bool)
                {
                    // Bool vector is special
                    // https://stackoverflow.com/questions/51352045/void-value-not-ignored-as-it-ought-to-be-on-non-void-function
                    // need workaround

                    // hdf5 stores bool arrays in uint8 anyway
                    // std::vector<uint8_t> data;
                    // for(auto seq_it = value.begin(); seq_it != value.end(); seq_it++)
                    // {
                    //     data.push_back(static_cast<uint8_t>(seq_it->as<bool>()));
                    // }
                    // hdf5util::setAttributeVector(g, attributeName, data);

                    boost::shared_array<bool> data(new bool[nelements]);
                    size_t i = 0;
                    for(auto seq_it = value.begin(); seq_it != value.end(); seq_it++, i++)
                    {
                        data[i] = seq_it->as<bool>();
                    }
                    setAttributeArray(g, attributeName, data, nelements);
                } 
                else if(is_string) 
                {
                    std::vector<std::string> data;
                    for(auto seq_it = value.begin(); seq_it != value.end(); seq_it++)
                    {
                        data.push_back(seq_it->as<std::string>());
                    }
                    setAttributeVector(g, attributeName, data);
                } 
                else 
                {
                    std::cout << "Tried to write YAML list of unknown typed elements: " << value << std::endl;
                }
            }

            // TODO: check list of strings
            

        } 
        else if(value.Type() == YAML::NodeType::Map) 
        {
            // recursion
            setAttributeMeta(g, value, attributeName);
            // // check if Map is known type
            // if(YAML::isMatrix(value))
            // {
            //     // std::cout << attributeName << ": Matrix" << std::endl;
            //     // std::cout << value << std::endl;
            //     Eigen::MatrixXd mat;
            //     if(YAML::convert<Eigen::MatrixXd>::decode(value, mat))
            //     {
            //         setAttributeMatrix(g, attributeName, mat);
            //     } else {
            //         std::cout << "ERROR matrix" << std::endl;
            //     }

            //     // std::cout << attributeName << ": written." << std::endl;
            // } else {
            //     // recursion
            //     setAttributeMeta(g, value, attributeName);
            // }
        } 
        else 
        {
            std::cout << attributeName << ": UNKNOWN -> Error" << std::endl;
            std::cout << value << std::endl;
        }
    }
}

template<typename HT>
YAML::Node getAttributeMeta(
    const HT& g)
{
    YAML::Node ret(YAML::NodeType::Map);

    for(std::string attributeName : g.listAttributeNames())
    {
        std::vector<YAML::Node> yamlNodes;
        std::vector<std::string> yamlNames = splitGroupNames(attributeName);

        // auto node_iter = ret;
        yamlNodes.push_back(ret);
        for(size_t i=0; i<yamlNames.size()-1; i++)
        {
            YAML::Node tmp = yamlNodes[i][yamlNames[i]];
            yamlNodes.push_back(tmp);
        }

        YAML::Node back = yamlNodes.back();

        HighFive::Attribute h5attr = g.getAttribute(attributeName);
        std::vector<size_t> dims = h5attr.getSpace().getDimensions();
        HighFive::DataType h5type = h5attr.getDataType();
        if(dims.size() == 0)
        {
            // Bool problems
            if(h5type == HighFive::AtomicType<bool>())
            {
                back[yamlNames.back()] = *getAttribute<bool>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<char>())
            {
                back[yamlNames.back()] = *getAttribute<char>(g, attributeName);
            } 
            else if(h5type == HighFive::AtomicType<unsigned char>())
            {
                back[yamlNames.back()] = *getAttribute<unsigned char>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<short>())
            {
                back[yamlNames.back()] = *getAttribute<short>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<unsigned short>())
            {   
                back[yamlNames.back()] = *getAttribute<unsigned short>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<int>())
            {   
                back[yamlNames.back()] = *getAttribute<int>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<unsigned int>())
            {   
                back[yamlNames.back()] = *getAttribute<unsigned int>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<long int>())
            {   
                back[yamlNames.back()] = *getAttribute<long int>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<unsigned long int>())
            {   
                back[yamlNames.back()] = *getAttribute<unsigned long int>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<float>())
            {   
                back[yamlNames.back()] = *getAttribute<float>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<double>())
            {   
                back[yamlNames.back()] = *getAttribute<double>(g, attributeName);
            }
            else if(h5type == HighFive::AtomicType<std::string>()) 
            {
                back[yamlNames.back()] = *getAttribute<std::string>(g, attributeName);
            } 
            else {
                // std::cout << h5type.string() << ": type not implemented. " << std::endl;
            }
        }
        else if(dims.size() == 1)
        {
            back[yamlNames.back()] = YAML::Load("[]");
            // Sequence
            if(h5type == HighFive::AtomicType<bool>())
            {
                std::vector<uint8_t> data = *getAttributeVector<uint8_t>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(static_cast<bool>(value));
                }
            }
            else if(h5type == HighFive::AtomicType<char>())
            {
                std::vector<char> data = *getAttributeVector<char>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            } 
            else if(h5type == HighFive::AtomicType<unsigned char>())
            {
                std::vector<unsigned char> data = *getAttributeVector<unsigned char>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<short>())
            {
                std::vector<short> data = *getAttributeVector<short>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<unsigned short>())
            {   
                std::vector<unsigned short> data = *getAttributeVector<unsigned short>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<int>())
            {   
                std::vector<int> data = *getAttributeVector<int>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<unsigned int>())
            {   
                std::vector<unsigned int> data = *getAttributeVector<unsigned int>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<long int>())
            {   
                std::vector<long int> data = *getAttributeVector<long int>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<unsigned long int>())
            {   
                std::vector<unsigned long int> data = *getAttributeVector<unsigned long int>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<float>())
            {   
                std::vector<float> data = *getAttributeVector<float>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<double>())
            {   
                std::vector<double> data = *getAttributeVector<double>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else if(h5type == HighFive::AtomicType<std::string>())
            {
                std::vector<std::string> data = *getAttributeVector<std::string>(g, attributeName);
                for(auto value : data)
                {
                    back[yamlNames.back()].push_back(value);
                }
            }
            else {
                std::cout << h5type.string() << ": type not supported for YAML lists. " << std::endl;
            }

        }
        else if(dims.size() == 2)
        {
            // Matrix
            Eigen::MatrixXd mat = *getAttributeMatrix(g, attributeName);
            back[yamlNames.back()] = mat;
        }

        ret = yamlNodes.front();
    }

    return ret;
}


} // namespace hdf5util

} // namespace lvr2