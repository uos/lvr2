
namespace lvr2 {

namespace hdf5features {

template<typename Derived>
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void MatrixIO<Derived>::save(std::string groupName,
    std::string datasetName,
    const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat)
{
    HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, groupName);
    save(g, datasetName, mat);
}

template<typename Derived>
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void MatrixIO<Derived>::save(HighFive::Group& group,
    std::string datasetName,
    const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat)
{
    if(m_file_access->m_hdf5_file && m_file_access->m_hdf5_file->isValid())
    {
        std::vector<hsize_t> chunkSizes = {_Rows, _Cols};
        std::vector<size_t > dims = {_Rows, _Cols};
        HighFive::DataSpace dataSpace(dims);
        HighFive::DataSetCreateProps properties;

        if(m_file_access->m_chunkSize)
        {
            for(size_t i = 0; i < chunkSizes.size(); i++)
            {
                if(chunkSizes[i] > dims[i])
                {
                    chunkSizes[i] = dims[i];
                }
            }
            properties.add(HighFive::Chunking(chunkSizes));
        }
        if(m_file_access->m_compress)
        {
            //properties.add(HighFive::Shuffle());
            properties.add(HighFive::Deflate(9));
        }

        std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<_Scalar>(
            group, datasetName, dataSpace, properties
        );

        const _Scalar* ptr = mat.data();
        dataset->write(ptr);
        m_file_access->m_hdf5_file->flush();

    } else {
        throw std::runtime_error("[Hdf5IO - ChannelIO]: Hdf5 file not open.");
    }
}


template<typename Derived>
template<typename MatrixT>
boost::optional<MatrixT> MatrixIO<Derived>::load(std::string groupName,
    std::string datasetName)
{
    boost::optional<MatrixT> ret;

    if(hdf5util::exist(m_file_access->m_hdf5_file, groupName))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, groupName, false);
        ret = load<MatrixT>(g, datasetName);
    }

    return ret;
}

template<typename Derived>
template<typename MatrixT>
boost::optional<MatrixT> MatrixIO<Derived>::load(HighFive::Group& group,
    std::string datasetName)
{
    boost::optional<MatrixT> ret;

    if(m_file_access->m_hdf5_file && m_file_access->m_hdf5_file->isValid())
    {
        if(group.exist(datasetName))
        {
            HighFive::DataSet dataset = group.getDataSet(datasetName);
            std::vector<size_t> dim = dataset.getSpace().getDimensions();

            size_t elementCount = 1;
            for (auto e : dim)
                elementCount *= e;

            MatrixT mat;
            dataset.read(mat.data());
            ret = mat;
        }
    } else {
        throw std::runtime_error("[Hdf5 - MatrixIO]: Hdf5 file not open.");
    }

    return ret;
}

template<typename Derived>
template<typename MatrixT>
boost::optional<MatrixT> MatrixIO<Derived>::loadMatrix(std::string groupName,
    std::string datasetName)
{
    return load<MatrixT>(groupName, datasetName);
}



} // namespace hdf5features

} // namespace lvr2