
namespace lvr2 
{

template<typename FeatureBase>
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void MatrixIO<FeatureBase>::save(std::string groupName,
    std::string datasetName,
    const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat)
{
    std::vector<size_t > dims = {_Rows, _Cols};
    const _Scalar* ptr = mat.data();
    m_featureBase->m_kernel->saveArray(groupName, datasetName, data, dims);
}

template<typename FeatureBase>
template<typename MatrixT>
boost::optional<MatrixT> MatrixIO<FeatureBase>::load(std::string groupName,
    std::string datasetName)
{
    boost::optional<MatrixT> ret;
    std::vector<size_t> dims;
    boost::shared_array<T> arr = m_featureBase->m_kernel->loadArray(groupName, datasetName, dims);

    if(dims.size() != 2)
    {
        std::cout << timestamp << "MatrixIO: Warning: Loaded array is not twodimensional: " 
                  << dims.size();
    }

    MatrixT(dims[0], dims[1]);
    
    /// UFFFF.....
    MatrixT mat;
    mat.data = arr.get();
    ret = mat;

    return ret;
}

template<typename FeatureBase>
template<typename MatrixT>
boost::optional<MatrixT> MatrixIO<FeatureBase>::load(HighFive::Group& group,
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

template<typename FeatureBase>
template<typename MatrixT>
boost::optional<MatrixT> MatrixIO<FeatureBase>::loadMatrix(std::string groupName,
    std::string datasetName)
{
    return load<MatrixT>(groupName, datasetName);
}


} // namespace lvr2