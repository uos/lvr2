
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
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
boost::optional<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> MatrixIO<FeatureBase>::load(std::string groupName,
    std::string datasetName)
{
    boost::optional<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> ret;
    std::vector<size_t> dims;
    boost::shared_array<_Scalar> arr = m_featureBase->m_kernel->template loadArray(groupName, datasetName, dims);

    if(dims.size() != 2)
    {
        std::cout << timestamp << "MatrixIO: Warning: Loaded array is not twodimensional: " 
                  << dims.size();
    }

    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> mat(dims[0], dims[1]);
    mat.data = arr.get();
    ret = mat;

    return ret;
}



} // namespace lvr2