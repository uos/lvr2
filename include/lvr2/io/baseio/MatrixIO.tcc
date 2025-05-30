#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2 
{

namespace baseio
{

template<typename BaseIO>
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void MatrixIO<BaseIO>::save(std::string groupName,
    std::string datasetName,
    const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat) const
{
    std::vector<size_t > dims = {_Rows, _Cols};
    const _Scalar* ptr = mat.data();

    Eigen::Map<Eigen::Matrix<_Scalar, _Rows, _Cols>> map(const_cast<_Scalar*>(ptr));

    Eigen::Matrix<double, _Rows, _Cols> dmat;
    dmat = map.template cast<double>();
    boost::shared_array<double> d_ptr(dmat.data());

    m_BaseIO->m_kernel->saveDoubleArray(groupName, datasetName, dims, d_ptr);
}

template<typename BaseIO>
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void MatrixIO<BaseIO>::saveMatrix(std::string groupName,
    std::string datasetName,
    const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat) const
{
    save(groupName, datasetName, mat);   
}

template<typename BaseIO>
template<typename MatrixT>
MatrixT MatrixIO<BaseIO>::loadMatrix(std::string groupName,
    std::string datasetName)
{
    std::vector<size_t> dims = {MatrixT::RowsAtCompileTime, MatrixT::ColsAtCompileTime};
  
    // Currently no ideo how to do it in a generalized way. Load 
    // matrix cofficients as double values and cast them back
    // into the desired type
    boost::shared_array<double> 
        arr = m_BaseIO->m_kernel->template loadDoubleArray(groupName, datasetName, dims);

    Eigen::Map<Eigen::Matrix<double, MatrixT::RowsAtCompileTime, MatrixT::ColsAtCompileTime>> map(arr.get());

    MatrixT mat(dims[0], dims[1]);
    mat = map.template cast<typename MatrixT::Scalar>();
    
    return mat;
}

} // namespace baseio

} // namespace lvr2