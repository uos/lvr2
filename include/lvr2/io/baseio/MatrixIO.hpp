
#ifndef MATRIXIO
#define MATRIXIO

#include <Eigen/Dense>
#include "lvr2/io/baseio/BaseIO.hpp"
namespace lvr2 
{

namespace baseio
{

template<typename BaseIO>
class MatrixIO {
public:

    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    void save(
        std::string groupName,
        std::string datasetName,
        const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat
    ) const;

    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    void saveMatrix(
        std::string groupName,
        std::string datasetName,
        const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat
    ) const;

    template<typename MatrixT>
    MatrixT loadMatrix(std::string groupName,
        std::string datasetName);

protected:
    BaseIO* m_BaseIO = static_cast<BaseIO*>(this);

};

} // namespace baseio

} // namespace lvr2

#include "MatrixIO.tcc"

#endif // MATRIXIO
