#pragma once

#ifndef LVR2_IO_HDF5_MATRIXIO_HPP
#define LVR2_IO_HDF5_MATRIXIO_HPP

#include <Eigen/Dense>

namespace lvr2 {

template<typename FeatureBase>
class MatrixIO {
public:

    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    void saveMatrix(std::string groupName,
        std::string datasetName,
        const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat
    );

    template<typename MatrixT>
    MatrixT loadMatrix(std::string groupName,
        std::string datasetName);

protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

};

} // namespace lvr2

#include "MatrixIO.tcc"

#endif // LVR2_IO_HDF5_MATRIXIO_HPP