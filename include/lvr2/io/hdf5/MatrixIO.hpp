#pragma once

#ifndef LVR2_IO_HDF5_MATRIXIO_HPP
#define LVR2_IO_HDF5_MATRIXIO_HPP

#include <Eigen/Dense>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

namespace lvr2 {

namespace hdf5features {

template<typename Derived>
class MatrixIO {
public:

    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    void save(std::string groupName,
        std::string datasetName,
        const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat
    );

    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    void save(HighFive::Group& group,
        std::string datasetName,
        const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& mat
    );

    template<typename MatrixT>
    boost::optional<MatrixT> load(HighFive::Group& group,
        std::string datasetName);

    template<typename MatrixT>
    boost::optional<MatrixT> load(std::string groupName,
        std::string datasetName);

    template<typename MatrixT>
    boost::optional<MatrixT> loadMatrix(std::string groupName,
        std::string datasetName);

protected:
    Derived* m_file_access = static_cast<Derived*>(this);

};

} // namespace hdf5features

} // namespace lvr2

#include "MatrixIO.tcc"

#endif // LVR2_IO_HDF5_MATRIXIO_HPP