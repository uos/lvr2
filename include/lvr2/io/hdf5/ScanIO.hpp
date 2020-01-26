#pragma once

#ifndef LVR2_IO_HDF5_SCANIO_HPP
#define LVR2_IO_HDF5_SCANIO_HPP

#include "ArrayIO.hpp"
#include "MatrixIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
class ScanIO
{
  public:
    void save(uint scanPos, uint scanNr, const ScanPtr& buffer);
    void save(HighFive::Group& group, uint scanNr, const ScanPtr& buffer);
    void save(HighFive::Group& group, const ScanPtr& buffer);

    ScanPtr load(uint scanPos, uint scanNr);
    ScanPtr load(HighFive::Group& group, uint scanNr);
    ScanPtr load(HighFive::Group& group);
    // ScanPtr loadScan(HighFive::Group& group, std::string name);

  protected:
    bool isScan(HighFive::Group& group);

    Derived* m_file_access = static_cast<Derived*>(this);

    // dependencies
    ArrayIO<Derived>* m_arrayIO = static_cast<ArrayIO<Derived>*>(m_file_access);
    MatrixIO<Derived>* m_matrixIO = static_cast<MatrixIO<Derived>*>(m_file_access);

    static constexpr const char* ID = "ScanIO";
    static constexpr const char* OBJID = "Scan";
};

} // namespace hdf5features

/**
 *
 * @brief Hdf5Construct Specialization for hdf5features::ScanIO
 * - Constructs dependencies (ArrayIO, MatrixIO)
 * - Sets type variable
 *
 */
template <typename Derived>
struct Hdf5Construct<hdf5features::ScanIO, Derived>
{

    // DEPS
    using dep1 = typename Hdf5Construct<hdf5features::ArrayIO, Derived>::type;
    using dep2 = typename Hdf5Construct<hdf5features::MatrixIO, Derived>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<hdf5features::ScanIO>::type;
};

} // namespace lvr2

#include "ScanIO.tcc"

#endif // LVR2_IO_HDF5_SCANIO_HPP
