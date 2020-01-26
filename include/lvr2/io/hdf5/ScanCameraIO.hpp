#pragma once

#ifndef LVR2_IO_HDF5_SCANCAMERAIO_HPP
#define LVR2_IO_HDF5_SCANCAMERAIO_HPP

#include "ScanImageIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
class ScanCameraIO
{
  public:
    void save(uint scanPos, uint camNr, const ScanCameraPtr& buffer);
    void save(HighFive::Group& group, uint camNr, const ScanCameraPtr& buffer);
    void save(HighFive::Group& group, const ScanCameraPtr& buffer);

    ScanCameraPtr load(uint scanPos, uint camNr);
    ScanCameraPtr load(HighFive::Group& group, uint camNr);
    ScanCameraPtr load(HighFive::Group& group);

  protected:
    bool isScanCamera(HighFive::Group& group);

    Derived* m_file_access = static_cast<Derived*>(this);

    // dependencies
    ScanImageIO<Derived>* m_scanImageIO = static_cast<ScanImageIO<Derived>*>(m_file_access);

    static constexpr const char* ID = "ScanCameraIO";
    static constexpr const char* OBJID = "ScanCamera";
};

} // namespace hdf5features

/**
 *
 * @brief Hdf5Construct Specialization for hdf5features::ScanCameraIO
 * - Constructs dependencies (ArrayIO, MatrixIO)
 * - Sets type variable
 *
 */
template <typename Derived>
struct Hdf5Construct<hdf5features::ScanCameraIO, Derived>
{

    // DEPS
    using deps = typename Hdf5Construct<hdf5features::ScanImageIO, Derived>::type;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<hdf5features::ScanCameraIO>::type;
};

} // namespace lvr2

#include "ScanCameraIO.tcc"

#endif // LVR2_IO_HDF5_SCANCAMERAIO_HPP
