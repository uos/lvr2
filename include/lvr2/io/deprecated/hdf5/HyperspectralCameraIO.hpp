#pragma once

#ifndef LVR2_IO_HDF5_HYPERSPECTRALCAMERAIO_HPP
#define LVR2_IO_HDF5_HYPERSPECTRALCAMERAIO_HPP

#include "ArrayIO.hpp"
#include "MatrixIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
class HyperspectralCameraIO
{
  public:
    void save(HighFive::Group& group, const HyperspectralCameraPtr& buffer);

    HyperspectralCameraPtr load(uint scanPos);
    // HyperspectralCameraPtr load(HighFive::Group& group, uint camNr);
    HyperspectralCameraPtr load(HighFive::Group& group);
    HyperspectralCameraPtr loadHyperspectralCamera(uint scanPos);

  protected:
    bool isHyperspectralCamera(HighFive::Group& group);

    Derived* m_file_access = static_cast<Derived*>(this);

    // dependencies
    ArrayIO<Derived>* m_arrayIO = static_cast<ArrayIO<Derived>*>(m_file_access);
    MatrixIO<Derived>* m_matrixIO = static_cast<MatrixIO<Derived>*>(m_file_access);

    static constexpr const char* ID = "HyperspectralCameraIO";
    static constexpr const char* OBJID = "HyperspectralCamera";
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
struct Hdf5Construct<hdf5features::HyperspectralCameraIO, Derived>
{

    // DEPS
    using dep1 = typename Hdf5Construct<hdf5features::ArrayIO, Derived>::type;
    using dep2 = typename Hdf5Construct<hdf5features::MatrixIO, Derived>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<hdf5features::HyperspectralCameraIO>::type;
};

} // namespace lvr2

#include "HyperspectralCameraIO.tcc"

#endif // LVR2_IO_HDF5_HYPERSPECTRALCAMERAIO_HPP
