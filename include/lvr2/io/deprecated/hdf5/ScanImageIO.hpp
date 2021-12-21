#pragma once

#ifndef LVR2_IO_HDF5_SCANIMAGEIO_HPP
#define LVR2_IO_HDF5_SCANIMAGEIO_HPP

#include "ImageIO.hpp"
#include "MatrixIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
class ScanImageIO
{
  public:
    void save(const std::string& , const ScanImagePtr& buffer);

    void save(uint scanPos,
              uint camNr,
              uint imgNr,
              const ScanImagePtr& scanImagePtr);

    void save(HighFive::Group& group, uint camNr, uint imgNr, const ScanImagePtr& buffer);
    void save(HighFive::Group& group, uint imgNr, const ScanImagePtr& buffer);
    void save(HighFive::Group& group, const ScanImagePtr& buffer);

    ScanImagePtr load(uint scanPos, uint camNr, uint imgNr);
    ScanImagePtr load(HighFive::Group& group, uint camNr, uint imgNr);
    ScanImagePtr load(HighFive::Group& group, uint imgNr);
    ScanImagePtr load(HighFive::Group& group);

  protected:
    bool isScanImage(HighFive::Group& group);

    Derived* m_file_access = static_cast<Derived*>(this);

    // dependencies
    ImageIO<Derived>* m_imageIO = static_cast<ImageIO<Derived>*>(m_file_access);
    MatrixIO<Derived>* m_matrixIO = static_cast<MatrixIO<Derived>*>(m_file_access);

    static constexpr const char* ID = "ScanImageIO";
    static constexpr const char* OBJID = "ScanImage";
};

} // namespace hdf5features

/**
 *
 * @brief Hdf5Construct Specialization for hdf5features::ScanImageIO
 * - Constructs dependencies (ArrayIO, MatrixIO)
 * - Sets type variable
 *
 */
template <typename Derived>
struct Hdf5Construct<hdf5features::ScanImageIO, Derived>
{
    // DEPS
    using dep1 = typename Hdf5Construct<hdf5features::ImageIO, Derived>::type;
    using dep2 = typename Hdf5Construct<hdf5features::MatrixIO, Derived>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<hdf5features::ScanImageIO>::type;
};

} // namespace lvr2

#include "ScanImageIO.tcc"

#endif // LVR2_IO_HDF5_SCANIMAGEIO_HPP
