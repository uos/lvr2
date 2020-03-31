#pragma once

#ifndef LVR2_IO_HDF5_HYPERSPECTRALCAMERAIO_HPP
#define LVR2_IO_HDF5_HYPERSPECTRALCAMERAIO_HPP

#include "lvr2/io/descriptions/ArrayIO.hpp"
#include "lvr2/io/descriptions/MatrixIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

template <typename FeatureBase>
class HyperspectralCameraIO
{
public:
  void saveHyperspectralCamera(const size_t& scanPosNo, const HyperspectralCameraPtr &buffer);
  void saveHyperspectralCamera(std::string &group, const HyperspectralCameraPtr &buffer);

  HyperspectralCameraPtr loadHyperspectralCamera(const size_t& scanPosNo);
  
protected:
  bool isHyperspectralCamera(std::string &path);

  FeatureBase *m_featureBase = static_cast<FeatureBase *>(this);

  // dependencies
  ArrayIO<FeatureBase> *m_arrayIO = static_cast<ArrayIO<FeatureBase> *>(m_featureBase);
  MatrixIO<FeatureBase> *m_matrixIO = static_cast<MatrixIO<FeatureBase> *>(m_featureBase);

  static constexpr const char *ID = "HyperspectralCameraIO";
  static constexpr const char *OBJID = "HyperspectralCamera";
};

/**
 *
 * @brief FeatureConstruct Specialization for ScanCameraIO
 * - Constructs dependencies (ArrayIO, MatrixIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<HyperspectralCameraIO, FeatureBase>
{

    // DEPS
    using dep1 = typename FeatureConstruct<ArrayIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<MatrixIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<HyperspectralCameraIO>::type;
};

} // namespace lvr2

#include "HyperspectralCameraIO.tcc"

#endif // LVR2_IO_HDF5_HYPERSPECTRALCAMERAIO_HPP
