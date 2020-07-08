#pragma once

#ifndef LVR2_IO_HDF5_FULLWAVEFORMIO_HPP
#define LVR2_IO_HDF5_FULLWAVEFORMIO_HPP

#include "lvr2/io/descriptions/ArrayIO.hpp"
#include "lvr2/io/descriptions/MatrixIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

template <typename FeatureBase>
class FullWaveformIO
{
public:
  void saveFullWaveform(const size_t& scanPosNo, const size_t& scanNo, const FullWaveformPtr &buffer);

  FullWaveformPtr loadFullWaveform(const size_t& scanPosNo, const size_t& scanNo);
  
protected:
  FeatureBase *m_featureBase = static_cast<FeatureBase *>(this);

  // dependencies
  ArrayIO<FeatureBase> *m_arrayIO = static_cast<ArrayIO<FeatureBase> *>(m_featureBase);
  MatrixIO<FeatureBase> *m_matrixIO = static_cast<MatrixIO<FeatureBase> *>(m_featureBase);

  static constexpr const char *ID = "FullWaveformIO";
  static constexpr const char *OBJID = "FullWaveform";
};

/**
 *
 * @brief FeatureConstruct Specialization for ScanCameraIO
 * - Constructs dependencies (ArrayIO, MatrixIO)
 * - Sets type variable
 *
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<FullWaveformIO, FeatureBase>
{

    // DEPS
    using dep1 = typename FeatureConstruct<ArrayIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<MatrixIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<FullWaveformIO>::type;
};

} // namespace lvr2

#include "FullWaveformIO.tcc"

#endif // LVR2_IO_HDF5_FULLWAVEFORMIO_HPP
