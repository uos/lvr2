#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_WAVEFORMIO_HPP
#define LVR2_IO_DESCRIPTIONS_WAVEFORMIO_HPP

#include "lvr2/io/scanio/ArrayIO.hpp"
#include "lvr2/io/scanio/MatrixIO.hpp"
#include "lvr2/io/scanio/yaml/Matrix.hpp"
#include "lvr2/io/scanio/yaml/Waveform.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <sstream>
#include <yaml-cpp/yaml.h>
namespace lvr2 
{

template <typename FeatureBase>
class FullWaveformIO
{
public:
  void saveFullWaveform(const size_t& scanPosNo, const size_t& scanNo, const WaveformPtr &buffer);
  void saveLabelWaveform(const std::string& group, const WaveformPtr &buffer);

  WaveformPtr loadLabelWaveform(const std::string& groupName);
  
  WaveformPtr loadFullWaveform(const size_t& scanPosNo, const size_t& scanNo);
protected:
  FeatureBase *m_featureBase = static_cast<FeatureBase *>(this);

// dependencies
 MatrixIO<FeatureBase>* m_matrixIO = static_cast<MatrixIO<FeatureBase>*>(m_featureBase);
  static constexpr const char *ID = "FullWaveformIO";
  static constexpr const char *OBJID = "FullWaveform";
};

/**
 *
 * @brief FeatureConstruct Specialization for WaveformIO
 * - Constructs dependencies (ArrayIO, MatrixIO)
 * - Sets type variable
 *
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<FullWaveformIO, FeatureBase>
{

    // DEPS
    using deps = typename FeatureConstruct<MatrixIO, FeatureBase>::type;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<FullWaveformIO>::type;
};

} // namespace lvr2

#include "WaveformIO.tcc"

#endif // LVR2_IO_HDF5_WAVEFORMIO_HPP
