#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_WAVEFORMIO_HPP
#define LVR2_IO_DESCRIPTIONS_WAVEFORMIO_HPP

#include <sstream>
#include <yaml-cpp/yaml.h>

#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/baseio/BaseIO.hpp"
#include "lvr2/io/baseio/ArrayIO.hpp"
#include "lvr2/io/baseio/MatrixIO.hpp"
#include "lvr2/io/YAML.hpp"

using lvr2::baseio::MatrixIO;
using lvr2::baseio::FeatureConstruct;

namespace lvr2 
{
namespace scanio
{

  template <typename BaseIO>
  class FullWaveformIO
  {
  public:
    void saveFullWaveform(const size_t &scanPosNo, const size_t &scanNo, const WaveformPtr &buffer);
    void saveLabelWaveform(const std::string &group, const WaveformPtr &buffer);

    WaveformPtr loadLabelWaveform(const std::string &groupName);

    WaveformPtr loadFullWaveform(const size_t &scanPosNo, const size_t &scanNo);

  protected:
    BaseIO *m_baseIO = static_cast<BaseIO *>(this);

    // dependencies
    MatrixIO<BaseIO> *m_matrixIO = static_cast<MatrixIO<BaseIO> *>(m_baseIO);
    static constexpr const char *ID = "FullWaveformIO";
    static constexpr const char *OBJID = "FullWaveform";
  };

} // namespace scanio

/**
 *
 * @brief FeatureConstruct Specialization for WaveformIO
 * - Constructs dependencies (ArrayIO, MatrixIO)
 * - Sets type variable
 *
 *
 */
template <typename T>
struct FeatureConstruct<lvr2::scanio::FullWaveformIO, T>
{

    // DEPS
    using deps = typename FeatureConstruct<MatrixIO, T>::type;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::FullWaveformIO>::type;
};

} // namespace lvr2

#include "WaveformIO.tcc"

#endif // LVR2_IO_HDF5_WAVEFORMIO_HPP
