#pragma once

#ifndef LABELIO
#define LABELIO

#include "lvr2/io/baseio/ArrayIO.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "WaveformIO.hpp"

#include <yaml-cpp/yaml.h>

using lvr2::baseio::ArrayIO;
using lvr2::baseio::FeatureConstruct;

namespace lvr2
{
namespace scanio
{

template <typename BaseIO>
class LabelIO
{
public:
  void saveLabels(std::string &group, LabelRootPtr rootPtr) const;

  LabelRootPtr loadLabels(const std::string& groupName) const;
  
protected:

  BaseIO *m_baseIO = static_cast<BaseIO*>(this);

  // dependencies
  ArrayIO<BaseIO> *m_arrayIO = static_cast<ArrayIO<BaseIO> *>(m_baseIO);
  FullWaveformIO<BaseIO>* m_fullWaveformIO = static_cast<FullWaveformIO<BaseIO>*>(m_baseIO);
  static constexpr const char* ID = "LabelIO";
  static constexpr const char* OBJID = "Label";

};

} // namespace  scanio

template<typename T>
struct FeatureConstruct<scanio::LabelIO, T> {
    
    // DEPS
    using dep1 = typename FeatureConstruct<baseio::ArrayIO, T>::type;
    using dep2 = typename FeatureConstruct<scanio::FullWaveformIO, T>::type;
    using deps = typename dep1::template Merge<dep2>;
 

    // add actual feature
    using type = typename deps::template add_features<scanio::LabelIO>::type;
     
};



} // namespace lvr2

#include "LabelIO.tcc"

#endif // LABELIO
