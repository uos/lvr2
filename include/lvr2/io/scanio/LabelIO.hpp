#pragma once

#ifndef LVR2_IO_scanio_LABELIO_HPP
#define LVR2_IO_scanio_LABELIO_HPP

#include "lvr2/io/scanio/ArrayIO.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "WaveformIO.hpp"

#include <yaml-cpp/yaml.h>
namespace lvr2
{

namespace scanio
{

template <typename FeatureBase>
class LabelIO
{
public:
  void saveLabels(std::string &group, LabelRootPtr rootPtr) const;

  LabelRootPtr loadLabels(const std::string& groupName) const;
  
protected:

  FeatureBase *m_featureBase = static_cast<FeatureBase*>(this);

  // dependencies
  ArrayIO<FeatureBase> *m_arrayIO = static_cast<ArrayIO<FeatureBase> *>(m_featureBase);
  FullWaveformIO<FeatureBase>* m_fullWaveformIO =
        static_cast<FullWaveformIO<FeatureBase>*>(m_featureBase);
  static constexpr const char* ID = "LabelIO";
  static constexpr const char* OBJID = "Label";

};

} // namespace  scanio

template<typename FeatureBase>
struct FeatureConstruct<scanio::LabelIO, FeatureBase> {
    
    // DEPS
    using dep1 = typename FeatureConstruct<scanio::ArrayIO, FeatureBase >::type;
    using dep2 = typename FeatureConstruct<scanio::FullWaveformIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;
 

    // add actual feature
    using type = typename deps::template add_features<scanio::LabelIO>::type;
     
};



} // namespace lvr2

#include "LabelIO.tcc"

#endif // LVR2_IO_scanio_LABELIO_HPP
