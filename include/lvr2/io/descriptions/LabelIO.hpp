#pragma once

#ifndef LVR2_IO_HDF5_LABELIO_HPP
#define LVR2_IO_HDF5_LABELIO_HPP

#include "lvr2/io/descriptions/ArrayIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

template <typename FeatureBase>
class LabelIO
{
public:
  void saveLabels(std::string &group, LabelRootPtr rootPtr);

  LabelRootPtr loadLabels(const std::string& groupName);
  
protected:

  FeatureBase *m_featureBase = static_cast<FeatureBase*>(this);

  // dependencies
  ArrayIO<FeatureBase> *m_arrayIO = static_cast<ArrayIO<FeatureBase> *>(m_featureBase);
  static constexpr const char* ID = "LabelIO";
  static constexpr const char* OBJID = "Label";

};
template<typename FeatureBase>
struct FeatureConstruct<LabelIO, FeatureBase> {
    
    // DEPS
    using deps = typename FeatureConstruct<ArrayIO, FeatureBase>::type;
 

    // add actual feature
    using type = typename deps::template add_features<LabelIO>::type;
     
};



} // namespace lvr2

#include "LabelIO.tcc"

#endif // LABELIO
