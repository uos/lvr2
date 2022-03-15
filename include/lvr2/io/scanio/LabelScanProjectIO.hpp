#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_LABELSCANPROJECTIO_HPP
#define LVR2_IO_DESCRIPTIONS_LABELSCANPROJECTIO_HPP

#include "lvr2/types/ScanTypes.hpp"

#include <boost/optional.hpp>
#include <regex>

// Dependencies
#include "ScanProjectIO.hpp"
#include "LabelIO.hpp"

namespace lvr2
{

namespace scanio
{

template <typename FeatureBase>
class LabelScanProjectIO
{
  public:
    void saveLabelScanProject(const LabeledScanProjectEditMarkPtr& labelscanProjectPtr);

    LabeledScanProjectEditMarkPtr loadLabelScanProject();
    ScanProjectPtr loadScanProject();

  protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
    // dependencies
    ScanProjectIO<FeatureBase>* m_scanProjectIO =
        static_cast<ScanProjectIO<FeatureBase>*>(m_featureBase);
    LabelIO<FeatureBase>* m_labelIO = 
        static_cast<LabelIO<FeatureBase>*>(m_featureBase);

    // static constexpr const char* ID = "ScanProjectIO";
    // static constexpr const char* OBJID = "ScanProject";
};

} // namespace scanio

template <typename FeatureBase>
struct FeatureConstruct<lvr2::scanio::LabelScanProjectIO, FeatureBase>
{

    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::scanio::ScanProjectIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::LabelIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;

    // add the feature itself
    using type = typename deps::template add_features<scanio::LabelScanProjectIO>::type;
};

} // namespace lvr2

#include "LabelScanProjectIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_LABELSCANPROJECTIO_HPP
