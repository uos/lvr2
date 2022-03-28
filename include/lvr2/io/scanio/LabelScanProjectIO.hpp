#pragma once

#ifndef LABELSCANPROJECTIO
#define LABELSCANPROJECTIO

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

template <typename BaseIO>
class LabelScanProjectIO
{
  public:
    void saveLabelScanProject(const LabeledScanProjectEditMarkPtr& labelscanProjectPtr);

    LabeledScanProjectEditMarkPtr loadLabelScanProject();
    ScanProjectPtr loadScanProject();

  protected:
    BaseIO* m_baseIO = static_cast<BaseIO*>(this);
    // dependencies
    ScanProjectIO<BaseIO>* m_scanProjectIO = static_cast<ScanProjectIO<BaseIO>*>(m_baseIO);
    LabelIO<BaseIO>* m_labelIO = static_cast<LabelIO<BaseIO>*>(m_baseIO);

    // static constexpr const char* ID = "ScanProjectIO";
    // static constexpr const char* OBJID = "ScanProject";
};

} // namespace scanio

template <typename T>
struct FeatureConstruct<lvr2::scanio::LabelScanProjectIO, T>
{

    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::scanio::ScanProjectIO, T>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::LabelIO, T>::type;
    using deps = typename dep1::template Merge<dep2>;

    // add the feature itself
    using type = typename deps::template add_features<scanio::LabelScanProjectIO>::type;
};

} // namespace lvr2

#include "LabelScanProjectIO.tcc"

#endif // LABELSCANPROJECTIO
