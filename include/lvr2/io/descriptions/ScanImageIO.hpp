#pragma once

#ifndef LVR2_IO_HDF5_SCANIMAGEIO_HPP
#define LVR2_IO_HDF5_SCANIMAGEIO_HPP

#include "ImageIO.hpp"
#include "MatrixIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

template <typename FeatureBase>
class ScanImageIO
{
  public:
    void saveScanImage(const size_t& scanPos, const size_t& camNr, const size_t& imgNr, ScanImagePtr& buffer);
    ScanImagePtr loadScanImage(const size_t& scanPos, const size_t& camNr, const size_t& imgNr);
    
  protected:
    //bool isScanImage(const std::string& group);

    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

    // dependencies
    ImageIO<FeatureBase>* m_imageIO = static_cast<ImageIO<FeatureBase>*>(m_featureBase);
    MatrixIO<FeatureBase>* m_matrixIO = static_cast<MatrixIO<FeatureBase>*>(m_featureBase);

    static constexpr const char* ID = "ScanImageIO";
    static constexpr const char* OBJID = "ScanImage";
};

/**
 *
 * @brief FeatureConstruct Specialization for hdf5features::ScanImageIO
 * - Constructs dependencies (ArrayIO, MatrixIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<ScanImageIO, FeatureBase>
{
    // DEPS
    using dep1 = typename FeatureConstruct<ImageIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<MatrixIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<ScanImageIO>::type;
};

} // namespace lvr2

#include "ScanImageIO.tcc"

#endif // LVR2_IO_HDF5_SCANIMAGEIO_HPP
