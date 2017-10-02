/*
 * FurnitureFeatureClassifier.hpp
 *
 *  Created on: Nov 25, 2016
 *      Author: robot
 */

#ifndef INCLUDE_LVR_CLASSIFICATION_FURNITUREFEATURECLASSIFIER_HPP_
#define INCLUDE_LVR_CLASSIFICATION_FURNITUREFEATURECLASSIFIER_HPP_

#include <vector>
#include <iostream>

#include <lvr/classification/RegionClassifier.hpp>

namespace lvr {

enum PlanarClusterOrientation{ HORIZONTAL, VERTICAL, UNKNOWN};

struct PlanarClusterFeature
{
    int index;            // Index of the plane (all planes are numbered in the clustering state
    float cx;            // x-coordinate of the center of the bounding box
    float cy;            // y-coordinate of the center of the bounding box
    float cz;            // z-coordinate of the center of the bounding box
    float nx;            // Normal x
    float ny;            // Normal y
    float nz;            // Normal z
    float area;            // Area of the planer region
    float w;            // Bounding box width (parallel to x axis)
    float h;            // Bounding box height (parallel to y axis)
    float d;            // Bounding box depth (parallel to z axis)
    PlanarClusterOrientation orientation;        // Orientation flag
};



template<typename VertexT, typename NormalT>
class FurnitureFeatureClassifier : public RegionClassifier<VertexT, NormalT>{
public:
    FurnitureFeatureClassifier(vector<Region<VertexT, NormalT>* >* region);
    virtual ~FurnitureFeatureClassifier();


    /**
     * @brief Returns the r component for the given region
     */
    virtual uchar r(int region) {  return 0; }

    /**
     * @brief Returns the g component for the given region
     */
    virtual uchar g(int region) {  return 255;}

    /**
     * @brief Returns the b component for the given region
     */
    virtual uchar b(int region) {  return 0; }

    /***
     * @brief     Try generate a classification label for region \ref region
     *             and store it in the feature vector \ref m_features if classification
     *             was successful.
     */
    void classifyRegion(int region);

    /***
     * @brief     Returns the number of generated features in the classification stage
     */
    size_t numFeatures() { return m_features.size();}

    /***
     * @brief     Return the n-th features from the feature vector
     */
    PlanarClusterFeature& getFeature(size_t n);

    /**
     * @brief     Returns a pointer to the n-th Region from the internal
     *             region vector. You can get the number of a region from a
     *             \ref PlanarClusterFeature's id field.
     */
    Region<VertexT, NormalT>* getRegion(size_t n);

private:

    /// A vector containing the planar feature vectors for all
    /// classified regions
    vector<PlanarClusterFeature> m_features;
};

} /* namespace lvr */

#include "FurnitureFeatureClassifier.tcc"

#endif /* INCLUDE_LVR_CLASSIFICATION_FURNITUREFEATURECLASSIFIER_HPP_ */