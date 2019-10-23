/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Options.h
 *
 *  Created on: Nov 21, 2010
 *      Author: Thomas Wiemann
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include "lvr2/config/BaseOption.hpp"

#include <boost/program_options.hpp>
#include <float.h>
#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::ostream;
using std::string;
using std::vector;

namespace reconstruct
{

/**
 * @brief A class to parse the program options for the reconstruction
 *        executable.
 */
class Options : public lvr2::BaseOption
{
  public:
    /**
     * @brief   Ctor. Parses the command parameters given to the main
     *          function of the program
     */
    Options(int argc, char** argv);
    virtual ~Options();

    /**
     * @brief   Returns the given voxelsize
     */
    float getVoxelsize() const;

    /**
     * @brief   Returns the number of used threads
     */
    int getNumThreads() const;

    /**
     * @brief   Prints a usage message to stdout.
     */
    bool printUsage() const;

    /**
     * @brief   Returns true if an output filen name was set
     */
    bool filenameSet() const;

    /**
     * @brief   Returns true if the face normals of the
     *          reconstructed mesh should be saved to an
     *          extra file ("face_normals.nor")
     */
    bool saveFaceNormals() const;

    /**
     *@brief    Returns true of region coloring is enabled.
     */
    bool colorRegions() const;

    /**
     * @brief   Returns true if the interpolated normals
     *          should be saved in the putput file
     */
    bool saveNormals() const;

    /**
     * @brief   Returns true if the Marching Cubes grid should be save
     */
    bool saveGrid() const;

    /**
     * @brief   Returns true if the original points should be stored
     *          together with the reconstruction
     */
    bool saveOriginalData() const;

    /**
     * @brief   Returns true if cluster optimization is enabled
     */
    bool optimizePlanes() const;

    /**
     * @brief   Indicates whether to save the used points
     *          together with the interpolated normals.
     */
    bool savePointNormals() const;

    /**
     * @brief   If true, normals should be calculated even if
     *          they are already given in the input file
     */
    bool recalcNormals() const;

    /**
     * @brief   If true, RANSAC based normal estimation is used
     */
    bool useRansac() const;

    /**
     * @brief   True if texture analysis is enabled
     */
    bool doTextureAnalysis() const;

    /**
     * @brief   If true, textures will be generated during
     *          finalization of mesh.
     */
    bool generateTextures() const;

    /**
     * @brief   Returns the number of neighbors
     *          for normal interpolation
     */
    int getKi() const;

    /**
     * @brief   Returns the number of neighbors used for
     *          initial normal estimation
     */
    int getKn() const;

    /**
     * @brief   Returns the number of neighbors used for distance
     *          function evaluation
     */
    int getKd() const;

    /**
     * @brief Return whether the mesh should be retesselated or not.
     */
    bool retesselate() const;

    /**
     * @brief  True if region clustering without plane optimization is required.
     */
    bool clusterPlanes() const;

    /**
     * @brief  True if region clustering without plane optimization is required.
     */
    bool writeClassificationResult() const;

    /**
     * @brief   Returns the input file name
     */
    string getInputFileName() const;

    /**
     * @brief   Returns the output file name
     */
    string getOutputFileName() const;

    /**
     * @brief   Returns the output file names
     */
    vector<string> getOutputFileNames() const;

    /**
     * @brief   Returns the name of the classifier used to color the mesh
     */
    string getClassifier() const;

    /**
     * @brief   Returns the name of the given file with scan poses used for normal flipping.
     */
    string getScanPoseFile() const;

    /**
     * @brief   Returns the number of intersections. If the return value
     *          is positive it will be used for reconstruction instead of
     *          absolute voxelsize.
     */
    int getIntersections() const;

    /**
     * @brief   Returns to number plane optimization iterations
     */
    int getPlaneIterations() const;

    /**
     * @brief   Returns the name of the used point cloud handler.
     */
    string getPCM() const;

    /**
     * @brief   Returns the name of the used point cloud handler.
     */
    string getDecomposition() const;

    /**
     * @brief   Returns the normal threshold for plane optimization.
     */
    float getNormalThreshold() const;

    /**
     * @brief   Returns the threshold for the size of small
     *          region deletion after plane optimization.
     */
    int getSmallRegionThreshold() const;

    /**
     * @brief   Minimum value for plane optimzation
     */
    int getMinPlaneSize() const;

    /**
     * @brief   Number of iterations for contour cleanup
     */
    int getCleanContourIterations() const;

    /**
     * @brief   Returns the number of dangling artifacts to remove from
     *          a created mesh.
     */
    int getDanglingArtifacts() const;

    /**
     * @brief   Returns the region threshold for hole filling
     */
    int getFillHoles() const;

    /**
     * @brief   Returns the texel size for texture resolution
     */
    float getTexelSize() const;

    /**
     * @brief   Returns the sharp feature threshold when using sharp feature decomposition
     */
    float getSharpFeatureThreshold() const;

    /**
     * @brief   Returns the sharp corner threshold when using sharp feature decomposition
     */
    float getSharpCornerThreshold() const;

    /**
     * @brief   Returns the fusion threshold for tesselation
     */
    float getLineFusionThreshold() const;

    /**
     * @brief   Whether to extend the grid. Enable by default.
     */
    bool extrude() const;

    /**
     * @brief Reduction ratio for mesh reduction via edge collapse
     */
    float getEdgeCollapseReductionRatio() const;

    unsigned int getNumStatsColors() const;

    unsigned int getNumCCVColors() const;

    unsigned int getCoherenceThreshold() const;

    float getColorThreshold() const;

    float getStatsThreshold() const;

    float getFeatureThreshold() const;

    bool getUseCrossCorr() const;

    float getPatternThreshold() const;

    float* getStatsCoeffs() const;

    string getTexturePack() const;

    int getMinimumTransformationVotes() const;

    int getTexMinClusterSize() const;

    int getTexMaxClusterSize() const;

    bool vertexColorsFromPointcloud() const;

    bool useGPU() const;

    vector<float> getFlippoint() const;

    bool texturesFromImages() const;

    string getProjectDir() const;

  private:
    /// The set voxelsize
    float m_voxelsize;

    /// The number of uesed threads
    int m_numThreads;

    /// The putput file name for face normals
    string m_faceNormalFile;

    /// The number of used default values
    int m_numberOfDefaults;

    /// The number of neighbors for distance function evaluation
    int m_kd;

    /// The number of neighbors for normal estimation
    int m_kn;

    /// The number of neighbors for normal interpolation
    int m_ki;

    /// The number of intersections used for reconstruction
    int m_intersections;

    /// Whether or not the mesh should be retesselated while being finalized
    bool m_retesselate;

    /// Whether or not the mesh should be retesselated while being finalized
    bool m_generateTextures;

    /// Whether or not the classifier shall dump meta data to a file 'clusters.clu'
    bool m_writeClassificationResult;

    /// The used point cloud manager
    string m_pcm;

    /// Number of iterations for plane optimzation
    int m_planeIterations;

    /// Threshold for plane optimization
    float m_planeNormalThreshold;

    /// Threshold for small ragions
    int m_smallRegionThreshold;

    /// Number of dangling artifacts to remove
    int m_rda;

    /// Threshold for hole filling
    int m_fillHoles;

    /// Threshold for plane optimization
    int m_minPlaneSize;

    int m_cleanContourIterations;

    /// Texel size
    float m_texelSize;

    /// Threshold for line fusing when tesselating
    float m_lineFusionThreshold;

    /// Sharp feature threshold when using sharp feature decomposition
    float m_sft;

    /// Sharp corner threshold when using sharp feature decomposition
    float m_sct;

    /// Name of the classifier object to color the mesh
    string m_classifier;

    /// Reduction ratio for mesh reduction via edge collapse
    float m_edgeCollapseReductionRatio;

    /// Path to texture pack
    string m_texturePack;

    /// Coefficents file for texture matching based on statistics
    string m_statsCoeffs;

    /// Number of colors for texture statistics
    unsigned int m_numStatsColors;

    /// Number of colors for texture matching based on color information
    unsigned int m_numCCVColors;

    /// Coherence threshold for texture matching based on color information
    unsigned int m_coherenceThreshold;

    /// Threshold for texture matching based on colors
    float m_colorThreshold;

    /// Threshold for texture matching based on statistics
    float m_statsThreshold;

    /// Threshold for texture matching based on features
    float m_featuresThreshold;

    /// Whether to use texture matching based on cross correlation
    bool m_useCrossCorr;

    /// Threshold for pattern extraction from textures
    float m_patternThreshold;

    /// Minimum number of vote to consider a texture transformation as "correct"
    int m_minimumTransformationVotes;

    /// Minimum number of textures of a cluster needed for generating textures
    int m_texMinClusterSize;

    int m_texMaxClusterSize;

    /// Use pointcloud colors to paint vertices
    bool m_vertexColorsFromPointcloud;

    // Flippoint for gpu normal calculation
    vector<float> m_flippoint;
};

/// Overlaoeded outpur operator
inline ostream& operator<<(ostream& os, const Options& o)
{
    o.printTransformation(os);

    if (o.getIntersections() > 0)
    {
        cout << "##### Intersections \t\t: " << o.getIntersections() << endl;
    }
    else
    {
        cout << "##### Voxelsize \t\t: " << o.getVoxelsize() << endl;
    }
    cout << "##### Number of threads \t: " << o.getNumThreads() << endl;
    cout << "##### Point cloud manager \t: " << o.getPCM() << endl;
    if (o.useRansac())
    {
        cout << "##### Use RANSAC\t\t: YES" << endl;
    }
    else
    {
        cout << "##### Use RANSAC\t\t: NO" << endl;
    }

    cout << "##### Voxel decomposition: \t: " << o.getDecomposition() << endl;
    cout << "##### Classifier:\t\t: " << o.getClassifier() << endl;
    if (o.writeClassificationResult())
    {
        cout << "##### Dump classification\t: YES" << endl;
    }
    else
    {
        cout << "##### Dump classification\t: NO" << endl;
    }
    cout << "##### k_n \t\t\t: " << o.getKn() << endl;
    cout << "##### k_i \t\t\t: " << o.getKi() << endl;
    cout << "##### k_d \t\t\t: " << o.getKd() << endl;
    if (o.getDecomposition() == "SF")
    {
        cout << "##### Sharp feature threshold \t: " << o.getSharpFeatureThreshold() << endl;
        cout << "##### Sharp corner threshold \t: " << o.getSharpCornerThreshold() << endl;
    }
    if (o.retesselate())
    {
        cout << "##### Retesselate \t\t: YES" << endl;
        cout << "##### Line fusion threshold \t: " << o.getLineFusionThreshold() << endl;
    }
    if (o.saveFaceNormals())
    {
        cout << "##### Write Face Normals \t: YES" << endl;
    }

    if (o.getFillHoles())
    {
        cout << "##### Fill holes \t\t: " << o.getFillHoles() << endl;
    }
    else
    {
        cout << "##### Fill holes \t\t: NO" << endl;
    }

    if (o.getDanglingArtifacts())
    {
        cout << "##### Remove DAs \t\t: " << o.getDanglingArtifacts() << endl;
    }
    else
    {
        cout << "##### Remove DAs \t\t: NO" << endl;
    }

    if (o.optimizePlanes())
    {
        cout << "##### Optimize Planes \t\t: YES" << endl;
        cout << "##### Plane iterations\t\t: " << o.getPlaneIterations() << endl;
        cout << "##### Normal threshold \t\t: " << o.getNormalThreshold() << endl;
        cout << "##### Region threshold\t\t: " << o.getSmallRegionThreshold() << endl;
        cout << "##### Region min size\t\t: " << o.getMinPlaneSize() << endl;
    }
    if (o.saveNormals())
    {
        cout << "##### Save normals \t\t: YES" << endl;
    }
    if (o.saveOriginalData())
    {
        cout << "##### Save input data \t\t: YES" << endl;
    }

    if (o.recalcNormals())
    {
        cout << "##### Recalc normals \t\t: YES" << endl;
    }
    if (o.savePointNormals())
    {
        cout << "##### Save points normals \t: YES" << endl;
    }
    if (o.vertexColorsFromPointcloud())
    {
        cout << "##### Vertex colors: \t\t: YES" << endl;
    }
    if (o.generateTextures())
    {
        cout << "##### Generate Textures \t: YES" << endl;
        cout << "##### Texel size \t\t: " << o.getTexelSize() << endl;
        cout << "##### Texture Min#Cluster \t: " << o.getTexMinClusterSize() << endl;
        cout << "##### Texture Max#Cluster \t: " << o.getTexMaxClusterSize() << endl;

        if (o.doTextureAnalysis())
        {
            cout << "##### Texture Analysis \t: ON" << endl;
        }
        else
        {
            cout << "##### Texture Analysis \t\t: OFF" << endl;
        }
    }
    if (o.getEdgeCollapseReductionRatio() > 0.0)
    {
        cout << "##### Edge collapse reduction ratio\t: " << o.getEdgeCollapseReductionRatio()
             << endl;
    }

    if (o.useGPU())
    {
        cout << "##### GPU normal estimation \t: ON" << endl;
        std::vector<float> flipPoint = o.getFlippoint();
        cout << "##### Flippoint \t\t: (" << flipPoint[0] << ", " << flipPoint[1] << ", "
             << flipPoint[2] << ")" << endl;
    }
    else
    {
        cout << "##### GPU normal estimation \t: OFF" << endl;
    }

    return os;
}

} // namespace reconstruct

#endif /* OPTIONS_H_ */
