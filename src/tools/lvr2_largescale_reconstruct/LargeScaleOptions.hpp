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
 * LargeScaleOptions.hpp
 *
 *      Author: Isaak Mitschke
 */

#ifndef LARGESCALEOPTIONS_H_
#define LARGESCALEOPTIONS_H_

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

namespace LargeScaleOptions
{

/**
 * @brief A class to parse the program options for the reconstruction
 *           executable.
 */
class Options : public lvr2::BaseOption
{
  public:
    /**
     * @brief     Ctor. Parses the command parameters given to the main
     *               function of the program
     */
    Options(int argc, char** argv);
    virtual ~Options();

    /**
     * @brief   Returns if the new chunks should be written as a .ply-mesh
     */
    bool getBigMesh() const;

    /**
     * @brief   Returns if the mesh of every chunk additionally should be written as a .ply
     */
    bool getDebugChunks() const;

    /**
     * @brief   Returns if the GPU shuold be used for the normal estimation
     */
    bool useGPU() const;

    /**
     * @brief   Returns all voxelsizes as a vector
     */
    vector<float> getVoxelSizes() const;

    /**
     * @brief   Returns the first given voxelsize
     */
    float getVoxelsize() const;

    /**
     * @brief    Returns the given voxelsize for bigGrid
     */
    float getBGVoxelsize() const;

    /**
     * @brief   Returns the scaling factor
     */
    float getScaling() const;

    /**
     * @brief   Returns the chunksize
     */
     // TODO data types don't match!!
    int getChunkSize() const;

    /**
     * @brief   Only used in kd-tree (partMethod=0). Returns the maximum number of points in a leaf.
     */
    unsigned int getNodeSize() const;

    /**
     * @brief   Retuns flag for partition-method (0 = kd-Tree; 1 = VGrid)
     */
    int getPartMethod() const;

    /**
     * @brief    Returns the number of neighbors
     *             for normal interpolation
     */
    int getKi() const;

    /**
     * @brief    Returns the number of neighbors used for distance
     *             function evaluation
     */
    int getKd() const;

    /**
     * @brief    Returns the number of neighbors used for
     *             initial normal estimation
     */
    int getKn() const;

    /**
     * @brief   If true, RANSAC based normal estimation is used
     */
    bool useRansac() const;

    /**
     * @brief   Returns the flipPoint for GPU normal computation
     */
    vector<float> getFlippoint() const;

    /**
     * @brief   Whether to extend the grid. Enabled by default.
     */
    bool extrude() const;

    /*
     * Definition from here on are for the combine-process of partial meshes
     */

    /**
     * @brief   Returns the number of dangling artifacts to remove from
     *          a created mesh.
     */
    int getDanglingArtifacts() const;

    /**
     * @brief    Number of iterations for contour cleanup
     */
    int getCleanContourIterations() const;

    /**
     * @brief   Returns the region threshold for hole filling
     */
    int getFillHoles() const;

    /**
     * @brief     Returns true if cluster optimization is enabled
     */
    bool optimizePlanes() const;

    /**
     * @brief   Returns the normal threshold for plane optimization.
     */
    float getNormalThreshold() const;
    /**
     * @brief   Returns to number plane optimization iterations
     */
    int getPlaneIterations() const;

    /**
     * @brief   Returns the threshold for the size of small
     *          region deletion after plane optimization.
     */
    int getSmallRegionThreshold() const;

    /**
     * @brief Return whether the mesh should be retesselated or not.
     */
    bool retesselate() const;

    /**
     * @brief   Returns the fusion threshold for tesselation
     */
    float getLineFusionThreshold() const;

    /*
     * Definition from here on are not used (anymore?)
     */

    /**
     * @brief    Returns the number of used threads
     */
    int getNumThreads() const;

    /**
     * @brief    Prints a usage message to stdout.
     */
    bool printUsage() const;

    /**
     * @brief    Returns true if an output filen name was set
     */
    bool filenameSet() const;

    /**
     * @brief     Returns true if the face normals of the
     *             reconstructed mesh should be saved to an
     *             extra file ("face_normals.nor")
     */
    bool saveFaceNormals() const;

    /**
     *@brief    Returns true of region coloring is enabled.
     */
    bool colorRegions() const;

    /**
     * @brief    Returns true if the interpolated normals
     *             should be saved in the putput file
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
     * @brief     Indicates whether to save the used points
     *             together with the interpolated normals.
     */
    bool savePointNormals() const;

    /**
     * @brief    If true, normals should be calculated even if
     *             they are already given in the input file
     */
    bool recalcNormals() const;

    /**
     * @brief   True if texture analysis is enabled
     */
    bool doTextureAnalysis() const;

    /**
     * @brief    If true, textures will be generated during
     *          finalization of mesh.
     */
    bool generateTextures() const;

    /**
     * @brief  True if region clustering without plane optimization is required.
     */
    bool clusterPlanes() const;

    /**
     * @brief  True if region clustering without plane optimization is required.
     */
    bool writeClassificationResult() const;

    /**
     * @brief    Returns the output file name
     */
    std::vector<string> getInputFileName() const;

    /**
     * @brief    Returns the name of the classifier used to color the mesh
     */
    string getClassifier() const;

    /**
     * @brief    Returns the name of the given file with scan poses used for normal flipping.
     */
    string getScanPoseFile() const;

    /**
     * @brief   Returns the number of intersections. If the return value
     *          is positive it will be used for reconstruction instead of
     *          absolute voxelsize.
     */
    int getIntersections() const;

    /**
     * @brief   Returns the name of the used point cloud handler.
     */
    string getPCM() const;

    /**
     * @brief   Returns the name of the used point cloud handler.
     */
    string getDecomposition() const;

    /**
     * @brief   Minimum value for plane optimzation
     */
    int getMinPlaneSize() const;

    /**
     * @brief     Returns the maximum recursion depth for region growing
     */
    int getDepth() const;

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
     * @brief     Number of edge collapses
     */
    int getNumEdgeCollapses() const;

    bool onlyNormals() const;

    /**
     * @brief    Edge collapse method
     */
    string getEdgeCollapseMethod() const;

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
    unsigned int getBufferSize() const;

    bool interpolateBoxes() const;


    string getOutputFolderPath() const;

    bool getUseNormals() const;

    size_t getVolumenSize() const;

    size_t getLineReaderBuffer() const;

  private:
    /// flag to generate a .ply file for the reconstructed mesh
    bool m_bigMesh;

    /// flag to generate debug meshes for every chunk as a .ply
    bool m_debugChunks;

    /// The set voxelsizes
    vector<float> m_voxelSizes;

    /// The set voxelsize
    float m_voxelsize;

    /// The set voxelsize for BigGrid
    float m_voxelsizeBG;

    float m_scaling;

    /// gridsize for virtual grid
    int m_chunkSize;

    /// sets partition method to kd-tree or virtual grid
    int m_partMethod;

    /// The number of neighbors for normal interpolation
    int m_ki;

    /// The number of neighbors for distance function evaluation
    int m_kd;

    /// The number of neighbors for normal estimation
    int m_kn;

    /// flipPoint for GPU normal computation
    vector<float> m_flippoint;

    /// extruded flag
    bool m_extrude;

    /// Number of dangling artifacts to remove
    int m_removeDanglingArtifacts;

    /// number of cleanContour iterations
    int m_cleanContourIterations;

    /// Threshold for hole filling
    int m_fillHoles;

    /// Threshold for plane optimization
    float m_planeNormalThreshold;

    /// Number of iterations for plane optimzation
    int m_planeIterations;

    /// Threshold for plane optimization
    int m_minPlaneSize;

    /// Threshold for small regions
    int m_smallRegionThreshold;

    /// Whether or not the mesh should be retesselated while being finalized
    bool m_retesselate;

    /// Threshold for line fusing when tesselating
    float m_lineFusionThreshold;


    /*
     * Definition from here on are not used (anymore?)
     */

    /// The number of used threads
    int m_numThreads;

    /// The putput file name for face normals
    string m_faceNormalFile;

    /// The number of used default values
    int m_numberOfDefaults;

    /// The number of intersections used for reconstruction
    int m_intersections;

    /// Whether or not the mesh should be retesselated while being finalized
    bool m_generateTextures;

    /// Whether or not the classifier shall dump meta data to a file 'clusters.clu'
    bool m_writeClassificationResult;

    /// The used point cloud manager
    string m_pcm;

    /// Maximum recursion depth for region growing
    int m_depth;

    /// Texel size
    float m_texelSize;

    /// Sharp feature threshold when using sharp feature decomposition
    float m_sft;

    /// Sharp corner threshold when using sharp feature decomposition
    float m_sct;

    /// Name of the classifier object to color the mesh
    string m_classifier;

    /// Edge collapse method
    string m_ecm;

    /// Number of edge collapses
    int m_numEdgeCollapses;

    string m_outputFolderPath;

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

    unsigned int m_bufferSize;

    unsigned int m_octreeNodeSize;

    bool m_interpolateBoxes;

    bool m_use_normals;

    size_t m_volumenSize;

    size_t m_lineReaderBuffer;

    bool m_onlyNormals;

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

    cout << "##### Voxel decomposition \t: " << o.getDecomposition() << endl;
    cout << "##### Classifier\t\t: " << o.getClassifier() << endl;
    if (o.writeClassificationResult())
    {
        cout << "##### Dump classification\t: YES" << endl;
    }
    else
    {
        cout << "##### Dump classification\t: NO" << endl;
    }
    cout << "##### k_i \t\t\t: " << o.getKi() << endl;
    cout << "##### k_d \t\t\t: " << o.getKd() << endl;
    cout << "##### k_n \t\t\t: " << o.getKn() << endl;
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
    if (o.generateTextures())
    {
        cout << "##### Generate Textures \t: YES" << endl;
        cout << "##### Texel size \t\t: " << o.getTexelSize() << endl;
        if (o.doTextureAnalysis())
        {
            cout << "##### Texture Analysis \t: OFF" << endl;
        }
        else
        {
            cout << "##### Texture Analysis \t\t: OFF" << endl;
        }
    }
    if (o.getDepth())
    {
        cout << "##### Recursion depth \t\t: " << o.getDepth() << endl;
    }
    if (o.getNumEdgeCollapses())
    {
        cout << "##### Edge collapse method \t\t: " << o.getEdgeCollapseMethod() << endl;
        cout << "##### Number of edge collapses\t: " << o.getNumEdgeCollapses() << endl;
    }

    if (o.getNodeSize())
    {
        cout << "##### Leaf Size \t\t: " << o.getNodeSize() << endl;
    }

    cout << "##### Interpolating Boxes \t: " << o.interpolateBoxes() << endl;

    if (o.getBufferSize())
    {
        cout << "##### Buffer Size \t\t: " << o.getBufferSize() << endl;
    }
    cout << "##### Volumen Size \t\t: " << o.getVolumenSize() << endl;
    return os;
}

} // namespace LargeScaleOptions

#endif /* LARGESCALEOPTIONS_H_ */
