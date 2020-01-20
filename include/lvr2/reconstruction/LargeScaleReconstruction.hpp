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

#ifndef LAS_VEGAS_LARGESCALERECONSTRUCTION_HPP
#define LAS_VEGAS_LARGESCALERECONSTRUCTION_HPP

#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{
template <typename BaseVecT>
class LargeScaleReconstruction
{

  public:
    /**
     * Constructor - uses default parameter for reconstruction)
     * @param h5File HDF5 file, which may or may not contain chunked and reconstructed scans
     */
    LargeScaleReconstruction(std::string h5File);

    /**
     * Constructor with parameters
     */
    LargeScaleReconstruction(std::string h5File,
                             float voxelSize,
                             float bgVoxelSize,
                             float scale,
                             size_t chunkSize,
                             uint nodeSize,
                             int partMethod,
                             int ki,
                             int kd,
                             int kn,
                             bool useRansac,
                             bool extrude,
                             int removeDanglingArtifacts,
                             int cleanContours,
                             int fillHoles,
                             bool optimizePlanes,
                             float getNormalThreshold,
                             int planeIterations,
                             int minPlaneSize,
                             int smallRegionThreshold,
                             bool retesselate,
                             float lineFusionThreshold);

    /**
     * Cnstructor with LargeScaleOption as parameters
     * @param options
     */
    LargeScaleReconstruction(LargeScaleOptions::Options options);

    /**
     * this method splits the given PointClouds in to Chunks and calculates all required values for
     * a later reconstruction
     *
     * @tparam BaseVecT
     * @param scans vector of new scan to be added
     * @return
     */
    int mpiChunkAndReconstruct(ScanProjectEditMarkPtr project);

    int resetEditMark(ScanProjectEditMarkPtr project);

  private:
    // TODO: add chunks vector somewhere

    // path to hdf5 path containing previously reconstructed scans (or no scans) only
    string m_filePath;

    // voxelsize for reconstruction. Default: 10
    float m_voxelSize;

    // voxelsize for the BigGrid. Default: 10
    float m_bgVoxelSize;

    // scale factor. Default: 1
    float m_scale;

    // ChunkSize, should be constant through all processes . Default: 20
    size_t m_chunkSize;

    // Max. Number of Points in a leaf (used to devide pointcloud). Default: 1000000
    uint m_nodeSize;

    // int flag to trigger partition-method (0 = kd-Tree; 1 = VGrid)
    int m_partMethod;

    // Number of normals used in the normal interpolation process. Default: 10
    int m_Ki;

    // Number of normals used for distance function evaluation. Default: 5
    int m_Kd;

    // Size of k-neighborhood used for normal estimation. Default: 10
    int m_Kn;

    // Set this flag for RANSAC based normal estimation. Default: false
    bool m_useRansac;

    // Do not extend grid. Can be used  to avoid artifacts in dense data sets but. Disabling
    // will possibly create additional holes in sparse data sets. Default: false
    bool m_extrude;

    /*
     * Definition from here on are for the combine-process of partial meshes
     */

    // flag to trigger the removal of dangling artifacts. Default: 0
    int m_removeDanglingArtifacts;

    // Remove noise artifacts from contours. Same values are between 2 and 4. Default: 0
    int m_cleanContours;

    // Maximum size for hole filling. Default: 0
    int m_fillHoles;

    // Shift all triangle vertices of a cluster onto their shared plane. Default: false
    bool m_optimizePlanes;

    // (Plane Normal Threshold) Normal threshold for plane optimization. Default: 0.85
    float m_getNormalThreshold;

    // Number of iterations for plane optimization. Default: 3
    int m_planeIterations;

    // Minimum value for plane optimization. Default: 7
    int m_MinPlaneSize;

    // Threshold for small region removal. If 0 nothing will be deleted. Default: 0
    int m_SmallRegionThreshold;

    // Retesselate regions that are in a regression plane. Implies --optimizePlanes. Default: false
    bool m_retesselate;

    // Threshold for fusing line segments while tesselating. Default: 0.01
    float m_LineFusionThreshold;
};
} // namespace lvr2

#include "lvr2/reconstruction/LargeScaleReconstruction.tcc"

#endif // LAS_VEGAS_LARGESCALERECONSTRUCTION_HPP
