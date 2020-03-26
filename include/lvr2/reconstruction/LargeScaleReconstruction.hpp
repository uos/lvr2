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
#include "lvr2/reconstruction/PointsetGrid.hpp"
#include "lvr2/reconstruction/FastBox.hpp"
#include "lvr2/algorithm/ChunkManager.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"


namespace lvr2
{
    struct LSROptions
    {
        //flag to trigger .ply output of big Mesh
        bool bigMesh = true;

        // flag to trigger .ply output of chunks
        bool debugChunks = false;

        // flag to trigger GPU usage
        bool useGPU = false;

        // voxelsizes for reconstruction.
        std::vector<float> voxelSizes{0.1};

        // voxelsize for the BigGrid.
        float bgVoxelSize = 1;

        // scale factor.
        float scale = 1;

        // Max. Number of Points in a leaf (used to devide pointcloud).
        uint nodeSize = 1000000;

        // int flag to trigger partition-method (0 = kd-Tree; 1 = VGrid)
        int partMethod = 1;

        //Number of normals used in the normal interpolation process.
        int ki = 20;

        //Number of normals used for distance function evaluation.
        int kd = 20;

        // Size of k-neighborhood used for normal estimation.
        int kn = 20;

        //Set this flag for RANSAC based normal estimation.
        bool useRansac = false;

        // FlipPoint for GPU normal computation
        std::vector<float> flipPoint{10000000, 10000000, 10000000};

        // Do not extend grid. Can be used to avoid artifacts in dense data sets but. Disabling
        // will possibly create additional holes in sparse data sets.
        bool extrude = false;

        /*
         * Definition from here on are for the combine-process of partial meshes
         */

        // number for the removal of dangling artifacts.
        int removeDanglingArtifacts = 0;

        //Remove noise artifacts from contours. Same values are between 2 and 4.
        int cleanContours = 0;

        //Maximum size for hole filling.
        int fillHoles = 0;

        // Shift all triangle vertices of a cluster onto their shared plane.
        bool optimizePlanes = false;

        // (Plane Normal Threshold) Normal threshold for plane optimization.
        float planeNormalThreshold = 0.85;

        // Number of iterations for plane optimization.
        int planeIterations = 3;

        // Minimum value for plane optimization.
        int minPlaneSize = 7;

        // Threshold for small region removal. If 0 nothing will be deleted.
        int smallRegionThreshold = 0;

        // Retesselate regions that are in a regression plane. Implies --optimizePlanes.
        bool retesselate = false;

        // Threshold for fusing line segments while tesselating.
        float lineFusionThreshold = 0.01;

        vector<float> getFlipPoint() const
        {
            std::vector<float> dest = flipPoint;
            if(dest.size() != 3)
            {
                std::vector<float> dest = std::vector<float>();
                dest.push_back(10000000);
                dest.push_back(10000000);
                dest.push_back(10000000);
            }
            return dest;
        }
    };

    template <typename BaseVecT>
    class LargeScaleReconstruction
    {


    public:
        /**
         * Constructor - uses default parameter for reconstruction
         */
        LargeScaleReconstruction();


        /**
         * Constructor with parameters
         */
        LargeScaleReconstruction(vector<float> voxelSizes, float bgVoxelSize, float scale,
                uint nodeSize, int partMethod,int ki, int kd, int kn, bool useRansac, std::vector<float> flipPoint,
                bool extrude, int removeDanglingArtifacts, int cleanContours, int fillHoles, bool optimizePlanes,
                float getNormalThreshold, int planeIterations, int minPlaneSize, int smallRegionThreshold,
                bool retesselate, float lineFusionThreshold, bool bigMesh, bool debugChunks, bool useGPU);

        /**
         * Constructor with parameters in a struct
         *
         */
         LargeScaleReconstruction(LSROptions options);

        /**
         * this method splits the given PointClouds in to Chunks and calculates all required values for a later reconstruction
         *
         * @param project ScanProject containing Scans
         * @param newChunksBB sets the Bounding Box of the reconstructed area
         * @param chunkManager a chunkManager to handle chunks
         * @tparam BaseVecT
         * @return
         */
        int mpiChunkAndReconstruct(ScanProjectEditMarkPtr project, BoundingBox<BaseVecT>& newChunksBB, std::shared_ptr<ChunkHashGrid> chunkManager);

        /**
         *
         * this methods splits the given PointClouds via kd-Tree and calculates all required values for a later reconstruction
         * @param project ScanProject containing Scans
         * @return
         */
        int mpiAndReconstruct(ScanProjectEditMarkPtr project);

        /**
         *
         * reconstruct a given area (+ neighboring chunks from a chunkmanager) with a given voxelsize
         *
         * @param newChunksBB area to be reconstructed
         * @param chunkHashGrid chunkmanager to manage chunks
         * @param voxelSize reconstruction parameter
         * @return reconstructed HalfEdgeMesh<BaseVecT>
         */
        HalfEdgeMesh<BaseVecT> getPartialReconstruct(BoundingBox<BaseVecT> newChunksBB, std::shared_ptr<ChunkHashGrid> chunkHashGrid,  float voxelSize);




    private:

        /**
         * This method adds the tsdf-values of one chunk to the ChunkManager-Layer
         *
         * @params x, y, z grid-coordinates for the chunk
         * @param ps_grid HashGrid which contains the tsdf-values for the voxel
         * @param cm ChunkManager instance which manages the chunks
         * @param layerName the name of the chunkManager-layer
         */
        void addTSDFChunkManager(int x, int y, int z,
                shared_ptr<lvr2::PointsetGrid<BaseVector<float>, lvr2::FastBox<BaseVector<float>>>> ps_grid,
                shared_ptr<ChunkHashGrid> cm,
                std::string layerName);

        //flag to trigger .ply output of big Mesh
        bool m_bigMesh = true;

        // flag to trigger .ply output of chunks
        bool m_debugChunks = false;

        // flag to trigger GPU usage
        bool m_useGPU = false;

        // path to hdf5 path containing previously reconstructed scans (or no scans) only
        string m_filePath;

        // voxelsize for reconstruction. Default: 10
        vector <float> m_voxelSizes;

        // voxelsize for the BigGrid. Default: 10
        float m_bgVoxelSize;

        // scale factor. Default: 1
        float m_scale;

        //ChunkSize, should be constant through all processes . Default: 20
        float m_chunkSize;

        // Max. Number of Points in a leaf (used to devide pointcloud). Default: 1000000
        uint m_nodeSize;

        // int flag to trigger partition-method (0 = kd-Tree; 1 = VGrid)
        int m_partMethod;

        //Number of normals used in the normal interpolation process. Default: 10
        int m_ki;

        //Number of normals used for distance function evaluation. Default: 5
        int m_kd;

        // Size of k-neighborhood used for normal estimation. Default: 10
        int m_kn;

        // flipPoint for GPU normal computation
        std::vector<float> m_flipPoint;

        //Set this flag for RANSAC based normal estimation. Default: false
        bool m_useRansac;

        // Do not extend grid. Can be used  to avoid artifacts in dense data sets but. Disabling
        // will possibly create additional holes in sparse data sets. Default: false
        bool m_extrude;

        /*
         * Definition from here on are for the combine-process of partial meshes
         */

        // flag to trigger the removal of dangling artifacts. Default: 0
        int m_removeDanglingArtifacts;

        //Remove noise artifacts from contours. Same values are between 2 and 4. Default: 0
        int m_cleanContours;

        //Maximum size for hole filling. Default: 0
        int m_fillHoles;

        // Shift all triangle vertices of a cluster onto their shared plane. Default: false
        bool m_optimizePlanes;

        // (Plane Normal Threshold) Normal threshold for plane optimization. Default: 0.85
        float m_planeNormalThreshold;

        // Number of iterations for plane optimization. Default: 3
        int m_planeIterations;

        // Minimum value for plane optimization. Default: 7
        int m_minPlaneSize;

        // Threshold for small region removal. If 0 nothing will be deleted. Default: 0
        int m_smallRegionThreshold;

        // Retesselate regions that are in a regression plane. Implies --optimizePlanes. Default: false
        bool m_retesselate;

        // Threshold for fusing line segments while tesselating. Default: 0.01
        float m_lineFusionThreshold;


    };
} // namespace lvr2

#include "lvr2/reconstruction/LargeScaleReconstruction.tcc"

#endif // LAS_VEGAS_LARGESCALERECONSTRUCTION_HPP
