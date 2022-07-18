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
#include "lvr2/algorithm/ChunkHashGrid.hpp"
#include "lvr2/reconstruction/BigGrid.hpp"


#if defined CUDA_FOUND
    #define GPU_FOUND

    #include "lvr2/reconstruction/cuda/CudaSurface.hpp"

    typedef lvr2::CudaSurface GpuSurface;
#elif defined OPENCL_FOUND
    #define GPU_FOUND

    #include "lvr2/reconstruction/opencl/ClSurface.hpp"
    typedef lvr2::ClSurface GpuSurface;
#endif


namespace lvr2
{
    enum class LSROutput
    {
        /// Output one big Mesh. Uses A LOT of memory.
        BigMesh,
        /// Output one mesh per chunk as ply.
        ChunksPly,
        /// Output one h5 file containing one mesh per chunk.
        ChunksHdf5,

#ifdef LVR2_USE_3DTILES
        /// Output a 3D Tiles tileset
        Tiles3d,
#endif
    };

    struct LSROptions
    {
        /// what to produce as output
        std::unordered_set<LSROutput> output{LSROutput::BigMesh};
        bool hasOutput(LSROutput o) const
        {
            return output.find(o) != output.end();
        }

        /// Use GPU for normal computation.
        bool useGPU = false;

        /// Use GPU for signed distance computation.
        bool useGPUDistances = false;

        /// voxelsizes for reconstruction. Only the first one produces most types of output.
        std::vector<float> voxelSizes{0.1};

        /// chunk size for the BigGrid and VGrid. Has to be a multiple of every voxel size.
        float bgVoxelSize = 10;

        /// scale factor.
        float scale = 1;

        /// Max. Number of Points in a leaf when using kd-tree.
        uint nodeSize = 1000000;

        /// int flag to trigger partition-method (0 = kd-Tree; 1 = VGrid)
        uint partMethod = 1;

        /// Number of normals used in the normal interpolation process.
        uint ki = 20;

        /// Number of normals used for distance function evaluation. Has to be > 0.
        uint kd = 20;

        /// Size of k-neighborhood used for normal estimation. Has to be > 0.
        uint kn = 20;

        ///Set this flag for RANSAC based normal estimation.
        bool useRansac = false;

        /// FlipPoint for GPU normal computation
        std::vector<float> flipPoint{10000000, 10000000, 10000000};

        /// Extend the grid. Might avoid additional holes in sparse data sets, but can cause
        /// artifacts in dense data sets.
        bool extrude = true;

        /// number for the removal of dangling artifacts.
        uint removeDanglingArtifacts = 0;

        /// Remove noise artifacts from contours. Same values are between 2 and 4.
        uint cleanContours = 0;

        /// Maximum size for hole filling.
        uint fillHoles = 0;

        /// Shift all triangle vertices of a cluster onto their shared plane.
        bool optimizePlanes = false;

        /// (Plane Normal Threshold) Normal threshold for plane optimization.
        float planeNormalThreshold = 0.85;

        /// Number of iterations for plane optimization.
        uint planeIterations = 3;

        /// Minimum value for plane optimization.
        uint minPlaneSize = 7;

        /// Threshold for small region removal. If 0 nothing will be deleted.
        uint smallRegionThreshold = 0;

        /// Retesselate regions that are in a regression plane. Implies optimizePlanes.
        bool retesselate = false;

        /// Threshold for fusing line segments while tesselating.
        float lineFusionThreshold = 0.01;

        /// when generating LSROutput::Tiles3d, compress the meshes using Draco
        bool tiles3dCompress = false;

        /// Make sure all options are correctly set and consistent with each other.
        void ensureCorrectness()
        {
            LSROptions defaultOptions;

            #define CHECK_OPTION(option, condition, message) \
                if (condition) \
                { \
                    std::cout << timestamp << "Warning: " << message << " Reverting to default value." << std::endl; \
                    option = defaultOptions.option; \
                }

            CHECK_OPTION(output, output.empty(), "No output specified.");

#ifndef GPU_FOUND
            if (useGPU || useGPUDistances)
            {
                std::cout << timestamp << "Warning: No GPU found. Falling back to CPU." << std::endl;
                useGPU = useGPUDistances = false;
            }
#endif

            CHECK_OPTION(voxelSizes, voxelSizes.empty(), "No voxel sizes specified.");

            CHECK_OPTION(bgVoxelSize, bgVoxelSize <= 0, "bgVoxelSize has to be greater than 0.");

            auto correctedVoxelSize = std::ceil(bgVoxelSize / voxelSizes[0]) * voxelSizes[0];
            if (correctedVoxelSize != bgVoxelSize)
            {
                std::cout << timestamp << "Warning: bgVoxelSize is not a multiple of voxelSizes[0]. Correcting to " << correctedVoxelSize << std::endl;
                bgVoxelSize = correctedVoxelSize;
            }
            auto it = voxelSizes.begin();
            while (it != voxelSizes.end())
            {
                if (*it <= 0)
                {
                    std::cout << timestamp << "Warning: voxelSizes cannot be negative. Ignoring " << *it << std::endl;
                    voxelSizes.erase(it);
                }
                else if (bgVoxelSize != std::ceil(bgVoxelSize / *it) * *it)
                {
                    std::cout << timestamp << "Warning: all voxelSizes have to divide bgVoxelSize. Ignoring " << *it << std::endl;
                    voxelSizes.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            CHECK_OPTION(partMethod, partMethod > 1, "partMethod has to be 0 or 1.");

            CHECK_OPTION(kd, kd <= 0, "kd has to be greater than 0.");
            CHECK_OPTION(kn, kn <= 0, "kn has to be greater than 0.");

            CHECK_OPTION(flipPoint, flipPoint.size() != 3, "flipPoint has to be of size 3.");

            if (retesselate)
            {
                optimizePlanes = true;
            }

#ifdef LVR2_USE_3DTILES
            if (tiles3dCompress && !hasOutput(LSROutput::Tiles3d))
            {
                std::cout << timestamp << "Warning: tiles3dCompress is only supported for LSROutput::Tiles3d." << std::endl;
                tiles3dCompress = false;
            }
#else
            if (tiles3dCompress)
            {
                std::cout << timestamp << "Warning: tiles3dCompress is only supported when compiling with 3D Tiles support." << std::endl;
                tiles3dCompress = false;
            }
#endif
        }
    };

    template <typename BaseVecT>
    class LargeScaleReconstruction
    {


    public:
        /**
         * Constructor with parameters in a struct
         */
         LargeScaleReconstruction(LSROptions options = LSROptions());

        /**
         * splits the given PointClouds and calculates the reconstruction
         *
         * @param project ScanProject containing Scans
         * @param newChunksBB sets the Bounding Box of the reconstructed area
         * @param chunkManager an optional chunkManager to handle chunks. Only needed when using the VGrid partition method.
         * @return
         */
        void chunkAndReconstruct(ScanProjectEditMarkPtr project, BoundingBox<BaseVecT>& newChunksBB, std::shared_ptr<ChunkHashGrid> chunkManager = nullptr);

        /**
         *
         * this method splits the given PointClouds in to Chunks and calculates all required values for a later
         * reconstruction within a MPI network
         * @param project ScanProject containing Scans
         * @return
         */
        int trueMpiAndReconstructMaster(ScanProjectEditMarkPtr project, BoundingBox<BaseVecT>& newChunksBB, std::shared_ptr<ChunkHashGrid> chunkManager, int size);

        /**
         *
         * this methods splits the given PointClouds via kd-Tree and calculates all required values for a later
         * reconstruction within a MPI network
         * @param project ScanProject containing Scans
         * @return
         */
        int trueMpiAndReconstructSlave();

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
         * This Method Schedules the work for the MPI Clients
         */
        uint* mpiScheduler(std::shared_ptr<vector<BoundingBox<BaseVecT>>> partitionBoxes, BigGrid<BaseVecT>& bg, BoundingBox<BaseVecT>& cbb, std::shared_ptr<ChunkHashGrid> chunkManager);

        /**
         * This Method Collects results of MPI Clients and saves the completed reconstruction
         */
        void mpiCollector(std::shared_ptr<vector<BoundingBox<BaseVecT>>> partitionBoxes, BoundingBox<BaseVecT>& cbb, std::shared_ptr<ChunkHashGrid> chunkManager, uint* partitionBoxesSkipped);


        LSROptions m_options;
    };
} // namespace lvr2

#include "lvr2/reconstruction/LargeScaleReconstruction.tcc"

#endif // LAS_VEGAS_LARGESCALERECONSTRUCTION_HPP
