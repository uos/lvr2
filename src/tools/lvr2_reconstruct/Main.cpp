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


#include <iostream>
#include <memory>
#include <tuple>
#include <stdlib.h>

#include <boost/optional.hpp>
#include <boost/shared_array.hpp>
#include <boost/smart_ptr/make_shared_array.hpp>

#include "lvr2/config/lvropenmp.hpp"

#include "lvr2/geometry/PMPMesh.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/attrmaps/StableVector.hpp"
#include "lvr2/attrmaps/VectorMap.hpp"
#include "lvr2/algorithm/FinalizeAlgorithms.hpp"
#include "lvr2/algorithm/NormalAlgorithms.hpp"
#include "lvr2/algorithm/ColorAlgorithms.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/algorithm/Tesselator.hpp"
#include "lvr2/algorithm/ClusterPainter.hpp"
#include "lvr2/algorithm/ClusterAlgorithms.hpp"
#include "lvr2/algorithm/CleanupAlgorithms.hpp"
#include "lvr2/algorithm/ReductionAlgorithms.hpp"
#include "lvr2/algorithm/Materializer.hpp"
#include "lvr2/algorithm/Texturizer.hpp"
#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp" // Has to be included before anything includes opencv stuff, see https://github.com/flann-lib/flann/issues/214 
#include "lvr2/algorithm/SpectralTexturizer.hpp"

#ifdef LVR2_USE_EMBREE
    #include "lvr2/algorithm/RaycastingTexturizer.hpp"
#endif

#include "lvr2/reconstruction/BilinearFastBox.hpp"
#include "lvr2/reconstruction/TetraederBox.hpp"
#include "lvr2/reconstruction/FastReconstruction.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/SearchTree.hpp"
#include "lvr2/reconstruction/SearchTreeFlann.hpp"
#include "lvr2/reconstruction/SearchTreeLBVH.hpp"
#include "lvr2/reconstruction/HashGrid.hpp"
#include "lvr2/reconstruction/PointsetGrid.hpp"
#include "lvr2/reconstruction/SharpBox.hpp"
#include "lvr2/types/PointBuffer.hpp"
#include "lvr2/types/MeshBuffer.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/PlutoMapIO.hpp"
#include "lvr2/io/meshio/HDF5IO.hpp"
#include "lvr2/io/meshio/DirectoryIO.hpp"
#include "lvr2/util/Factories.hpp"
#include "lvr2/algorithm/GeometryAlgorithms.hpp"
#include "lvr2/algorithm/UtilAlgorithms.hpp"
#include "lvr2/algorithm/KDTree.hpp"
#include "lvr2/util/Logging.hpp"
#include "lvr2/util/ScanProjectUtils.hpp"

#include "lvr2/geometry/BVH.hpp"

#include "lvr2/reconstruction/DMCReconstruction.hpp"

#include "Options.hpp"

#if defined LVR2_USE_CUDA
    #define GPU_FOUND

    #include "lvr2/reconstruction/CudaKSearchSurface.hpp"
    #include "lvr2/reconstruction/cuda/CudaSurface.hpp"

    typedef lvr2::CudaSurface GpuSurface;
#elif defined LVR2_USE_OPENCL
    #define GPU_FOUND

    #include "lvr2/reconstruction/opencl/ClSurface.hpp"
    typedef lvr2::ClSurface GpuSurface;
#endif

using boost::optional;
using std::unique_ptr;
using std::make_unique;

using namespace lvr2;

using Vec = BaseVector<float>;
using PsSurface = lvr2::PointsetSurface<Vec>;

auto buildCombinedPointCloud(lvr2::ScanProjectPtr& project, lvr2::ReductionAlgorithmPtr reduction_algorithm) -> lvr2::PointBufferPtr
{
    // === Build the PointCloud ===
    lvr2::Monitor mon(lvr2::LogLevel::info, "[LVR2 Reconstruct] Loading scan positions", project->positions.size());

    size_t npoints_total = 0;
    // Count total number of points
    for (auto pos: project->positions)
    {
        for (auto lidar: pos->lidars)
        {
            for (auto scan: lidar->scans)
            {
                npoints_total += scan->numPoints;
            }
        }
    }

    // Allocate buffers for coordinates, colors and normals
    lvr2::floatArr coords(new float[npoints_total * 3]);
    lvr2::floatArr normals(new float[npoints_total * 3]);
    lvr2::ucharArr colors(new uchar[npoints_total * 3]);
    auto coord_output = coords.get();
    auto normal_output = normals.get();
    auto color_output = colors.get();

    bool has_normals = true;
    bool has_colors = true;
    int color_width = -1;

    lvr2::logout::get() << "[LVR2 Reconstruct] Total number of points: " << npoints_total << lvr2::endl;

    for (ScanPositionPtr pos: project->positions)
    {
        ++mon;
        for (LIDARPtr lidar: pos->lidars)
        {
            for (ScanPtr scan: lidar->scans)
            {
                // Load scan
                bool was_loaded = scan->loaded();
                if (!scan->loaded())
                {
                    scan->load(reduction_algorithm);
                }

                // Transform the new pointcloud
                transformPointCloud<float>(
                    std::make_shared<Model>(scan->points),
                    (pos->transformation * lidar->transformation * scan->transformation).cast<float>());
                
                // Copy coordinates
                coord_output = std::copy(
                    scan->points->getPointArray().get(),
                    scan->points->getPointArray().get() + (scan->points->numPoints() * 3),
                    coord_output
                );

                // Copy normals
                if (scan->points->hasNormals() && has_normals)
                {
                    normal_output = std::copy(
                        scan->points->getNormalArray().get(),
                        scan->points->getNormalArray().get() + (scan->points->numPoints() * 3),
                        normal_output
                    );
                }
                else
                {
                    has_normals = false;
                }

                // Copy colors
                if (scan->points->hasColors() && has_colors)
                {
                    size_t width;
                    scan->points->getColorArray(width);
                    if (width == color_width || color_width == -1)
                    {
                        color_width = width;
                        color_output = std::copy(
                            scan->points->getColorArray(width).get(),
                            scan->points->getColorArray(width).get() + (scan->points->numPoints() * width),
                            color_output
                        );
                    }
                    else
                    {
                        has_colors = false;
                    }

                }
                else
                {
                    has_colors = false;
                }
                
                // If not previously loaded unload
                if (!was_loaded)
                {
                    scan->release();
                }
            }
        }
    }

    // Create new point buffer
    auto retval = std::make_shared<PointBuffer>(coords, npoints_total);
    // Add normals
    if (has_normals)
    {
        retval->setNormalArray(normals, npoints_total);
    }
    // Add colors
    if (has_colors)
    {
        retval->setColorArray(colors, npoints_total, color_width);
    }

    return retval;
}

template <typename BaseVecT>
PointsetSurfacePtr<BaseVecT> loadPointCloud(const reconstruct::Options& options)
{   

    // Create a point loader object
    ModelPtr model = ModelFactory::readModel(options.getInputFileName());
    PointBufferPtr buffer;
    // Parse loaded data
    if (!model)
    {
        ReductionAlgorithmPtr reduction_algorithm;
        // If the user supplied valid octree reduction parameters use octree reduction otherwise use no reduction
        if (options.getOctreeVoxelSize() > 0.0f)
        {
            reduction_algorithm = std::make_shared<OctreeReductionAlgorithm>(options.getOctreeVoxelSize(), options.getOctreeMinPoints());
        }
        else
        {
            reduction_algorithm = std::make_shared<NoReductionAlgorithm>();
        }
        
        lvr2::ScanProjectPtr project;
        // Vector of the ScanPositions to load
        std::vector<lvr2::ScanPositionPtr> positions;
        if (options.hasScanPositionIndex())
        {
            project = lvr2::loadScanPositionsExplicitly(
                options.getInputSchema(),
                options.getInputFileName(),
                options.getScanPositionIndex());
        }
        else
        {    
            project = lvr2::loadScanProject(options.getInputSchema(), options.getInputFileName());
        }
        
        buffer = buildCombinedPointCloud(project, reduction_algorithm);
    }
    else 
    {
        buffer = model->m_pointCloud;
    }

    // Create a point cloud manager
    string pcm_name = options.getPCM();
    PointsetSurfacePtr<Vec> surface;

    // Create point set surface object
    if(pcm_name == "PCL")
    {
        lvr2::logout::get() << lvr2::error << "[LVR2 Reconstruct] Using PCL as point cloud manager is not implemented yet!" << lvr2::endl;
        panic_unimplemented("PCL as point cloud manager");
    }
    else if(pcm_name == "FLANN" || pcm_name == "NABO" || pcm_name == "NANOFLANN" || pcm_name == "LVR2")
    {
        
        int plane_fit_method = options.getNormalEstimation();

        // plane_fit_method
        // - 0: PCA
        // - 1: RANSAC
        // - 2: Iterative
        surface = std::make_shared<AdaptiveKSearchSurface<BaseVecT>>(
            buffer,
            pcm_name,
            options.getKn(),
            options.getKi(),
            options.getKd(),
            plane_fit_method,
            options.getScanPoseFile()
        );
    }
    else if(pcm_name == "LBVH_CUDA")
    {
        #ifdef LVR2_USE_CUDA
            surface = std::make_shared<CudaKSearchSurface<BaseVecT>>(
                buffer,
                options.getKn()
            );
        #else
            lvr2::logout::get() << lvr2::error << "[LVR2 Reconstruct] ERROR: Cuda not found. Do not use LBVH_CUDA." << lvr2::endl;
            return nullptr;
        #endif
    }
    else
    {
        lvr2::logout::get() << lvr2::error << "[LVR2 Reconstruct] Unable to create PointCloudManager." << lvr2::endl;
        lvr2::logout::get() << lvr2::error << "[LVR2 Reconstruct] Unknown option '" << pcm_name << "'." << lvr2::endl;
        return nullptr;
    }

    // Set search options for normal estimation and distance evaluation
    surface->setKd(options.getKd());
    surface->setKi(options.getKi());
    surface->setKn(options.getKn());

    auto flipPointOptional = options.getFlippoint();
    BaseVector<float> flipPoint(0.0f, 0.0f, 0.0f);

    if(flipPointOptional)
    {
        vector<float> v = *flipPointOptional;
        flipPoint[0] = v[0];
        flipPoint[1] = v[1];
        flipPoint[2] = v[2];
        lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Flip point for normal estimation set to : " << flipPoint << lvr2::endl;
    }
    else
    {
         lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] No flip point set, defaulting to (0,0,0) " <<  lvr2::endl;
    }
    surface->setFlipPoint(flipPoint);
    

    // Calculate normals if necessary
    if(!buffer->hasNormals() || options.recalcNormals())
    {
        if(options.useGPU())
        {
            #ifdef GPU_FOUND
                size_t num_points = buffer->numPoints();
                floatArr points = buffer->getPointArray();
                floatArr normals = floatArr(new float[ num_points * 3 ]);
                lvr2::logout::get() << lvr2::info << "Generating GPU kd-tree" << lvr2::endl;
                GpuSurface gpu_surface(points, num_points);
                

                gpu_surface.setKn(options.getKn());
                gpu_surface.setKi(options.getKi());
                gpu_surface.setFlippoint(flipPoint[0], flipPoint[1], flipPoint[2]);

                lvr2::logout::get() << lvr2::info << "Estimating Normals GPU" << lvr2::endl;
                gpu_surface.calculateNormals();
                gpu_surface.getNormals(normals);

                buffer->setNormalArray(normals, num_points);
                gpu_surface.freeGPU();
            #else
                lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] ERROR: GPU Driver not installed" << lvr2::endl;
                surface->calculateSurfaceNormals();
            #endif
        }
        else
        {
            surface->calculateSurfaceNormals();
        }
    }
    else
    {
        lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Using given normals." << lvr2::endl;
    }
    if(pcm_name == "LBVH_CUDA")
    {
        surface = std::make_shared<AdaptiveKSearchSurface<BaseVecT>>(
            buffer,
            "FLANN",
            options.getKn(),
            options.getKi(),
            options.getKd(),
            0,
            options.getScanPoseFile()
        );
    }
    return surface;
}

std::pair<shared_ptr<GridBase>, unique_ptr<FastReconstructionBase<Vec>>>
    createGridAndReconstruction(
        const reconstruct::Options& options,
        PointsetSurfacePtr<Vec> surface
    )
{
    // Determine whether to use intersections or voxelsize
    bool useVoxelsize = options.getIntersections() <= 0;
    float resolution = useVoxelsize ? options.getVoxelsize() : options.getIntersections();

    // Create a point set grid for reconstruction
    std::string decompositionType = options.getDecomposition();

    // Fail safe check
    if(decompositionType != "MT" && decompositionType != "MC" && decompositionType != "DMC" && decompositionType != "PMC" && decompositionType != "SF" )
    {
        lvr2::logout::get() << lvr2::warning << "[LVR2 Reconstruct] Unsupported decomposition type " << decompositionType << ". Defaulting to PMC." << lvr2::endl;
        decompositionType = "PMC";
    }

    if(decompositionType == "MC")
    {
        auto grid = std::make_shared<PointsetGrid<Vec, FastBox<Vec>>>(
            resolution,
            surface,
            surface->getBoundingBox(),
            useVoxelsize,
            options.extrude()
        );

        grid->calcDistanceValues();
        lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Grid Cells: " << grid->getCells().size() << lvr2::endl;
        auto reconstruction = std::make_unique<FastReconstruction<Vec, FastBox<Vec>>>(grid);
        return std::make_pair(grid, std::move(reconstruction));
    }
    else if(decompositionType == "PMC")
    {
        BilinearFastBox<Vec>::m_surface = surface;
        auto grid = std::make_shared<PointsetGrid<Vec, BilinearFastBox<Vec>>>(
            resolution,
            surface,
            surface->getBoundingBox(),
            useVoxelsize,
            options.extrude()
        );
        grid->calcDistanceValues();
        lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Grid Cells: " << grid->getCells().size() << lvr2::endl;
        auto reconstruction = std::make_unique<FastReconstruction<Vec, BilinearFastBox<Vec>>>(grid);
        return std::make_pair(grid, std::move(reconstruction));
    }
    // else if(decompositionType == "DMC")
    // {
    //     auto reconstruction = make_unique<DMCReconstruction<Vec, FastBox<Vec>>>(
    //         surface,
    //         surface->getBoundingBox(),
    //         options.extrude()
    //     );
    //     return make_pair(nullptr, std::move(reconstruction));
    // }
    else if(decompositionType == "MT")
    {
        auto grid = std::make_shared<PointsetGrid<Vec, TetraederBox<Vec>>>(
            resolution,
            surface,
            surface->getBoundingBox(),
            useVoxelsize,
            options.extrude()
        );
        grid->calcDistanceValues();
        auto reconstruction = make_unique<FastReconstruction<Vec, TetraederBox<Vec>>>(grid);
        return make_pair(grid, std::move(reconstruction));
    }
    else if(decompositionType == "SF")
    {
        SharpBox<Vec>::m_surface = surface;
        auto grid = std::make_shared<PointsetGrid<Vec, SharpBox<Vec>>>(
            resolution,
            surface,
            surface->getBoundingBox(),
            useVoxelsize,
            options.extrude()
        );
        grid->calcDistanceValues();
        auto reconstruction = make_unique<FastReconstruction<Vec, SharpBox<Vec>>>(grid);
        return make_pair(grid, std::move(reconstruction));
    }

    lvr2::logout::get() << lvr2::warning << "[LVR2 Reconstruct] Unsupported decomposition type " << decompositionType << "." << lvr2::endl;
    return make_pair(nullptr, nullptr);
}

template <typename Vec>
void addSpectralTexturizers(const reconstruct::Options& options, lvr2::Materializer<Vec>& materializer)
{
    if (!options.hasScanPositionIndex())
    {
        return;
    }

    if(options.getScanPositionIndex().size() > 1)
    {
        lvr2::logout::get() << lvr2::warning 
            << "[LVR2 Reconstruct] Warning: Spectral texturizing only supports one scan position. Ignoring all but the first." 
            << lvr2::endl;
    }

    // load panorama from hdf5 file
    auto project = lvr2::loadScanPositionsExplicitly(
        options.getInputSchema(),
        options.getInputFileName(),
        options.getScanPositionIndex()
    );

    // If there is no spectral data
    if (project->positions[0]->hyperspectral_cameras.empty()
     || project->positions[0]->hyperspectral_cameras[0]->panoramas.empty())
    {
        return;
    }

    auto panorama = project->positions[0]->hyperspectral_cameras[0]->panoramas[0];

    int texturizer_count = options.getMaxSpectralChannel() - options.getMinSpectralChannel();
    texturizer_count = std::max(texturizer_count, 0);

    // go through all spectralTexturizers of the vector
    for(int i = 0; i < texturizer_count; i++)
    {
        // if the spectralChannel doesnt exist, skip it
        if(panorama->num_channels < options.getMinSpectralChannel() + i)
        {
            continue;
        }

        auto spec_text = std::make_shared<SpectralTexturizer<Vec>>(
            options.getTexelSize(),
            options.getTexMinClusterSize(),
            options.getTexMaxClusterSize()
        );

        // set the spectral texturizer with the current spectral channel
        spec_text->init_image_data(panorama, std::max(options.getMinSpectralChannel(), 0) + i);
        // set the texturizer for the current spectral channel
        materializer.addTexturizer(spec_text);
    }
}

template <typename MeshVec, typename ClusterVec>
void addRaycastingTexturizer(const reconstruct::Options& options, lvr2::Materializer<Vec>& materializer, const BaseMesh<MeshVec>& mesh, const ClusterBiMap<ClusterVec> clusters)
{
#ifdef LVR2_USE_EMBREE

    ScanProjectPtr project;
    if (options.hasScanPositionIndex())
    {
        project = lvr2::loadScanPositionsExplicitly(
            options.getInputSchema(),
            options.getInputFileName(),
            options.getScanPositionIndex());
    }
    else
    {
        project = lvr2::loadScanProject(
            options.getInputSchema(),
            options.getInputFileName());
    }

    auto texturizer = std::make_shared<RaycastingTexturizer<Vec>>(
        options.getTexelSize(),
        options.getTexMinClusterSize(),
        options.getTexMaxClusterSize(),
        mesh,
        clusters,
        project
    );

    materializer.addTexturizer(texturizer);
#else
    lvr2::logout::get() << lvr2::warning << "[LVR2 Reconstruct] This software was compiled without support for Embree!\n";
    lvr2::logout::get() << lvr2::warning << "[LVR2 Reconstruct] The RaycastingTexturizer needs the Embree library." << lvr2::endl;
#endif
}

template <typename BaseMeshT, typename BaseVecT>
BaseMeshT reconstructMesh(reconstruct::Options options, PointsetSurfacePtr<BaseVecT> surface)
{
    // =======================================================================
    // Reconstruct mesh from point cloud data
    // =======================================================================
    // Create an empty mesh
    BaseMeshT mesh;

    shared_ptr<GridBase> grid;
    unique_ptr<FastReconstructionBase<Vec>> reconstruction;
    std::tie(grid, reconstruction) = createGridAndReconstruction(options, surface);

    // Reconstruct mesh
    reconstruction->getMesh(mesh);

    // Save grid to file
    if(options.saveGrid() && grid)
    {
        grid->saveGrid("fastgrid.grid");
    }

    return std::move(mesh);
}

template <typename BaseMeshT>
void optimizeMesh(reconstruct::Options options, BaseMeshT& mesh)
{
    // =======================================================================
    // Optimize mesh
    // =======================================================================
    if(options.getDanglingArtifacts())
    {
        lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Removing dangling artifacts" << lvr2::endl;
        removeDanglingCluster(mesh, static_cast<size_t>(options.getDanglingArtifacts()));
    }

    cleanContours(mesh, options.getCleanContourIterations(), 0.0001);

    if(options.getFillHoles())
    {
        mesh.fillHoles(options.getFillHoles());
    }

    // Reduce mesh complexity
    const auto reductionRatio = options.getEdgeCollapseReductionRatio();
    if (reductionRatio > 0.0)
    {
        if (reductionRatio > 1.0)
        {
            throw "[LVR2 Reconstruct] The reduction ratio has to be between 0 and 1";
        }

        size_t old = mesh.numVertices();
        size_t target = old * (1.0 - reductionRatio);
        lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Trying to remove " << old - target << " / " << old << " vertices." << lvr2::endl;
        mesh.simplify(target);
        lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Removed " << old - mesh.numVertices() << " vertices." << lvr2::endl;
    }

    auto faceNormals = calcFaceNormals(mesh);


    if (options.optimizePlanes())
    {
        ClusterBiMap<FaceHandle> clusterBiMap = iterativePlanarClusterGrowingRANSAC(
            mesh,
            faceNormals,
            options.getNormalThreshold(),
            options.getPlaneIterations(),
            options.getMinPlaneSize()
        );

        if(options.getSmallRegionThreshold() > 0)
        {
            deleteSmallPlanarCluster(mesh, clusterBiMap, static_cast<size_t>(options.getSmallRegionThreshold()));
        }

        cleanContours(mesh, options.getCleanContourIterations(), 0.0001);

        if(options.getFillHoles())
        {
            mesh.fillHoles(options.getFillHoles());
        }
    
        // Recalculate the face normals because the faces were modified previously
        faceNormals = calcFaceNormals(mesh);
        // Regrow clusters after hole filling and small region removal
        clusterBiMap = planarClusterGrowing(mesh, faceNormals, options.getNormalThreshold());

        if (options.retesselate())
        {
            Tesselator<Vec>::apply(mesh, clusterBiMap, faceNormals, options.getLineFusionThreshold());
        }
    }

}

template <typename BaseVecT>
struct cmpBaseVecT
{
    bool operator()(const BaseVecT& lhs, const BaseVecT& rhs) const
    {
        return (lhs[0] < rhs[0])
            || (lhs[0] == rhs[0] && lhs[1] < rhs[1])
            || (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] < rhs[2]);

    }
};

template <typename BaseMeshT, typename BaseVecT>
auto loadExistingMesh(reconstruct::Options options)
{
    meshio::HDF5IO io(
        std::make_shared<HDF5Kernel>(options.getInputMeshFile()),
        std::make_shared<MeshSchemaHDF5>()
    );
    MeshBufferPtr mesh_buffer = io.loadMesh(options.getInputMeshName());


    // Handle Maps needed during mesh construction
    std::map<size_t, VertexHandle> indexToVertexHandle;
    std::map<BaseVecT, VertexHandle, cmpBaseVecT<BaseVecT>> positionToVertexHandle;
    std::map<size_t, FaceHandle> indexToFaceHandle;
    // Create all this stuff manually instead of using the constructors to
    // ensure the Handles are correct.
    BaseMeshT mesh;
    DenseFaceMap<Normal<float>> faceNormalMap;
    ClusterBiMap<FaceHandle> clusterBiMap;

    // Add vertices
    floatArr vertices = mesh_buffer->getVertices();
    for (size_t i = 0; i < mesh_buffer->numVertices(); i++)
    {
        BaseVecT vertex_pos(
            vertices[i * 3 + 0],
            vertices[i * 3 + 1],
            vertices[i * 3 + 2]);

        // If the vertex position already exists do not add new vertex
        if (positionToVertexHandle.count(vertex_pos))
        {
            VertexHandle vertexH = positionToVertexHandle.at(vertex_pos);
            indexToVertexHandle.insert(std::pair(i, vertexH));
        }
        else
        {
            VertexHandle vertexH = mesh.addVertex(vertex_pos);
            indexToVertexHandle.insert(std::pair(i, vertexH));
        }
    }

    // Add faces
    indexArray faces = mesh_buffer->getFaceIndices();
    floatArr faceNormals = mesh_buffer->getFaceNormals();
    for (size_t i = 0; i < mesh_buffer->numFaces(); i++)
    {
        VertexHandle v0 = indexToVertexHandle.at(faces[i * 3 + 0]);
        VertexHandle v1 = indexToVertexHandle.at(faces[i * 3 + 1]);
        VertexHandle v2 = indexToVertexHandle.at(faces[i * 3 + 2]);
        // Add face
        FaceHandle faceH = mesh.addFace(v0, v1, v2);
        indexToFaceHandle.insert(std::pair(i, faceH));

        if (faceNormals)
        {
            // Add normal
            Normal<float> normal(
                faceNormals[i * 3 + 0],
                faceNormals[i * 3 + 1],
                faceNormals[i * 3 + 2]
            );
            
            faceNormalMap.insert(faceH, normal);
        }
        
    }

    if (!faceNormals)
    {
        lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Calculating face normals" << lvr2::endl;
        faceNormalMap = calcFaceNormals(mesh);
    }

    // Add clusters
    for (size_t i = 0;; i++)
    {
        std::string clusterString =  "cluster" + std::to_string(i) + "_face_indices";
        auto clusterIndicesOptional = mesh_buffer->getIndexChannel(clusterString);
        // If the cluster does not exist break
        if (!clusterIndicesOptional) break;

        ClusterHandle clusterH = clusterBiMap.createCluster();
        for (size_t j = 0; j < clusterIndicesOptional->numElements(); j++)
        {
            FaceHandle faceH = indexToFaceHandle.at(j);
            clusterBiMap.addToCluster(clusterH, faceH);
        }
    }

    // Load the pointcloud
    PointsetSurfacePtr<Vec> surface = loadPointCloud<BaseVecT>(options);

    return std::make_tuple(std::move(mesh), std::move(surface), std::move(faceNormalMap), std::move(clusterBiMap));
}

int main(int argc, char** argv)
{
    // =======================================================================
    // Parse and print command line parameters
    // =======================================================================
    // Parse command line arguments
    reconstruct::Options options(argc, argv);

    options.printLogo();

    // Exit if options had to generate a usage message
    // (this means required parameters are missing)
    if (options.printUsage())
    {
        return EXIT_SUCCESS;
    }

    std::cout << options << std::endl;

    // =======================================================================
    // Load (and potentially store) point cloud
    // =======================================================================
    OpenMPConfig::setNumThreads(options.getNumThreads());

    lvr2::PMPMesh<Vec> mesh;
    PointsetSurfacePtr<Vec> surface;
    DenseFaceMap<Normal<float>> faceNormals;
    ClusterBiMap<FaceHandle> clusterBiMap;

    if (options.useExistingMesh())
    {
        lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Loading existing mesh '" << options.getInputMeshName() << "' from file '" << options.getInputMeshFile() << "'" << lvr2::endl;
        std::tie(mesh, surface, faceNormals, clusterBiMap) = loadExistingMesh<lvr2::PMPMesh<Vec>, Vec>(options);
    }
    else
    {
        // Load PointCloud
        surface = loadPointCloud<Vec>(options);
        if (!surface)
        {
            lvr2::logout::get() << lvr2::error << "[LVR2 Reconstruct] Failed to create pointcloud. Exiting." << lvr2::endl;
            exit(EXIT_FAILURE);
        }
        
        lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Pointcloud loaded starting to reconstruct surfaces ..." << lvr2::endl;

        // Reconstruct simple mesh
        mesh = reconstructMesh<lvr2::PMPMesh<Vec>>(options, surface);
        lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Reconstructed mesh (vertices, faces): " << mesh.numVertices() << ", " << mesh.numFaces() << ")" << lvr2::endl;
    }

    // Save points and normals only
    if(options.savePointNormals())
    {
        ModelPtr pn(new Model(surface->pointBuffer()));
        ModelFactory::saveModel(pn, "pointnormals.ply");
    }

    // Optimize the mesh if requested
    optimizeMesh(options, mesh);
    

    // Calc normals and clusters
    faceNormals = calcFaceNormals(mesh);
    clusterBiMap = planarClusterGrowing(mesh, faceNormals, options.getNormalThreshold());

    // =======================================================================
    // Finalize mesh
    // =======================================================================
    // Prepare color data for finalizing

    ColorGradient::GradientType t = ColorGradient::gradientFromString(options.getClassifier());

    ClusterPainter painter(clusterBiMap);
    auto clusterColors = boost::optional<DenseClusterMap<RGB8Color>>(painter.colorize(mesh, t));
    auto vertexColors = calcColorFromPointCloud(mesh, surface);

    // Calc normals for vertices
    auto vertexNormals = calcVertexNormals(mesh, faceNormals, *surface);

    // Prepare finalize algorithm
    TextureFinalizer<Vec> finalize(clusterBiMap);
    finalize.setVertexNormals(vertexNormals);

    // Vertex colors:
    // If vertex colors should be generated from pointcloud:
    if (options.vertexColorsFromPointcloud())
    {
        // set vertex color data from pointcloud
        finalize.setVertexColors(*vertexColors);
    }
    else if (clusterColors)
    {
        // else: use simpsons painter for vertex coloring
        finalize.setClusterColors(*clusterColors);
    }

    // Materializer for face materials (colors and/or textures)
    Materializer<Vec> materializer(
        mesh,
        clusterBiMap,
        faceNormals,
        *surface
    );

    auto texturizer = std::make_shared<Texturizer<Vec>>(
        options.getTexelSize(),
        options.getTexMinClusterSize(),
        options.getTexMaxClusterSize()
    );




    // When using textures ...
    if (options.generateTextures())
    {
        addSpectralTexturizers(options, materializer);

#ifdef LVR2_USE_EMBREE
        if (options.useRaycastingTexturizer())
        {
            addRaycastingTexturizer(options, materializer, mesh, clusterBiMap);
        }
        else
        {
            materializer.addTexturizer(texturizer);
        }
#else
        materializer.addTexturizer(texturizer);
#endif
            
    }

    // Generate materials
    MaterializerResult<Vec> matResult = materializer.generateMaterials();

    // Add material data to finalize algorithm
    finalize.setMaterializerResult(matResult);
    
    // Run finalize algorithm
    auto buffer = finalize.apply(mesh);

    // When using textures ...
    if (options.generateTextures())
    {
        // Set optioins to save them to disk
        //materializer.saveTextures();
        buffer->addIntAtomic(1, "mesh_save_textures");
        buffer->addIntAtomic(1, "mesh_texture_image_extension");
    }

    // =======================================================================
    // Write all results (including the mesh) to file
    // =======================================================================
    // Create output model and save to file
    auto m = ModelPtr( new Model(buffer));

    if(options.saveOriginalData())
    {
        m->m_pointCloud = surface->pointBuffer();
    }

    for(const std::string& output_filename : options.getOutputFileNames())
    {
        boost::filesystem::path outputDir(options.getOutputDirectory());
        boost::filesystem::path selectedFile( output_filename );
        boost::filesystem::path outputFile = outputDir/selectedFile;
        std::string extension = selectedFile.extension().string();

        lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Saving mesh to "<< output_filename << "." << lvr2::endl;

        if (extension == ".h5")
        {

            HDF5KernelPtr kernel = HDF5KernelPtr(new HDF5Kernel(outputFile.string()));
            MeshSchemaHDF5Ptr schema = MeshSchemaHDF5Ptr(new MeshSchemaHDF5());
            auto mesh_io = meshio::HDF5IO(kernel, schema);

            mesh_io.saveMesh(
                options.getMeshName(),
                buffer
                );

            continue;
        }

        if (extension == "")
        {
            DirectoryKernelPtr kernel = DirectoryKernelPtr(new DirectoryKernel(outputFile.string()));
            MeshSchemaDirectoryPtr schema = MeshSchemaDirectoryPtr(new MeshSchemaDirectory());
            auto mesh_io = meshio::DirectoryIO(kernel, schema);

            mesh_io.saveMesh(
                options.getMeshName(),
                buffer
                );

            continue;
        }

        ModelFactory::saveModel(m, outputFile.string());
    }

    if (matResult.m_keypoints)
    {
        // save materializer keypoints to hdf5 which is not possible with ModelFactory
        //PlutoMapIO map_io("triangle_mesh.h5");
        //map_io.addTextureKeypointsMap(matResult.m_keypoints.get());
    }

    lvr2::logout::get() << lvr2::info << "[LVR2 Reconstruct] Program end." << lvr2::endl;

    return 0;
}
