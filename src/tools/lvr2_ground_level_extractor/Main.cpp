/*
 * Main.cpp
 *
 *  Created on: 18.11.2020
 *      Author: Mario Dyczka (mdyczka@uos.de)
 */

#include "Main.hpp"
#include "Options.hpp"

using boost::optional;
using std::unique_ptr;
using std::make_unique;

using namespace lvr2;

using Vec = BaseVector<float>;
using VecD = BaseVector<double>;

/**
 * @brief Loads the point cloud data and creates an adaptiveKSearchSuface.
 * 
 * @tparam BaseVecT Sets which BaseVector template is used.
 * @param data Contains the path of the point cloud.
 * @return PointsetSurfacePtr<BaseVecT> Returns a point cloud manager for @data.
 */
template <typename BaseVecT>
PointsetSurfacePtr<BaseVecT> loadPointCloud(string data)
{
    ModelPtr baseModel = ModelFactory::readModel(data);
    if (!baseModel)
    {
        std::cout << timestamp.getElapsedTime() << "IO Error: Unable to parse " << data << std::endl;
        return nullptr;
    }

    PointBufferPtr baseBuffer = baseModel->m_pointCloud;
    PointsetSurfacePtr<BaseVecT> surface;
    surface = std::make_shared<AdaptiveKSearchSurface<BaseVecT>>(baseBuffer,"FLANN");
    surface->calculateSurfaceNormals();
    return surface;
}

/**
 * @brief Creates textures for a provided @mesh and calculates the texture coordinates. A texture is either generated from the @mesh compared to the point cloud it is
 * based on or a GeoTIFF. It @returns a format that can be used in the TextureFinalizer.
 * 
 * @tparam BaseVecT Sets which BaseVector template is used.
 * @param mesh Model the texture is projected onto.
 * @param clusters Cluster containing all faces of the model.
 * @param surface Point cloud manager that calculates the bounding box.
 * @param texelSize Texel size of the generated texture. Also sets its resolution.
 * @param affineMatrix If the mesh was transformed or a GeoTIFF is used as texture, the transformation matrix needs to be provided. Transformation matrix without Translation.
 * @param fullAffineMatrix Transformation matrix with Translation.
 * @param io If provided, contains the data of a GeoTIFF.
 * @param tree Search Tree that utilises the FLANN to enable Radius and Nearest Neighbor Search on the point clouds data. 
 * @param startingBand First Band to extract from the GeoTIFF.
 * @param numberOfBands Number of Bands to extract from the GeoTIFF.
 * @param colorScale The colour scale that is used in the generated texture: GREY, JET, HOT, HSV, SHSV, SIMPSONS.
 * @param noTransformation If the transformation matrix was not applied to the @mesh, this should be set to true.
 * @return MaterializerResult<BaseVecT> Return a structure that contains the texture and the texture coordinates. It can be used in conjunction with the TextureFinlaizer 
 * to create a textured OBJ.
 */
template<typename BaseVecT>
MaterializerResult<BaseVecT> projectTexture(const lvr2::HalfEdgeMesh<BaseVecT>& mesh, const ClusterBiMap<FaceHandle>& clusters, const PointsetSurface<Vec>& surface, 
float texelSize, Eigen::MatrixXd affineMatrix, Eigen::MatrixXd fullAffineMatrix, GeoTIFFIO* io,SearchTreeFlann<BaseVecT>& tree, int startingBand, int numberOfBands,
string colorScale, bool noTransformation)
{
    // =======================================================================
    // Prepare necessary preconditions to create MaterializerResult
    // =======================================================================
    DenseClusterMap<Material> clusterMaterials;
    SparseVertexMap<ClusterTexCoordMapping> vertexTexCoords;
    
    // The keypoint_map is never utilised in the finalizer and is ignored henceforth
    std::unordered_map<BaseVecT, std::vector<float>> keypoints_map;
    StableVector<TextureHandle, Texture> textures;

    for (auto clusterH : clusters)
    {   
        // =======================================================================
        // Generate Bounding Box for Texture coordinate calculations
        // =======================================================================
        const Cluster<FaceHandle>& cluster = clusters.getCluster(clusterH);

        auto bb = surface.getBoundingBox();
        auto min = bb.getMin();
        auto max = bb.getMax();

        ssize_t xMax = (ssize_t)(std::round(max.x));
        ssize_t xMin = (ssize_t)(std::round(min.x));
        ssize_t yMax = (ssize_t)(std::round(max.y));
        ssize_t yMin = (ssize_t)(std::round(min.y));
        ssize_t zMax = (ssize_t)(std::round(max.z));
        ssize_t zMin = (ssize_t)(std::round(min.z));

        Texture tex;
        // If a GeoTIFF was read --> Extract its Bands
        // Else, create a height difference texture
        
        if(io)
        {           
            tex = readGeoTIFF(io,startingBand,startingBand + numberOfBands -1, colorScale);           
        }
        else
        {
            tex = generateHeightDifferenceTexture<VecD,double>(surface,tree,mesh,texelSize,affineMatrix,colorScale);
        }     

        // Rotates the extreme Values to fit the Texture
        if(affineMatrix.size() != 0)
        {
            Eigen::Vector4d pointMax(xMax,yMax,zMax,1);
            Eigen::Vector4d pointMin(xMin,yMin,zMin,1);

            Eigen::Vector4d solution;
            solution = affineMatrix*pointMax;
            auto xOldMax = solution.coeff(0);
            auto yOldMax = solution.coeff(1);

            solution = affineMatrix*pointMin;
            auto xOldMin = solution.coeff(0);
            auto yOldMin = solution.coeff(1); 

            if(xOldMin < xOldMax)
            {
                xMax = xOldMax;
                xMin = xOldMin;
            }
            else
            {
                xMax = xOldMin;
                xMin = xOldMax;
            }

            if(yOldMin < yOldMax)
            {
                yMax = yOldMax;
                yMin = yOldMin;
            }
            else
            {
                yMax = yOldMin;
                yMin = yOldMax;
            }                
        }

        ssize_t xDim = (abs(xMax) + abs(xMin));
        ssize_t yDim = (abs(yMax) + abs(yMin));   
        BaseVecT correct(xMin,yMin,0); 

        // Code copied from Materializer.tcc; this part essentially does what the materializer does
        // save Texture as Material so it can be correctly generated by the finalizer
        Material material;
        material.m_texture = textures.push(tex);
        
        std::array<unsigned char, 3> arr = {255, 255, 255};
        
        material.m_color = std::move(arr);            
        clusterMaterials.insert(clusterH, material);

        std::unordered_set<VertexHandle> clusterVertices;
        // Get a set of all unique vertices
        for (auto faceH : cluster.handles)
        {
            for (auto vertexH : mesh.getVerticesOfFace(faceH))
            {
                clusterVertices.insert(vertexH);
            }
        }        

        // Calculate the Texture Coordinates for all Vertices
        for (auto vertexH : clusterVertices)
        {            
            auto pos = mesh.getVertexPosition(vertexH);

            // Correct coordinates            
            float yPixel = 0;
            float xPixel = 0;
            
            if(io)
            {
                // Calculate Texture Coordinates based on GeoTIFF Data
                double geoTransform[6];
                int y_dim_tiff = io->getRasterHeight();
                int x_dim_tiff = io->getRasterWidth();
                float values[2];
                io->getMaxMinOfBand(values,1);
                io->getGeoTransform(geoTransform);   
                
                // To correctly depict the GeoTIFFs data we need the referenced coordinates
                // Even if they were not applied in the model generation process
                if(noTransformation)
                {
                    Eigen::Vector4d point(pos[0],pos[1],0,1);

                    Eigen::Vector4d solution;
                    solution = fullAffineMatrix*point;
                    pos[0] = solution.coeff(0);
                    pos[1] = solution.coeff(1);
                }
                else
                {
                    pos[0] = pos[0] + fullAffineMatrix(12);
                    pos[1] = pos[1] + fullAffineMatrix(13);
                } 

                xPixel = (pos[0] - geoTransform[0])/geoTransform[1];
                xPixel /= x_dim_tiff;
                yPixel = (pos[1] - geoTransform[3])/geoTransform[5];
                yPixel /= y_dim_tiff;
            }
            else
            {
                // Calculate Texture Coordinates based on normalised Model coordinates
                pos = pos - correct;
                xPixel = pos[0]/xDim;
                yPixel = 1 - pos[1]/yDim;
            }         
            
            TexCoords texCoords(xPixel,yPixel);
            if (vertexTexCoords.get(vertexH))
            {
                vertexTexCoords.get(vertexH).get().push(clusterH, texCoords);
            }
            else
            {
                ClusterTexCoordMapping mapping;
                mapping.push(clusterH, texCoords);
                vertexTexCoords.insert(vertexH, mapping);
            }
        }
    }
    
    return MaterializerResult<BaseVecT>(
        clusterMaterials,
        textures,
        vertexTexCoords,
        keypoints_map
    );

}

int main(int argc, char* argv[])
{  
    // =======================================================================
    // Parse command line parameters
    // =======================================================================
    std::cout << std::fixed;

    ground_level_extractor::Options options(argc,argv);
    options.printLogo();

    if (options.printUsage())
    {
        return 0;
    }
    std::cout << options << std::endl; 

    // =======================================================================
    // Load Pointcloud and create Model + Surface + SearchTree
    // =======================================================================
    lvr2::HalfEdgeMesh<VecD> mesh;
    auto surface = loadPointCloud<Vec>(options.getInputFileName());   
    if(surface == nullptr)
    {
        std::cout << timestamp.getElapsedTime() << "IO Error: Unable to interpret " << options.getInputFileName() << std::endl;
        return 0;
    } 
    PointBufferPtr baseBuffer = surface->pointBuffer();
    auto tree = SearchTreeFlann<VecD> (baseBuffer);

    // Get the pointcloud coordinates from the FloatChannel
    FloatChannel arr =  *(baseBuffer->getFloatChannel("points"));   
    PointsetSurfacePtr<Vec> usedSurface = surface;
    FloatChannel usedArr = arr;       
    float resolution = options.getResolution();
    float texelSize = resolution/2; 

    // Read what mode to use for DTM Creation
    int mode = -1;
    if(options.getExtractionMethod() == "NN")
    {
        mode = 1;
    }
    else if(options.getExtractionMethod() == "IMA")
    {
        mode = 0;
    }
    else if(options.getExtractionMethod() == "THM")
    {
        mode = 2;
    }
    else
    {
        std::cout << timestamp.getElapsedTime() << "IO Error: Unable to interpret " << options.getExtractionMethod() << std::endl;
        return 0;
    }

    // =======================================================================
    // Load Additional Reference Points
    // =======================================================================    
    std::string currenSystem;
    int numberOfPoints;
    VecD *srcPoints;
    VecD *dstPoints;
     // Read Coordinate System and Reference Points from the provided File
    if(!options.getInputReferencePairs().empty())
    {
        std::ifstream input;
        VecD s,d;
        char ch;
        std::string num;
        input.open(options.getInputReferencePairs());
        if(input.fail())
        {
            std::cout << timestamp.getElapsedTime() << "IO Error: Unable to read " << options.getInputReferencePairs() << std::endl;
            return 0;
        }
        std::getline(input, currenSystem);        
        std::getline(input,num);
        numberOfPoints = std::stoi(num);        
        srcPoints = new VecD[numberOfPoints];
        dstPoints = new VecD[numberOfPoints];
        for(int i = 0; i < numberOfPoints; i++){
            input >> s.x >> ch >> s.y >> ch >> s.z;
            srcPoints[i] = s;
            input >> d.x >> ch >> d.y >> ch >> d.z;
            dstPoints[i] = d;
        }       
        
    }
    
    // =======================================================================
    // Read GeoTIFF and Warp
    // =======================================================================
    GeoTIFFIO* io = NULL;
    string newGeoTIFFName = options.getOutputFileName()+".tif";

    if(!options.getTargetSystem().empty())
    {
        if(!options.getInputReferencePairs().empty())
        {
            string targetSystem = options.getTargetSystem();
            // Transform reference points into target system
            transformPoints(currenSystem,targetSystem,dstPoints,dstPoints,numberOfPoints);

            if(!options.getInputGeoTIFF().empty())
            {
                GDALDatasetH src = GDALOpen(options.getInputGeoTIFF().c_str(),GA_ReadOnly);
                if(src == NULL)
                {
                    std::cout << timestamp.getElapsedTime() << "IO Error: Unable to read " << options.getInputGeoTIFF() << std::endl;
                }
                else
                {
                    GDALDatasetH dt;
                    // Creates a new GeoTIFF file with the transformed info of the old one
                    warpGeoTIFF(src,dt,targetSystem,newGeoTIFFName);
                    io = new GeoTIFFIO(newGeoTIFFName);
                }
            }
        }
    
    }
    else if(!options.getInputGeoTIFF().empty())
    {
        io = new GeoTIFFIO(options.getInputGeoTIFF());
    }
    
    // =======================================================================
    // Compute Affine Transform Matrix from Transformed Reff Points
    // =======================================================================
    Eigen::MatrixXd affineMatrix, affineTranslation, fullAffineMatrix, checkMatrix;
    bool noTransformation = false;
    if(!options.getInputReferencePairs().empty())
    {    
        // Right now, LVR2 doesn't support Large Coordinates and we can't use the Translation fully
        // In Functions where we use the Matrix we need to exclude the Translation 
        tie(affineMatrix,affineTranslation) = computeAffineGeoRefMatrix(srcPoints,dstPoints,numberOfPoints); 
        fullAffineMatrix = affineTranslation * affineMatrix;

        // Check, if Rotation is supported
        for (int i = 0; i < 16; i++)
        {
            // This is supposed to detect numbers that cannot be represented with float accuracy after Transformation
            // If there is an easier or more accurate way to achieve this, insert it here
            // Optimal Solution would probably be to output vertices as doubles
            if(abs(affineMatrix(i)) != 0)
            {
                if(abs(affineMatrix(i)) < 0.00001 || abs(affineMatrix(i)) > 1000)
                {
                    noTransformation = true;
                    affineMatrix = checkMatrix;
                    break;
                }
            }
        }        
    } 

    // =======================================================================
    // Extract ground from the point cloud
    // =======================================================================
    std::cout << timestamp.getElapsedTime() << "Start" << std::endl;
    if(mode == 0)
    {
        std::cout << "Moving Average" << std::endl;
        improvedMovingAverage<VecD,double>(mesh,usedArr,usedSurface,tree,options.getMinRadius(),options.getMaxRadius(),options.getNumberNeighbors(),options.getNumberNeighbors()+1,
            options.getRadiusSteps(),options.getResolution(),affineMatrix);
    }
    else if(mode == 1)
    {
        std::cout << "Nearest Neighbor" << std::endl;
        nearestNeighborMethod<VecD,double>(mesh,usedArr,usedSurface,tree,options.getNumberNeighbors(),options.getResolution(),affineMatrix);
    }
    else if(mode == 2)
    {
        std::cout << "Threshold Method"<< std::endl;
        thresholdMethod<VecD,double>(mesh,usedArr,usedSurface,options.getResolution(),tree,options.getSWSize(),options.getSWThreshold(),options.getLWSize(),options.getLWThreshold(),
            options.getSlopeThreshold(),affineMatrix);
    }
    std::cout << timestamp.getElapsedTime() << "End" << std::endl;
    
    // =======================================================================
    // Setup LVR_2 Function to allow the export of the Mesh as obj/ply
    // =======================================================================
    
    // Creating a cluster map made up of one cluster is necessary to use the finalizer and project the texture
    ClusterBiMap<FaceHandle> clusterBiMap;
    MeshHandleIteratorPtr<FaceHandle> iterator = mesh.facesBegin();
    auto newCluster = clusterBiMap.createCluster();
    for (size_t i = 0; i < mesh.numFaces(); i++)
    { 
        clusterBiMap.addToCluster(newCluster,*iterator);

        ++iterator;
    }  
    
    // Initialise Finalizer with ClusterMap
    TextureFinalizer<VecD> finalize(clusterBiMap);
    
    // Generate Texture for the OBJ file
    auto matResult = 
    projectTexture<VecD>(mesh,clusterBiMap,*usedSurface,texelSize,affineMatrix,fullAffineMatrix,io,tree,options.getStartingBand(),
    options.getNumberOfBands(),options.getColorScale(), noTransformation);
   
    // Pass Texture and Texture Coordinate into the Finalizer
    finalize.setMaterializerResult(matResult);  
    
    // Convert Mesh into Buffer and create Model
    auto buffer = finalize.apply(mesh);
    buffer->addIntAtomic(1, "mesh_save_textures");
    buffer->addIntAtomic(1, "mesh_texture_image_extension");
    std::cout << timestamp.getElapsedTime() << "Setting Model" << std::endl;
    auto m = ModelPtr(new Model(buffer)); 

    // =======================================================================
    // Export Files as PLY and OBJ with a JPEG as Texture
    // =======================================================================    
    std::cout << timestamp.getElapsedTime() << "Saving Model as ply" << std::endl;
    ModelFactory::saveModel(m,options.getOutputFileName() + ".ply");

    std::cout << timestamp.getElapsedTime() << "Saving Model as obj" << std::endl;
    ModelFactory::saveModel(m,options.getOutputFileName() + ".obj");  

    if(!options.getInputReferencePairs().empty())
    {
        std::ofstream file;
        file.open (options.getOutputFileName() + "_transformmatrix.txt");
        if(!noTransformation)
        {
            file << "Transformation Matrix without Translation\n" << affineMatrix << "\n" << "Translation\n" << affineTranslation;
        }
        else
        {
            std::cout << timestamp.getElapsedTime() << "Transformation cannot be applied without destroying the model. Full Transformation can be found in " << options.getOutputFileName() + "_transformmatrix.txt" << std::endl;
            file << "Full Transformation\n" << fullAffineMatrix;
        }
        file.close();
        delete(srcPoints);
        delete(dstPoints);

        if(!options.getInputGeoTIFF().empty())
        {
            delete(io);
        }
    }
    return 0;
}