/*
 * Main.hpp
 *
 *  Created on: 20.12.2020
 *      Author: Mario Dyczka (mdyczka@uos.de)
 */

#include <iostream>
#include <memory>
#include <tuple>
#include <map>
#include <chrono>
#include <ctime>  

#include <boost/optional.hpp>

#include <gdal.h>
#include <gdalwarper.h>
#include <ogr_spatialref.h>
#include <ogr_geometry.h>

#include "lvr2/geometry/HalfEdgeMesh.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/algorithm/FinalizeAlgorithms.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/algorithm/Texturizer.hpp"
#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/SearchTreeFlann.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/GeoTIFFIO.hpp"
#include "lvr2/display/ColorMap.hpp"

using boost::optional;
using std::unique_ptr;
using std::make_unique;

using namespace lvr2;

using Vec = BaseVector<float>;
using VecD = BaseVector<double>;

/**
 * @brief Calculates the affine Transformation between two sets of points using the Eigen Library.
 * 
 * @param srcPoints Set of points from the coordinate system you want to transform from.
 * @param destPoints Set of points from the coordinate system you want to transform to.
 * @param numberPoints Number of points in the sets.
 * @return std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> Return the transformation Matrix split into translational 
 *  and rotational part.
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> computeAffineGeoRefMatrix(VecD* srcPoints, VecD* destPoints, int numberPoints)
{
    // Create one M x 12 Matrix (contains the reference point coordinates in the point clouds systems), 
    //  one M x 1 Vector (contains the reference point coordinates in the target system)
    //  and one 12 x 1 Vector (will contain the transformation matrix values)
    // M = 3 * numberPoints
    Eigen::MatrixXd src(3 * numberPoints,12);
    Eigen::VectorXd dest(3 * numberPoints);
    for(int i = 0; i < numberPoints; i++)
    {
        src.row(i*3) << srcPoints[i].x, srcPoints[i].y, srcPoints[i].z, 1, 0, 0, 0, 0, 0, 0, 0, 0;
        src.row(i*3+1) << 0, 0, 0, 0, srcPoints[i].x, srcPoints[i].y, srcPoints[i].z, 1, 0, 0, 0, 0;
        src.row(i*3+2) << 0, 0, 0, 0, 0, 0, 0, 0, srcPoints[i].x, srcPoints[i].y, srcPoints[i].z, 1;
        dest.row(i*3) << destPoints[i].x;
        dest.row(i*3+1) << destPoints[i].y;
        dest.row(i*3+2) << destPoints[i].z;       
    }
       
    Eigen::VectorXd affineValues(12);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(src, Eigen::ComputeFullU | Eigen::ComputeFullV);
    affineValues = svd.solve(dest);    
    Eigen::MatrixXd affineMatrix(4,4);
    // Here we seperate Translation and Rotation, because we cannot ouput models with large coordinates
    auto ar = affineValues.array();
    affineMatrix << 
    ar[0], ar[1], ar[2], 0,
    ar[4], ar[5], ar[6], 0,  
    ar[8], ar[9], ar[10], 0, 
    0, 0, 0, 1;

    Eigen::MatrixXd affineTranslation(4,4);
    affineTranslation<< 
    1, 0, 0, ar[3],
    0, 1, 0, ar[7], 
    0, 0, 1, ar[11], 
    0, 0, 0, 1;
   
    return {affineMatrix,affineTranslation};
}

/**
 * @brief Warps a GeoTIFF into another geo-referenced coordinate system chosen by the user @geogCS. 
 *  The warped GeoTIFF is then stored as a new file @newGeoTIFFName.
 *  Based on the "GDAL Warp API tutorial" on https://gdal.org/tutorials/warp_tut.html.
 * 
 * @param src GeoTIFF dataset carrying the source information.
 * @param dt GeoTIFF dataset that will contain the warped information.
 * @param geogCS Coordinate system we want to warp into.
 * @param newGeoTIFFName Name of the created GeoTIFF.
 */
void warpGeoTIFF(GDALDatasetH& src, GDALDatasetH& dt, const std::string& geogCS, const std::string& newGeoTIFFName )
{    
    const char *pszSrcWKT = NULL;
    char *pszDstWKT  = NULL;

    // Initialise Driver
    GDALDriverH hDriver = GDALGetDriverByName( "GTiff" );
    CPLAssert( hDriver != NULL );

    // Get Coordinate Information from Source
    pszSrcWKT = GDALGetProjectionRef(src);
    CPLAssert( pszSrcWKT != NULL && strlen(src) > 0 );

    GDALDataType eDT = GDALGetRasterDataType(GDALGetRasterBand(src,1));

    // Create Coordinate Informatiom for Destination    
    OGRSpatialReference oSRS;
    oSRS.SetFromUserInput(geogCS.c_str());
    oSRS.exportToWkt(&pszDstWKT);
    CPLAssert( Og == 0 );

    void *hTransformArg;
    hTransformArg =
        GDALCreateGenImgProjTransformer( src, pszSrcWKT, NULL, pszDstWKT,
                                        FALSE, 0, 1 );
    CPLAssert( hTransformArg != NULL );
    
    // Approximated output
    double adfDstGeoTransform[6];
    int nPixels=0, nLines=0;
    CPLErr eErr;
    eErr = GDALSuggestedWarpOutput( src,
                                    GDALGenImgProjTransform, hTransformArg,
                                    adfDstGeoTransform, &nPixels, &nLines );
    
    CPLAssert( eErr == CE_None );
    GDALDestroyGenImgProjTransformer( hTransformArg );
    dt = GDALCreate( hDriver, newGeoTIFFName.c_str(), nPixels, nLines,
                        GDALGetRasterCount(src), eDT, NULL );
    CPLAssert( dt != NULL );
    
    
    GDALSetProjection( dt, pszDstWKT );
    GDALSetGeoTransform( dt, adfDstGeoTransform );  

    // Extract and Set additional raster data
    for (size_t i = 1; i <= GDALGetRasterCount(src); i++)
    {
        auto bandSrc = GDALGetRasterBand(src,i);
        auto bandDst = GDALGetRasterBand(dt,i);
        auto noData = GDALGetRasterNoDataValue( bandSrc, nullptr);
        GDALSetRasterNoDataValue(bandDst, noData);
        auto unitType = GDALGetRasterUnitType(bandSrc);
        GDALSetRasterUnitType(bandDst,unitType);
        double max;
        double min;
        double mean;
        double stdDev;
        GDALGetRasterStatistics(bandSrc,FALSE,TRUE,&min,&max,&mean,&stdDev);
        GDALSetRasterStatistics(bandDst,min,max,mean,stdDev);

        GDALColorTableH hCT;
        hCT = GDALGetRasterColorTable( GDALGetRasterBand(src,i) );
        if( hCT != NULL )
        {
            GDALSetRasterColorTable( GDALGetRasterBand(dt,i), hCT );
        }
        
    }    

    // Warp Image
    GDALWarpOptions *psWarpOptions = GDALCreateWarpOptions();
    psWarpOptions->hSrcDS = src;
    psWarpOptions->hDstDS = dt;
    psWarpOptions->nBandCount = 0;
    psWarpOptions->pfnProgress = GDALTermProgress;
    psWarpOptions->papszWarpOptions = 
    CSLSetNameValue(psWarpOptions->papszWarpOptions,"OPTIMIZE_SIZE","TRUE");

    // Reprojections transformer
    psWarpOptions->pTransformerArg =
        GDALCreateGenImgProjTransformer( src,
                                        GDALGetProjectionRef(src),
                                        dt,
                                        GDALGetProjectionRef(dt),
                                        FALSE, 0.0, 1 );
    psWarpOptions->pfnTransformer = GDALGenImgProjTransform;

    // Execute warp
    GDALWarpOperation oOperation;
    oOperation.Initialize( psWarpOptions );
    oOperation.ChunkAndWarpImage( 0, 0,
                                GDALGetRasterXSize( dt ),
                                GDALGetRasterYSize( dt ) );
    
    GDALDestroyGenImgProjTransformer( psWarpOptions->pTransformerArg );
    
    GDALDestroyWarpOptions( psWarpOptions );    

    GDALClose( dt );
    GDALClose( src );
}

/**
 * @brief Transforms a set of points from one geo-referenced coordinate system to another utilising GDAL.
 * 
 * @tparam BaseVecT Sets which BaseVector template is used.
 * @param src EPSG Code of the source coordinate system.
 * @param dt EPSG Code of the target coordinate system.
 * @param srcPoints Set of source points.
 * @param destPoints Array that contains the set of transformed points after calculation.
 * @param numberOfPoints The number of points that are to be transformed.
 */
template <typename BaseVecT>
void transformPoints(string src,string dt, BaseVecT* srcPoints, BaseVecT* destPoints,int numberOfPoints)
{
    // =======================================================================
    // Get Coordinate Systems and Transform Points using GDAL
    // =======================================================================
    GDALAllRegister();
    OGRSpatialReference source, target;
    source.SetFromUserInput(src.c_str());
    std::cout << std::endl;
    target.SetFromUserInput(dt.c_str());
    OGRPoint p;
    
    for(int i = 0; i < numberOfPoints; i++)
    {    
        p.assignSpatialReference(&source);  
        p.setX(srcPoints[i].x);
        p.setY(srcPoints[i].y);
        p.setZ(srcPoints[i].z);
        p.transformTo(&target); 
        
        destPoints[i] = BaseVecT(p.getX(),p.getY(),p.getZ());
    }

}

/**
 * @brief Searches a cuboid area inside a point cloud for the lowest point inside and returns its z value.
 * 
 * @tparam BaseVecT Sets which BaseVector template is used.
 * @tparam Data Sets which data type (float,double) is used.
 * @param x x coordinate.
 * @param y y coordinate.
 * @param lowestZ Lowest recorded z coordinate value of the point clouds bounding box.
 * @param highestZ Highest recorded z coordinate value of the point clouds bounding box.
 * @param searchArea Size of the radius around the specified datum that is to be searched for points.
 * @param tree A search tree that houses the point clouds information and that can be used for radius search (uses the FLANN).
 * @param points Is used in conjunction with @tree to extract information about the points found using radius search.
 * @return Data If a point was found, it returns its z value. Otherwise, the max value of the defined @Data type is returned.
 */
template <typename BaseVecT, typename Data>
Data findLowestZ(Data x, Data y, Data lowestZ, Data highestZ, Data searchArea, SearchTreeFlann<BaseVecT>& tree,FloatChannel& points)
{    
    Data bestZ = highestZ;
    Data currentZ = lowestZ;
    bool found = false;

    // =======================================================================
    // Look for Point with the Lowest height in the x/y coordinate
    // =======================================================================
    // Utilises radiusSearch
    do
    {
        vector<size_t> neighbors;  
        vector<Data> distances; 

        // Look for the closest point whith a z-value lower then our currently best point
        // We increase the radius so we have the whole area the node is affected by covered
        // And later check if the nodes found are inside the square
        size_t numNeighbors = tree.radiusSearch(BaseVecT(x,y,currentZ), 100, 1.5 * searchArea, neighbors, distances);
        for (size_t j = 0; j < numNeighbors; j++)
        {
            size_t pointIdx = neighbors[j];
            auto cp = points[pointIdx];

            if(cp[2] <= bestZ)
            {
                if(sqrt(pow(x - cp[0],2)) <= searchArea && sqrt(pow(y - cp[1],2)) <= searchArea)
                {
                    bestZ = cp[2];
                    found = true;                   
                }
            }
        }   
        if(found)
        {
            return bestZ;
        }
        currentZ += searchArea;

    } while (currentZ <= highestZ);

    return std::numeric_limits<Data>::max();
    
}

/**
 * @brief Builds a DTM utilising Nearest Neighbor Search and LowestZ on a point cloud. 
 * If points found by Nearest Neighbor Search lie outside a predefined area, the affected node is excluded from the model.
 * 
 * @tparam BaseVecT Sets which BaseVector template is used.
 * @tparam Data Sets which data type (float,double) is used.
 * @param mesh HalfEdgeMesh the extracted ground data is written to.
 * @param points Contains the information of all the points from the point cloud.
 * @param surface Point cloud manager that can generate the point clouds bounding box.
 * @param tree Search Tree that utilises the FLANN to enable Radius and Nearest Neighbor Search on the point clouds data.
 * @param numNeighbors Number of neighbours used for Nearest Neighbor Search.
 * @param resolution Parameter that sets the distance between the vertices of the mesh.
 * @param affineMatrix If a matrix is provided, the mesh's vertices will be transformed using it.
 */
template <typename BaseVecT, typename Data>
void nearestNeighborMethod(lvr2::HalfEdgeMesh<VecD>& mesh, FloatChannel& points, PointsetSurfacePtr<Vec>& surface,
SearchTreeFlann<BaseVecT>& tree ,int numNeighbors, Data resolution, Eigen::MatrixXd& affineMatrix)
{
    // =======================================================================
    // Generating Boundingbox and Initialising 
    // =======================================================================
    auto bb = surface->getBoundingBox();
    auto max = bb.getMax();
    auto min = bb.getMin();
    
    ssize_t xMax = (ssize_t)(std::round(max.x));
    ssize_t xMin = (ssize_t)(std::round(min.x));
    ssize_t yMax = (ssize_t)(std::round(max.y));
    ssize_t yMin = (ssize_t)(std::round(min.y));
    ssize_t zMax = (ssize_t)(std::round(max.z));
    ssize_t zMin = (ssize_t)(std::round(min.z));
    ssize_t xDim = abs(xMax) + abs(xMin);
    ssize_t yDim = abs(yMax) + abs(yMin);

    vector<size_t> indices;
    vector<Data> distances;
    Data finalZ = 0;         
    int trustedNeighbors = 0;
    int numberNeighbors = numNeighbors;

    xMin = xMin * (1/resolution);
    xMax = xMax * (1/resolution);
    yMin = yMin * (1/resolution);
    yMax = yMax * (1/resolution);      
    
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle> dict1;
    
    // =======================================================================
    // Calculate Vertice Height and create Hexagonial Net
    // =======================================================================
    ProgressBar progressVert((xDim / resolution)*(yDim / resolution), timestamp.getElapsedTime() + "Calculating Grid");
    for (ssize_t x = xMin; x < xMax; x++)
    {        
        for (ssize_t y = yMin; y < yMax; y++)
        {               
            Data u_x = x * resolution;
            Data u_y = y * resolution;
            indices.clear();
            distances.clear();
            finalZ = 0;  

            // Check if there are ground points near the node
            Data closeZ = findLowestZ<BaseVecT,Data>(u_x,u_y,zMin,zMax,resolution/2,tree,points);
            if(closeZ == std::numeric_limits<Data>::max())
            {
                ++progressVert;
                continue;
            }         

            // Use Nearest Neighbor Search to find the necessary amount of neighbors     
            tree.kSearch(BaseVecT(u_x,u_y,closeZ),numberNeighbors,indices,distances);
            
            trustedNeighbors = numberNeighbors;
            for (int i = 0; i < numberNeighbors; i++)
            {
                if(distances[i] > resolution)
                {
                    trustedNeighbors--;
                    break;
                }
                else
                {
                    auto index = indices[i];
                    auto nearest = points[index];
                    finalZ = finalZ + nearest[2];
                }
            }

            // If there are not enough points surrounding our node, it is left blank
            // Else, the nodes height value is set to its neighbors aithmetic mean
            if(trustedNeighbors < numberNeighbors)
            {
                ++progressVert;
                continue;
            }  
            else
            {
                finalZ = finalZ/trustedNeighbors;
            }

            //Valid Nodes are saved 
            Data d_x = u_x;
            Data d_y = u_y;
            Data d_z = finalZ;
            
            //Apply Translation Matrix
            if(affineMatrix.size() != 0)
            {
                Eigen::Vector4d point(u_x,u_y,finalZ,1);

                Eigen::Vector4d solution;
                solution = affineMatrix*point;
                d_x = solution.coeff(0);
                d_y = solution.coeff(1);
                d_z = solution.coeff(2);
            }

            VertexHandle v1 = mesh.addVertex(VecD(d_x,d_y,d_z)); 
            dict1.emplace(std::make_tuple(x,y),v1);
            ++progressVert;
        }

    }
    std::cout << std::endl;

    // Write Nodes to the mesh structure and connect them to faces
    ProgressBar progressGrid(dict1.size(), timestamp.getElapsedTime() + "Writing Grid");
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator pf1;
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator pf2;

    for(auto it = dict1.begin(); it != dict1.end(); )
    {
        tuple t = it->first;
        ssize_t x = std::get<0>(t);
        ssize_t y = std::get<1>(t);    
       
        pf1 = dict1.find(make_tuple(x,y+1));
        if(pf1 != dict1.end())
        {   
            pf2 = dict1.find(make_tuple(x-1,y));
            if(pf2 != dict1.end())
            {
                mesh.addFace(it->second,pf1->second,pf2->second);
            }
        }

        pf1 = dict1.find(make_tuple(x,y+1));
        if(pf1 != dict1.end())
        {   
            pf2 = dict1.find(make_tuple(x+1,y+1));
            if(pf2 != dict1.end())
            {
                mesh.addFace(pf2->second,pf1->second,it->second);
            }
        }

        it++;
        ++progressGrid;
    }

    std::cout << std::endl;
}

/**
 * @brief Utilises an improved version of the Moving Average algorithm to generate a DTM from a point cloud. Based on 
 * [W. Maleika. Moving average optimization in digital terrain model generation based on test multibeam echosounder data.Geo-Marine Letters, 35(1):61–68, 2015.].
 * The algorithm searches for points in a predefined radius around the nodes. If enough points are found, the height of the nodes is calculated based on their elevation.
 * How much a single point influences the node is decided by the point's distance to the node. When the amount of points is too low, the search radius is extended,
 * until the maximum radius is reached. Should this happen, the node is excluded from the mesh.
 * 
 * @tparam BaseVecT Sets which BaseVector template is used.
 * @tparam Data Sets which data type (float,double) is used.
 * @param mesh HalfEdgeMesh the extracted ground data is written to.
 * @param points Contains the information of all the points from the point cloud.
 * @param surface Point cloud manager that can generate the point clouds bounding box.
 * @param tree Search Tree that utilises the FLANN to enable Radius and Nearest Neighbor Search on the point clouds data.
 * @param minRadius Starting Radius.
 * @param maxRadius Maximum Radius.
 * @param minNeighbors The minimum amount of neighbours. If fewer neighbours are found, the node is excluded from the model.
 * @param maxNeighbors The maximum amount of neighbours to look for.
 * @param radiusSteps The number of steps necessary to extend from the @minRadius to the @maxRadius.
 * @param resolution Parameter that sets the distance between the vertices of the mesh.
 * @param affineMatrix If a matrix is provided, the mesh's vertices will be transformed using it.
 */
template <typename BaseVecT, typename Data>
void improvedMovingAverage(lvr2::HalfEdgeMesh<VecD>& mesh, FloatChannel& points, PointsetSurfacePtr<Vec>& surface,
SearchTreeFlann<BaseVecT>& tree, float minRadius, float maxRadius, int minNeighbors, int maxNeighbors, int radiusSteps, float resolution, Eigen::MatrixXd& affineMatrix )
{
    // =======================================================================
    // Generating Boundingbox and Initialising 
    // =======================================================================
    auto bb = surface->getBoundingBox();
    auto max = bb.getMax();
    auto min = bb.getMin();

    ssize_t xMax = (ssize_t)(std::round(max.x));
    ssize_t xMin = (ssize_t)(std::round(min.x));
    ssize_t yMax = (ssize_t)(std::round(max.y));
    ssize_t yMin = (ssize_t)(std::round(min.y));
    ssize_t zMax = (ssize_t)(std::round(max.z));
    ssize_t zMin = (ssize_t)(std::round(min.z));
    ssize_t xDim = abs(xMax) + abs(xMin);
    ssize_t yDim = abs(yMax) + abs(yMin);

    size_t numberNeighbors = 0;

    int found = 0;

    xMin = xMin * (1/resolution);
    xMax = xMax * (1/resolution);
    yMin = yMin * (1/resolution);
    yMax = yMax * (1/resolution);

    Data finalZ = 0;  
    Data addedDistance = 0;
    vector<size_t> indices;  
    vector<Data> distances;  

    // Calculate the radius step size
    float radiusStepsize = (maxRadius - minRadius)/radiusSteps;
    float radius = 0;

    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle> dict;

    ProgressBar progressVert((xDim/resolution)*(yDim/resolution), timestamp.getElapsedTime() + "Calculating height values"); 

    // =======================================================================
    // Calculate Vertice Height and create Hexagonial Net
    // =======================================================================
    for (ssize_t x = xMin; x < xMax; x++)
    {        
        for (ssize_t y = yMin; y < yMax; y++)
        {           
            Data u_x = x * resolution;
            Data u_y = y * resolution;

            BaseVecT point;
            found = 0;
            finalZ = 0;  
            addedDistance = 0;
            indices.clear();
            distances.clear(); 

            radius = minRadius;
            // Use lowestZ to find the start of the ground area --> if there is no ground area, the node is skipped
            Data u_z = findLowestZ<BaseVecT,Data>(u_x,u_y,zMin,zMax,resolution/2,tree,points);
            if(u_z == std::numeric_limits<Data>::max())
            {
                ++progressVert; 
                continue;
            }
            point = BaseVecT(u_x,u_y,u_z);
            // If we don't find enough points in the current radius, we extend the radius
            // If we hit the maximum extension and still find nothing, the node is left blank
            while (found == 0)
            {
                numberNeighbors = tree.radiusSearch(point,maxNeighbors,radius,indices,distances);
                if(numberNeighbors >= minNeighbors)
                {
                    found = 1;
                    break;
                }
                else if(radius <= maxRadius)
                {   
                    radius += radiusStepsize;
                    continue;
                }
                else
                {
                    found = -1;
                    break;
                }                 
            }
            // The nodes height value is calculated by weighting the surrounding points depending on their distance to the node
            if(found == 1)
            {
                for (int i = 0; i < numberNeighbors; i++)
                {
                    size_t pointIdx = indices[i];
                    auto neighbor = points[pointIdx];
                    // When we are exactly on the point, distance is 0 and would divide by 0
                    Data distance = 1;
                    if(distances[i] != 0)
                    {
                        // Calculates inverted distance
                        distance = 1/distance;; 
                    }                        
                    
                    finalZ += neighbor[2] * distance;
                    addedDistance += distance;                
                }  
                
                finalZ = finalZ/addedDistance;                                 
                
            }
            else
            {
                ++progressVert; 
                continue;
            }

            // The viable nodes are saved
            Data d_x = u_x;
            Data d_y = u_y;
            Data d_z = finalZ;
            // Apply Translation Matrix
            if(affineMatrix.size() != 0)
            {
                Eigen::Vector4d point(u_x,u_y,finalZ,1);
                Eigen::Vector4d solution;
                solution = affineMatrix*point;
                d_x = solution.coeff(0);
                d_y = solution.coeff(1);
                d_z = solution.coeff(2);
            }
            VertexHandle v1 = mesh.addVertex(VecD(d_x,d_y,d_z)); 
            
            dict.emplace(std::make_tuple(x,y),v1);
            
            ++progressVert;         
        }        
        
    }
    std::cout << std::endl;
    //All nodes are put inside the mesh structure
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator pf1;
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator pf2;

    ProgressBar progressGrid(dict.size(), timestamp.getElapsedTime() + "Writing Grid");

    for(auto it = dict.begin(); it != dict.end(); )
    {
        tuple t = it->first;
        ssize_t x = std::get<0>(t);
        ssize_t y = std::get<1>(t);  

        pf1 = dict.find(make_tuple(x,y+1));
        if(pf1 != dict.end())
        {   
            pf2 = dict.find(make_tuple(x-1,y));
            if(pf2 != dict.end())
            {
                mesh.addFace(it->second,pf1->second,pf2->second);
            }
        }

        pf1 = dict.find(make_tuple(x,y+1));
        if(pf1 != dict.end())
        {   
            pf2 = dict.find(make_tuple(x+1,y+1));
            if(pf2 != dict.end())
            {
                mesh.addFace(pf2->second,pf1->second,it->second);
            }
        }

        it++;
        ++progressGrid;
    }

    std::cout << std::endl;
}

/**
 * @brief Extracts the ground points from a PLY point cloud and puts them into a HalfEdgeMesh structure using an algorithm based on 
 * [P.  Rashidi  and  H.  Rastiveis.   Extraction  of  Ground  Points  from  LiDAR  Data  Based  on Slope and Progressive Window Thresholding (SPWT).
 * Earth Observation and GeomaticsEngineering, 2(1):36–44, 2018.]
 * The point cloud is used to build a grid of nodes. Each node has to pass three tests to be included in the DTM. These tests try to confirm, whether the node is a ground point.
 * The Small Window Thresholding method compares a node's z value to its neighbours' z values. If the difference between the lowest neighbour and the node exceeds a threshold
 * set by the user, the node does not belong to the ground layer and it is excluded from the mesh. 
 * Large Window Thresholding operates similarly. It compares a node's z value to its neighbours in a larger radius.
 * The Slope Thresholding Method calculates the slope angle between a node and its direct neighbours. If the angle exceeds a threshold, the node is excluded from the model.
 * 
 * @tparam BaseVecT Sets which BaseVector template is used.
 * @tparam Data Sets which data type (float,double) is used.
 * @param mesh HalfEdgeMesh the extracted ground data is written to.
 * @param points Contains the information of all the points from the point cloud.
 * @param surface Point cloud manager that can generate the point clouds bounding box.
 * @param resolution Parameter that sets the distance between the vertices of the mesh.
 * @param tree Search Tree that utilises the FLANN to enable Radius and Nearest Neighbor Search on the point clouds data.
 * @param smallWindow Sets the size of the x * x window of neighbors' around a node that is used on Small Window Thresholding.
 * @param smallWindowHeight Sets the threshold for Small Window Thresholding.
 * @param largeWindow Sets the size of the x * x window of neighbors' around a node that is used on Large Window Thresholding.
 * @param largeWindowHeight Sets the threshold for Large Window Thresholding.
 * @param slopeThreshold Sets the threshold for Slope Thresholding.
 * @param affineMatrix If a matrix is provided, the mesh's vertices will be transformed using it.
 */
template <typename BaseVecT, typename Data>
void thresholdMethod(lvr2::HalfEdgeMesh<VecD>& mesh,FloatChannel& points, PointsetSurfacePtr<Vec>& surface, float resolution,
 SearchTreeFlann<BaseVecT>& tree, int smallWindow, float smallWindowHeight, int largeWindow, float largeWindowHeight, float slopeThreshold, Eigen::MatrixXd affineMatrix)
{
    // =======================================================================
    // Generate the Bounding Box and calculate the Grids size
    // =======================================================================
    auto bb = surface->getBoundingBox();
    auto max = bb.getMax();
    auto min = bb.getMin();
    
    ssize_t xMax = (ssize_t)(std::round(max.x));
    ssize_t xMin = (ssize_t)(std::round(min.x));
    ssize_t yMax = (ssize_t)(std::round(max.y));
    ssize_t yMin = (ssize_t)(std::round(min.y));
    ssize_t zMax = (ssize_t)(std::round(max.z));
    ssize_t zMin = (ssize_t)(std::round(min.z));
    ssize_t xDim = abs(xMax) + abs(xMin);
    ssize_t yDim = abs(yMax) + abs(yMin);

    int maxNeighbors = 1;
    Data averageHeight = 0;
    Data xR = xDim/resolution;
    int xReso = std::round(xR);
    Data yR = yDim/resolution;
    int yReso = std::round(yR);

    vector<vector<Data>> workGrid;
    workGrid.resize(xReso+1, vector<Data>(yReso+1,0));
    vector<size_t> indices;  
    vector<Data> distances;  
    
    // =======================================================================
    // Generate the Grid
    // =======================================================================
    ProgressBar progressGrid(xReso * yReso, timestamp.getElapsedTime() + "Calculating Grid");
    for(ssize_t y = 0; y < yReso; y++)
    {
        for(ssize_t x = 0; x < xReso; x++)
        {
            Data u_x = x * resolution;
            Data u_y = y * resolution;

            indices.clear();
            // Set the grid node's height according to its nearest neighbors
            
            int numberNeighbors = tree.kSearch(BaseVecT(u_x + xMin,u_y + yMin,findLowestZ<BaseVecT,Data>(u_x + xMin,u_y + yMin,zMin,zMax,resolution/2,tree,points)),
             maxNeighbors, indices, distances);

            if(numberNeighbors == 0)
            {
                workGrid[x][y] = std::numeric_limits<Data>::max();
                ++progressGrid;
                continue;
            }
            averageHeight = 0;

            for(int i = 0; i < numberNeighbors; i++)
            {
                auto p = points[indices[i]];
                averageHeight += p[2];
            }
            averageHeight /= numberNeighbors;
            workGrid[x][y] = averageHeight;
            ++progressGrid;
        }
    } 
    std::cout << std::endl;
    
    // =======================================================================
    // Extraction of Ground Points via Thresholding
    // =======================================================================
    // Three steps that try to filter ground points from non-ground points
    ProgressBar progressPoints(xDim/resolution * yDim/resolution, timestamp.getElapsedTime() + "Checking Points");    
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle> dict;
    
    for (ssize_t y = 0; y < yReso; y++)
    {
        for (ssize_t x = 0; x < xReso; x++)
        {
            if(workGrid[x][y] == std::numeric_limits<Data>::max())
            {
                ++progressPoints;
                continue;
            }
            Data u_x = x * resolution;
            Data u_y = y * resolution;

            // =======================================================================
            // Small Window Thresholding
            // =======================================================================
            // check every pixel and compare with their neighbors --> sort out pixel, if larger than local minimum         
            Data lowestDist = std::numeric_limits<Data>::max();

            int swMax = (smallWindow-1)/2;
            int swMin = 0-swMax;

            for (int xwin = swMin; xwin <= swMax; xwin++)
            {
                for (int ywin = swMin; ywin <= swMax; ywin++)
                {
                    if((xwin + x) > xReso || (xwin + x) < 0)
                    {
                        continue;
                    }
                    if((ywin + y) > yReso || (ywin + y) < 0)
                    {
                        continue;
                    }
                    if(workGrid[xwin + x][ywin + y] == std::numeric_limits<Data>::max())
                    {
                        continue;
                    }
                    if(ywin == 0 && xwin == 0){
                        continue;
                    }
                    
                    if(workGrid[xwin + x][ywin + y] < lowestDist)
                    {
                        lowestDist = workGrid[xwin + x][ywin + y];
                    }
                }                
            }

            // Compare lowest z with points z
            // If height differen biggern then smallWindowHeight 
            // The point does not belong to the surface area
            if(abs(workGrid[x][y] - lowestDist) > smallWindowHeight)
            {
                ++progressPoints;
                continue;
            }

            // =======================================================================
            // Slope Thresholding
            // =======================================================================
            // Calculates the Slope between the observed point and its neighbors
            // If the Slope exerts a threshhold, the point is a non-ground point
            bool slopeGood = true;
            
            for (int xwin = -1; xwin <= 1; xwin++)
            {
                for (int ywin = -1; ywin <= 0; ywin++)
                {
                    //Only use points that we have set before
                    if(ywin == 0 && xwin == 1)
                    {
                        continue;
                    }
                    if((xwin + x) > xReso || (xwin + x) < 0)
                    {
                        continue;
                    }
                    if((ywin + y) > yReso || (ywin + y) < 0)
                    {
                        continue;
                    }
                    if(workGrid[xwin + x][ywin + y] == std::numeric_limits<Data>::max())
                    {
                        continue;
                    }
                    if(ywin == 0 && xwin == 0){
                        continue;
                    }
                    
                    float slope = 0;            
                    if(workGrid[xwin + x][ywin + y] != workGrid[x][y])
                    {
                        slope = atan(abs(workGrid[xwin + x][ywin + y] 
                         - workGrid[x][y])/sqrt(pow(xwin + x - x,2) + pow(ywin + y - y,2)));
                         slope = slope * 180/M_PI;
                    }                                        
                    
                    if(slope > slopeThreshold)
                    {
                        slopeGood = false;
                        break;
                    }
                }                
            }

            if(!slopeGood)
            {
                ++progressPoints;
                continue;
            }
           
            // Similar too Small Window Thresholding; used to elimnate large Objects like Trees
            lowestDist = std::numeric_limits<Data>::max();

            int lwMax = (largeWindow-1)/2;
            int lwMin = 0-swMax;

            for (int xwin = lwMin; xwin <= lwMax; xwin++)
            {
                for (int ywin = lwMin; ywin <= lwMax; ywin++)
                {
                    if((xwin + x) > xReso || (xwin + x) < 0)
                    {
                        continue;
                    }
                    if((ywin + y) > yReso || (ywin + y) < 0)
                    {
                        continue;
                    }
                    if(workGrid[xwin + x][ywin + y] == std::numeric_limits<Data>::max())
                    {
                        continue;
                    }
                    if(ywin == 0 && xwin == 0){
                        continue;
                    }

                    if(workGrid[xwin + x][ywin + y] < lowestDist)
                    {
                        lowestDist = workGrid[xwin + x][ywin + y];
                    }
                }                
            }
            
            if(abs(workGrid[x][y] - lowestDist) > largeWindowHeight)
            {
                ++progressPoints;
                continue;
            }
            
            // If the node passes through the three tests, it gets recoginised as ground point
            Data v_x = u_x+xMin;
            Data v_y = u_y+yMin;
            Data v_z = workGrid[x][y];

            // Apply transformation matrix
            if(affineMatrix.size() != 0)
            {
                Eigen::Vector4d point(v_x,v_y,v_z,1);
                Eigen::Vector4d solution;
                solution = affineMatrix*point;
                v_x = solution.coeff(0);
                v_y = solution.coeff(1);
                v_z = solution.coeff(2);
            }
            VertexHandle v = mesh.addVertex(VecD(v_x,v_y,v_z));
            dict.emplace(std::make_tuple(x,y),v);
            ++progressPoints;
        }
        
    }
    std::cout << std::endl;

    //Write nodes to mesh structure and create faces
    ProgressBar progressGrid2(dict.size(), timestamp.getElapsedTime() + "Writing Grid");
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator pf1;
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator pf2;

    for(auto it = dict.begin(); it != dict.end(); )
    {
        tuple t = it->first;
        ssize_t x = std::get<0>(t);
        ssize_t y = std::get<1>(t);  
       
        pf1 = dict.find(make_tuple(x,y+1));
        if(pf1 != dict.end())
        {   
            pf2 = dict.find(make_tuple(x-1,y));
            if(pf2 != dict.end())
            {
                mesh.addFace(it->second,pf1->second,pf2->second);
            }
        }

        pf1 = dict.find(make_tuple(x,y+1));
        if(pf1 != dict.end())
        {   
            pf2 = dict.find(make_tuple(x+1,y+1));
            if(pf2 != dict.end())
            {
                mesh.addFace(pf2->second,pf1->second,it->second);
            }
        }

        it++;
        ++progressGrid2;
    }

    std::cout << std::endl;
}

/**
 * @brief Generates a Texture that shows the distance between a mesh and the point clouds it is based on. 
 * Each of the texture's texels represents the distance between the mesh to the highest point of the point cloud inside the texel's area.
 * 
 * @tparam BaseVecT Sets which BaseVector template is used.
 * @tparam Data Sets which data type (float,double) is used.
 * @param surface Point cloud manager that can generate the point clouds bounding box.
 * @param tree Search Tree that utilises the FLANN to enable Radius and Nearest Neighbor Search on the point clouds data. 
 * @param mesh Mesh structure that is compared to the point cloud.
 * @param texelSize The x*x size of a texel. This also sets the resolution of the texture.
 * @param affineMatrix If the mesh was transformed using a matrix, it needs to be used here as well for the texture to be generated correctly.
 * @param colorScale The user can choose between the following colour scales: GREY, JET, HOT, HSV, SHSV, SIMPSONS.
 * @return Texture Depicts the height difference between @mesh and point cloud.
 */
template <typename BaseVecT, typename Data>
Texture generateHeightDifferenceTexture(const PointsetSurface<Vec>& surface ,SearchTreeFlann<BaseVecT>& tree,const lvr2::HalfEdgeMesh<VecD>& mesh, Data texelSize, 
Eigen::MatrixXd affineMatrix, string colorScale)
{
    // =======================================================================
    // Generate Bounding Box and prepare Variables
    // =======================================================================
    auto bb = surface.getBoundingBox();
    auto max = bb.getMax();
    auto min = bb.getMin();
    
    ssize_t xMax = (ssize_t)(std::round(max.x));
    ssize_t xMin = (ssize_t)(std::round(min.x));
    ssize_t yMax = (ssize_t)(std::round(max.y));
    ssize_t yMin = (ssize_t)(std::round(min.y));
    ssize_t zMax = (ssize_t)(std::round(max.z));
    ssize_t zMin = (ssize_t)(std::round(min.z));

    // Adjust Max/Min if affineMatrix was used
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
    
    ssize_t xDim = (abs(xMax) + abs(xMin))/texelSize; 
    ssize_t yDim = (abs(yMax) + abs(yMin))/texelSize;    

    // Initialise the texture that will contain the height information
    Texture texture(0, xDim, yDim, 3, 1, texelSize);

    // Contains the distances from each relevant point in the mesh to its closest neighbor
    Data* distance = new Data[xDim * yDim];
    Data maxDistance = 0;
    Data minDistance = std::numeric_limits<Data>::max();

    // Initialise distance vector
    for (int y = 0; y < yDim; y++)
    {
        for (int x = 0; x < xDim; x++)
        {
            distance[(yDim - y - 1) * (xDim) + x] = std::numeric_limits<Data>::min();
        }
    }

    // Get the Channel containing the point coordinates
    PointBufferPtr baseBuffer = surface.pointBuffer();   
    FloatChannel arr =  *(baseBuffer->getFloatChannel("points"));   
    MeshHandleIteratorPtr<FaceHandle> iterator = mesh.facesBegin();

    // =======================================================================
    // Iterate over all faces + calculate which Texel they are represented by
    // =======================================================================
    std::cout << timestamp.getElapsedTime() + "Generating Height Difference Texture" << std::endl;
    ProgressBar progressDistance(mesh.numFaces(), timestamp.getElapsedTime() + "Calculating Distance from Point Cloud to Model");
    
    for (size_t i = 0; i < mesh.numFaces(); i++)
    {
        BaseVecT correct(xMin,yMin,0);
        auto realPoint1 = mesh.getVertexPositionsOfFace(*iterator)[0];
        auto realPoint2 = mesh.getVertexPositionsOfFace(*iterator)[1];
        auto realPoint3 = mesh.getVertexPositionsOfFace(*iterator)[2];
        // Normalise Point Coordinates
        auto point1 = realPoint1 - correct;
        auto point2 = realPoint2 - correct;
        auto point3 = realPoint3 - correct;
        auto maxX = std::max(point1[0],std::max(point2[0],point3[0]));
        ssize_t fmaxX = (ssize_t)(maxX +1);
        auto minX = std::min(point1[0],std::min(point2[0],point3[0]));
        ssize_t fminX = (ssize_t)(minX -1);
        auto maxY = std::max(point1[1],std::max(point2[1],point3[1]));
        ssize_t fmaxY = (ssize_t)(maxY +1);
        auto minY = std::min(point1[1],std::min(point2[1],point3[1]));
        ssize_t fminY = (ssize_t)(minY -1);

        fminY = std::round(fminY/texelSize);
        fmaxY = std::round(fmaxY/texelSize);
        fminX = std::round(fminX/texelSize);
        fmaxX = std::round(fmaxX/texelSize);

        // Calculate the faces surface necessary for barycentric coordinate calculation
        Data faceSurface = 0.5 *((point2[0] - point1[0])*(point3[1] - point1[1])
            - (point2[1] - point1[1]) * (point3[0] - point1[0]));
        
        // Check Texels around the faces
        #pragma omp parallel for collapse(2)
        for (ssize_t y = fminY; y < fmaxY; y++)
        {
            for (ssize_t x = fminX; x < fmaxX; x++)
            {
                // We want the information in the center of the pixel
                Data u_x = x * texelSize + texelSize/2;
                Data u_y = y * texelSize + texelSize/2;

                // Check, if this face carries the information for texel (x,y)
                Data surface1 = 0.5 *((point2[0] - u_x)*(point3[1] - u_y)
                - (point2[1] - u_y) * (point3[0] - u_x));

                Data surface2 = 0.5 *((point3[0] - u_x)*(point1[1] - u_y)
                - (point3[1] - u_y) * (point1[0] - u_x));

                Data surface3 = 0.5 *((point1[0] - u_x)*(point2[1] - u_y)
                - (point1[1] - u_y) * (point2[0] - u_x));

                surface1 = surface1/faceSurface;
                surface2 = surface2/faceSurface;
                surface3 = surface3/faceSurface;                

                if(surface1 < 0 || surface2 < 0 || surface3 < 0)
                {
                    continue;
                }
                else
                {
                    ssize_t xTex = u_x/texelSize;
                    ssize_t yTex = u_y/texelSize;
                    if(((yDim - yTex  - 1) * (xDim) + xTex) < 0 || ((yDim - yTex  - 1) * (xDim) + xTex) > (yDim * xDim))
                    {
                        continue;
                    }

                    // Interpolate point via Barycentric Coordinates
                    // Then find nearest point in the point cloud
                    BaseVecT point = realPoint1 * surface1 + realPoint2 * surface2 + realPoint3 * surface3;
                    if(affineMatrix.size() != 0)
                    {            
                        Eigen::Vector4d p(point[0],point[1],point[2],1);

                        Eigen::Vector4d solution;
                        solution = affineMatrix.inverse() * p;
                        point[0] = solution.coeff(0);
                        point[1] = solution.coeff(1);
                        point[2] = solution.coeff(2);
                    }
                    vector<size_t> cv;  
                    vector<Data> distances;                      
                    
                    BaseVecT pointDist; 
                    pointDist[0] = point[0];
                    pointDist[1] = point[1];
                    pointDist[2] = zMax;                         
                    
                    size_t bestPoint = -1;
                    Data highestZ = zMin;

                    // Search from maximum to minimum height
                    // If we reach minimum height, stop looking --> texel is left blank
                    do
                    {
                        if(pointDist[2] <= zMin)
                        {
                            break;
                        }

                        cv.clear();
                        
                        distances.clear();
                        size_t neighbors = tree.radiusSearch(pointDist, 1000, texelSize, cv, distances);
                        for (size_t j = 0; j < neighbors; j++)
                        {
                            size_t pointIdx = cv[j];
                            auto cp = arr[pointIdx];

                            // The point we are looking for is the one with the highest z value inside the texel range
                            if(cp[2] >= highestZ)
                            {
                                if(sqrt(pow(point[0] - cp[0],2)) <= texelSize/2 && sqrt(pow(point[1] - cp[1],2)) <= texelSize/2)
                                {
                                    highestZ = cp[2];
                                    bestPoint = pointIdx;                                        
                                }
                            }
                        }   
                        // We make small steps so we dont accidentally miss points
                        pointDist[2] -= texelSize/4; 

                    } while(bestPoint == -1);  

                    if(bestPoint == -1)
                    {
                        distance[(yDim - yTex  - 1) * (xDim) + xTex] = std::numeric_limits<Data>::min();
                        continue;
                    }
                    auto p = arr[bestPoint];
                    // We only care about the height difference
                    distance[(yDim - yTex  - 1) * (xDim) + xTex] =  
                    sqrt(pow(point[2] - p[2],2));
                    if(maxDistance < distance[(yDim - yTex  - 1) * (xDim) + xTex])
                    {
                        maxDistance = distance[(yDim - yTex  - 1) * (xDim) + xTex];
                    }   

                    if(minDistance > distance[(yDim - yTex  - 1) * (xDim) + xTex])
                    {
                        minDistance = distance[(yDim - yTex  - 1) * (xDim) + xTex];
                    } 

                }

            }
            
        }
        ++progressDistance;
        ++iterator;
    }  
    std::cout << std::endl;

    // =======================================================================
    // Color the texels according to the recorded height difference
    // =======================================================================
    // color gradient behaves according to the highest distance
    // the jet color gradient is used

    ProgressBar progressColor(xDim * yDim, timestamp.getElapsedTime() + "Setting colors ");     

    ColorMap colorMap(maxDistance - minDistance);
    float color[3];
    GradientType type;
    // Get Color Scale --> Default to JET if not supported
    if(colorScale == "GREY")
    {
        type = GREY;
    }
    else if(colorScale == "JET")
    {
        type = JET;
    }
    else if(colorScale == "HOT")
    {
        type = HOT;
    }
    else if(colorScale == "HSV")
    {
        type = HSV;
    }
    else if(colorScale == "SHSV")
    {
        type = SHSV;
    }
    else if(colorScale == "SIMPSONS")
    {
        type = SIMPSONS;
    }
    else
    {
        type = JET;
    }

    for (int y = 0; y < yDim; y++)
    {
        for (int x = 0; x < xDim; x++)
        {
            if(distance[(yDim - y - 1) * (xDim) + x] == std::numeric_limits<Data>::min())
            {
                texture.m_data[(yDim - y - 1) * (xDim * 3) + x * 3 + 0] = 0;
                texture.m_data[(yDim - y - 1) * (xDim * 3) + x * 3 + 1] = 0;
                texture.m_data[(yDim - y - 1) * (xDim * 3) + x * 3 + 2] = 0;
            }
            else
            {
            colorMap.getColor(color,distance[(yDim - y - 1) * (xDim) + x] - minDistance,type);

            texture.m_data[(yDim - y - 1) * (xDim * 3) + x * 3 + 0] = color[0] * 255;
            texture.m_data[(yDim - y - 1) * (xDim * 3) + x * 3 + 1] = color[1] * 255;
            texture.m_data[(yDim - y - 1) * (xDim * 3) + x * 3 + 2] = color[2] * 255;
            }

            ++progressColor;
        }
    }
    delete distance;
    std::cout << std::endl;
    return texture;
}

/**
 * @brief Extracts one or three Bands from a GeoTIFF and writes the data into a Texture. 
 * One band is depicted in a colour scale of choice and three bands are interpreted as RGB.
 * 
 * @param io Contains the GeoTIFF's data.
 * @param firstBand The first band of the GeoTIFF to extract.
 * @param lastBand The last band of the GeoTIFF to extract. If this is equal to @firstBand, this band is extracted.
 * @param colorScale The colour scale the band is depicted in: GREY, JET, HOT, HSV, SHSV, SIMPSONS.
 * @return Texture Contains the extracted band's data.
 */
Texture readGeoTIFF(GeoTIFFIO* io, int firstBand, int lastBand, string colorScale)
{
    // =======================================================================
    // Read key Information from the TIFF
    // ======================================================================= 

    GradientType type;
    // Get Color Scale --> Default to JET if not supported
    if(colorScale == "GREY")
    {
        type = GREY;
    }
    else if(colorScale == "JET")
    {
        type = JET;
    }
    else if(colorScale == "HOT")
    {
        type = HOT;
    }
    else if(colorScale == "HSV")
    {
        type = HSV;
    }
    else if(colorScale == "SHSV")
    {
        type = SHSV;
    }
    else if(colorScale == "SIMPSONS")
    {
        type = SIMPSONS;
    }
    else
    {
        type = JET;
    }
    
    int yDimTiff = io->getRasterHeight();
    int xDimTiff = io->getRasterWidth();
    int numBands = io->getNumBands();
    double geoTransform[6];
    io->getGeoTransform(geoTransform);
    int bandRange = lastBand - firstBand + 1;
    float texelSize = geoTransform[1]; 
    // Create Texture with GeoTIFF's resolution
    Texture texture(0, xDimTiff, yDimTiff, 3, 1, texelSize);
    // =======================================================================
    // Insert Band Information into Texture
    // =======================================================================
    if(bandRange == 1)
    {
        cv::Mat *mat = io->readBand(firstBand);
        double noData = io->getNoDataValue(firstBand);
        // Get minimum/maximum of band and remove comma
        int counter = 0;

        // Since faulty GeoTIFFs with no Max/Min exists, this is done manually
        /*float values[2];
        io->getMaxMinOfBand(values,firstBand);
        auto max = values[0];
        auto min = values[1];*/

        double max = std::numeric_limits<double>::min();
        double min = std::numeric_limits<double>::max();
        for (ssize_t y = 0; y < yDimTiff; y++)
        {
            for (ssize_t x = 0; x < xDimTiff; x++)
            {                
                auto n = mat->at<float>((yDimTiff - y - 1) * (xDimTiff) + x);
                if(n == noData)
                {
                    continue;
                }
                if(n >= max)
                {
                    max = n;
                }
                if(n <= min)
                {
                    min = n;
                }
            }              
        }
        int multi = 1;
        if(abs(min) < 1 || abs(max) < 1)
        {
            multi = 1000;
        }
        max = (max-min)/(min+1);
        
        size_t maxV = (size_t)(max*multi);      
        // Build colorMap based on max/min
        ColorMap colorMap(maxV);
        float color[3];
        ProgressBar progressGeoTIFF(yDimTiff* xDimTiff, timestamp.getElapsedTime() + "Extracting GeoTIFF data ");
        for (ssize_t y = 0; y < yDimTiff; y++)
        {
            for (ssize_t x = 0; x < xDimTiff; x++)
            {                
                auto n = mat->at<float>((yDimTiff - y - 1) * (xDimTiff) + x);
                if(n < 0)
                {
                    texture.m_data[(yDimTiff  - y - 1) * (xDimTiff * 3) + x * 3 + 0] = 0;
                    texture.m_data[(yDimTiff  - y - 1) * (xDimTiff * 3) + x * 3 + 1] = 0;
                    texture.m_data[(yDimTiff  - y - 1) * (xDimTiff * 3) + x * 3 + 2] = 0;  
                    ++progressGeoTIFF;
                    continue;
                }
                colorMap.getColor(color, (n-min)/(min+1)*multi,type);
        
                texture.m_data[(yDimTiff  - y - 1) * (xDimTiff * 3) + x * 3 + 0] = color[0] * 255;
                texture.m_data[(yDimTiff  - y - 1) * (xDimTiff * 3) + x * 3 + 1] = color[1] * 255;
                texture.m_data[(yDimTiff  - y - 1) * (xDimTiff * 3) + x * 3 + 2] = color[2] * 255;
                ++progressGeoTIFF;                
            }              
        }
        delete(mat);
            
    }
    else if (bandRange == 3)
    {
        ProgressBar progressGeoTIFF(yDimTiff* xDimTiff*3, timestamp.getElapsedTime() + "Extracting GeoTIFF data ");
        for (int b = firstBand; b <= lastBand; b++)
        {
            cv::Mat *mat = io->readBand(b);
            double noData = io->getNoDataValue(firstBand);
            // Get minimum/maximum of band and find multipler that removes comma
            int counter = 0;
            /*float values[2];
            io->getMaxMinOfBand(values,b);
            
            int multi = 1;
            auto max = values[0];
            auto min = values[1];*/

            double max = std::numeric_limits<double>::min();
            double min = std::numeric_limits<double>::max();
            for (ssize_t y = 0; y < yDimTiff; y++)
            {
                for (ssize_t x = 0; x < xDimTiff; x++)
                {                
                    auto n = mat->at<float>((yDimTiff - y - 1) * (xDimTiff) + x);
                    if(n == noData)
                    {
                        continue;
                    }
                    if(n >= max)
                    {
                        max = n;
                    }
                    if(n <= min)
                    {
                        min = n;
                    }
                }              
            }

            auto dimV = max - min;
            for (ssize_t y = 0; y < yDimTiff; y++)
            {
                for (ssize_t x = 0; x < xDimTiff; x++)
                {          
                    // Calculate RGB Value      
                    auto n = mat->at<float>((yDimTiff - y - 1) * (xDimTiff) + x);
                    n /= dimV;
                    n = round(n*255);
                    if(n < 0 || n > 255)
                    {
                        n = 0;
                    }
                    texture.m_data[(yDimTiff  - y - 1) * (xDimTiff * 3) + x * 3 + (b - 1)] = n; 
                    ++progressGeoTIFF;              
                }              
            }
            delete(mat);
        }
        
    }
    else
    {
        std::cerr << "Wrong Number Of Bands ! Only 1 or 3 Bands are allowed!" << std::endl;
    } 

    std::cout << std::endl;
    return texture; 
}