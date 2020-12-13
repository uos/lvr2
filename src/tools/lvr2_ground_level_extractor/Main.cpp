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
#include <map>
#include <chrono>
#include <ctime>  

#include <boost/optional.hpp>

#include "lvr2/config/lvropenmp.hpp"

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
#include "lvr2/texture/TextureFactory.hpp"
//#include "lvr2/algorithm/ImageTexturizer.hpp"

#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp"
#include "lvr2/reconstruction/BilinearFastBox.hpp"
#include "lvr2/reconstruction/TetraederBox.hpp"
#include "lvr2/reconstruction/FastReconstruction.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/SearchTree.hpp"
#include "lvr2/reconstruction/SearchTreeFlann.hpp"
#include "lvr2/reconstruction/HashGrid.hpp"
#include "lvr2/reconstruction/PointsetGrid.hpp"
#include "lvr2/reconstruction/SharpBox.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/PlutoMapIO.hpp"
#include "lvr2/util/Factories.hpp"
#include "lvr2/algorithm/GeometryAlgorithms.hpp"
#include "lvr2/algorithm/UtilAlgorithms.hpp"
#include "lvr2/registration/KDTree.hpp"
#include "lvr2/display/ColorMap.hpp"


#include "gdal.h"
#include "gdalwarper.h"

#include <Eigen/QR>

#include "lvr2/geometry/BVH.hpp"

#include "lvr2/reconstruction/DMCReconstruction.hpp"

#include "lvr2/io/PLYIO.hpp"
#include "lvr2/io/GeoTIFFIO.hpp"

#include "Options.hpp"

#if defined CUDA_FOUND
    #define GPU_FOUND

    #include "lvr2/reconstruction/cuda/CudaSurface.hpp"

    typedef lvr2::CudaSurface GpuSurface;
#elif defined OPENCL_FOUND
    #define GPU_FOUND

    #include "lvr2/reconstruction/opencl/ClSurface.hpp"
    typedef lvr2::ClSurface GpuSurface;
#endif

using boost::optional;
using std::unique_ptr;
using std::make_unique;

using namespace lvr2;

using Vec = BaseVector<float>;
using VecD = BaseVector<double>;
using PsSurface = lvr2::PointsetSurface<Vec>;

int globalTexIndex = 0;
bool preventTranslation = true;

template <typename BaseVecT>
PointsetSurfacePtr<Vec> loadPointCloud(string data)
{
    // load point cloud data and create adaptiveKSearchSuface
    ModelPtr baseModel = ModelFactory::readModel(data);
    if (!baseModel)
    {
        std::cout << timestamp.getElapsedTime() << "IO Error: Unable to parse " << data << std::endl;
        return nullptr;
    }
    PointBufferPtr baseBuffer = baseModel->m_pointCloud;
    PointsetSurfacePtr<Vec> surface;
    surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(baseBuffer,"FLANN");
    surface->calculateSurfaceNormals();
    return surface;
}

template <typename BaseVecT, typename Data>
Texture generateHeightDifferenceTexture(const PointsetSurface<Vec>& surface ,SearchTreeFlann<BaseVecT>& tree,const lvr2::HalfEdgeMesh<VecD>& mesh, Data texelSize, Eigen::MatrixXd affineMatrix)
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
    if(affineMatrix.size() != 0)
    {
        Eigen::Vector4d pointMax(xMax,yMax,zMax,1);
        Eigen::Vector4d pointMin(xMin,yMin,zMin,1);

        Eigen::Vector4d solution;
        //Remove affine Translation if bool is set
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

    // initialise the texture that will contain the height information
    Texture texture(globalTexIndex++, xDim, yDim, 3, 1, texelSize);

    // contains the distances from each relevant point in the mesh to its closest neighbor
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

    // get the Channel containing the point coordinates
    PointBufferPtr baseBuffer = surface.pointBuffer();   
    FloatChannel arr =  *(baseBuffer->getFloatChannel("points"));   
    MeshHandleIteratorPtr<FaceHandle> iterator = mesh.facesBegin();

    // =======================================================================
    // Iterate over all faces + calculate which Texel they are represented by
    // =======================================================================

    ProgressBar progressDistance(mesh.numFaces(), timestamp.getElapsedTime() + "Calcing distances ");
    
    for (size_t i = 0; i < mesh.numFaces(); i++)
    {
        BaseVecT correct(xMin,yMin,0);
        auto realPoint1 = mesh.getVertexPositionsOfFace(*iterator)[0];
        auto realPoint2 = mesh.getVertexPositionsOfFace(*iterator)[1];
        auto realPoint3 = mesh.getVertexPositionsOfFace(*iterator)[2];

        auto point1 = realPoint1 - correct;
        auto point2 = realPoint2 - correct;
        auto point3 = realPoint3 - correct;
        auto maxX = std::max(point1[0],std::max(point2[0],point3[0]));
        ssize_t fmaxX = (ssize_t)(std::round(maxX));
        auto minX = std::min(point1[0],std::min(point2[0],point3[0]));
        ssize_t fminX = (ssize_t)(std::round(minX));
        auto maxY = std::max(point1[1],std::max(point2[1],point3[1]));
        ssize_t fmaxY = (ssize_t)(std::round(maxY));
        auto minY = std::min(point1[1],std::min(point2[1],point3[1]));
        ssize_t fminY = (ssize_t)(std::round(minY));

        // calculate the faces surface necessary for barycentric coordinate calculation
        Data faceSurface = 0.5 *((point2[0] - point1[0])*(point3[1] - point1[1])
            - (point2[1] - point1[1]) * (point3[0] - point1[0]));

        fminY = std::round(fminY/texelSize);
        fmaxY = std::round(fmaxY/texelSize);
        fminX = std::round(fminX/texelSize);
        fmaxX = std::round(fmaxX/texelSize);
        #pragma omp parallel for collapse(2)
        for (ssize_t y = fminY; y < fmaxY; y++)
        {
            for (ssize_t x = fminX; x < fmaxX; x++)
            {
                // we want the information in the center of the pixel
                Data u_x = x * texelSize + texelSize/2;
                Data u_y = y * texelSize + texelSize/2;

                // check, if this face carries the information for texel xy
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
                    ssize_t xTex = std::round(u_x/texelSize);
                    ssize_t yTex = std::round(u_y/texelSize); 
                    if(((yDim - yTex  - 1) * (xDim) + xTex) < 0 || ((yDim - yTex  - 1) * (xDim) + xTex) > (yDim * xDim))
                    {
                        continue;
                    }

                    // interpolate point
                    // find nearest point in pointcloud
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

                    // mode 2 show the height difference between ground and highest point on one texel                      
                    // search from maximum height
                    BaseVecT pointDist; 
                    pointDist[0] = point[0];
                    pointDist[1] = point[1];
                    pointDist[2] = zMax;     
                    
                    
                    size_t bestPoint = -1;
                    Data highestZ = zMin;

                    // search inside the radius for the closes Point
                    // if this doesn't work, search lower
                    // if we reach minimum height, stop looking --> texel is left blank
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

                            // the point we are looking for is the one with the 
                            // highest z value and the point that is still inside the texel range

                            if(cp[2] >= highestZ)
                            {
                                if(sqrt(pow(point[0] - cp[0],2)) <= texelSize/2 && sqrt(pow(point[1] - cp[1],2)) <= texelSize/2)
                                {
                                    highestZ = cp[2];
                                    bestPoint = pointIdx;                                        
                                }
                            }
                        }   
                        // we make small steps so we dont accidentally miss points, might be too small
                        pointDist[2] -= texelSize/4; 

                    } while(bestPoint == -1);                        
                    if(bestPoint == -1)
                    {
                        distance[(yDim - yTex  - 1) * (xDim) + xTex] = std::numeric_limits<Data>::min();
                        continue;
                    }
                    auto p = arr[bestPoint];
                    // we only care about the height difference
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
            colorMap.getColor(color,distance[(yDim - y - 1) * (xDim) + x] - minDistance,JET);

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
//TODO: Dynamische Matrix abhängig von anzahl an Punkten, src und dest dynamisch einfügen, M = 3 * Anzahl Punkte
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> computeAffineGeoRefMatrix(VecD srcPoints[4], VecD destPoints[4], int numberPoints)
{
    // Create one 12 x 12 Matrix and two 12 x 1 Vector, fill one with created Points
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
    
    /*src << 
    srcPoints[0].x, srcPoints[0].y, srcPoints[0].z, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, srcPoints[0].x, srcPoints[0].y, srcPoints[0].z, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, srcPoints[0].x, srcPoints[0].y, srcPoints[0].z, 1,
    srcPoints[1].x, srcPoints[1].y, srcPoints[1].z, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, srcPoints[1].x, srcPoints[1].y, srcPoints[1].z, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, srcPoints[1].x, srcPoints[1].y, srcPoints[1].z, 1,
    srcPoints[2].x, srcPoints[2].y, srcPoints[2].z, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, srcPoints[2].x, srcPoints[2].y, srcPoints[2].z, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, srcPoints[2].x, srcPoints[2].y, srcPoints[2].z, 1,
    srcPoints[3].x, srcPoints[3].y, srcPoints[3].z, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, srcPoints[3].x, srcPoints[3].y, srcPoints[3].z, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, srcPoints[3].x, srcPoints[3].y, srcPoints[3].z, 1;*/
    
    
    /*dest << 
    destPoints[0].x, destPoints[0].y, destPoints[0].z,
    destPoints[1].x, destPoints[1].y, destPoints[1].z,
    destPoints[2].x, destPoints[2].y, destPoints[2].z,
    destPoints[3].x, destPoints[3].y, destPoints[3].z;
    */
    Eigen::VectorXd affineValues(12);
    affineValues = src.colPivHouseholderQr().solve(dest);
    
    Eigen::MatrixXd affineMatrix(4,4);
   
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
   
    auto fullAffineMatrix = affineTranslation * affineMatrix;    

    return {affineMatrix,fullAffineMatrix};
}

void warpGeoTIFF(GDALDatasetH& src,GDALDatasetH& dt,const std::string& geogCS, const std::string& newTiffName )
{    
    const char *pszSrcWKT = NULL;
    char *pszDstWKT  = NULL;

    //Initialse Driver
    GDALDriverH hDriver = GDALGetDriverByName( "GTiff" );
    CPLAssert( hDriver != NULL );

    //Get Coordinate Information from Source
    pszSrcWKT = GDALGetProjectionRef(src);
    CPLAssert( pszSrcWKT != NULL && strlen(src) > 0 );

    GDALDataType eDT = GDALGetRasterDataType(GDALGetRasterBand(src,1));

    //Create Coordinate Informatiom for Destination    
    OGRSpatialReference oSRS;
    oSRS.SetFromUserInput(geogCS.c_str());
    oSRS.exportToWkt(&pszDstWKT);
    CPLAssert( Og == 0 );

    void *hTransformArg;
    hTransformArg =
        GDALCreateGenImgProjTransformer( src, pszSrcWKT, NULL, pszDstWKT,
                                        FALSE, 0, 1 );
    CPLAssert( hTransformArg != NULL );
    
    // approximated output
    double adfDstGeoTransform[6];
    int nPixels=0, nLines=0;
    CPLErr eErr;
    eErr = GDALSuggestedWarpOutput( src,
                                    GDALGenImgProjTransform, hTransformArg,
                                    adfDstGeoTransform, &nPixels, &nLines );
    
    CPLAssert( eErr == CE_None );
    GDALDestroyGenImgProjTransformer( hTransformArg );
    dt = GDALCreate( hDriver, newTiffName.c_str(), nPixels, nLines,
                        GDALGetRasterCount(src), eDT, NULL );
    CPLAssert( dt != NULL );
    
    
    GDALSetProjection( dt, pszDstWKT );
    GDALSetGeoTransform( dt, adfDstGeoTransform );  

    //Set No-Data Value and Color Info
    
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

    //Warp Image
    GDALWarpOptions *psWarpOptions = GDALCreateWarpOptions();
    psWarpOptions->hSrcDS = src;
    psWarpOptions->hDstDS = dt;
    psWarpOptions->nBandCount = 0;
    psWarpOptions->pfnProgress = GDALTermProgress;
    psWarpOptions->papszWarpOptions = 
    CSLSetNameValue(psWarpOptions->papszWarpOptions,"OPTIMIZE_SIZE","TRUE");

    //reprojections transformer
    psWarpOptions->pTransformerArg =
        GDALCreateGenImgProjTransformer( src,
                                        GDALGetProjectionRef(src),
                                        dt,
                                        GDALGetProjectionRef(dt),
                                        FALSE, 0.0, 1 );
    psWarpOptions->pfnTransformer = GDALGenImgProjTransform;

    //execute
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

template <typename BaseVecT>
void transformPoints(string src,string dt, BaseVecT srcPoints[4], BaseVecT destPoints[4])
{
    GDALAllRegister();
    OGRSpatialReference source, target;
    source.SetFromUserInput(src.c_str());
    std::cout << std::endl;
    target.SetFromUserInput(dt.c_str());
    OGRPoint p;
    
    for(int i = 0; i < 4; i++)
    {    
        p.assignSpatialReference(&source);  
        p.setX(srcPoints[i].x);
        p.setY(srcPoints[i].y);
        p.setZ(srcPoints[i].z);
        p.transformTo(&target); 
        
        destPoints[i] = BaseVecT(p.getX(),p.getY(),p.getZ());
    }

}

Texture readGeoTIFF(GeoTIFFIO* io, int firstBand, int lastBand)
{
    // =======================================================================
    // Read key Information from the TIFF
    // ======================================================================= 
    
    int yDimTiff = io->getRasterHeight();
    int xDimTiff = io->getRasterWidth();
    int numBands = io->getNumBands();
    double geoTransform[6];
    io->getGeoTransform(geoTransform);

    int bandRange = lastBand - firstBand + 1;
    float texelSize = geoTransform[1]; 
    
    Texture texture(globalTexIndex++, xDimTiff, yDimTiff, 3, 1, texelSize);

    // =======================================================================
    // Insert Band Information into Texture
    // =======================================================================
    if(bandRange == 1)
    {
        cv::Mat *mat = io->readBand(firstBand);
        // get minimum/maximum of band and remove comma
        int counter = 0;
        float values[2];
        io->getMaxMinOfBand(values,firstBand);
        
        auto max = values[0];
        auto min = values[1];
        int multi = 1;
        if(abs(min) < 1 || abs(max) < 1)
        {
            multi = 1000;
        }
        max = (max-min)/(min+1);
        
        size_t maxV = (size_t)(max*multi);      
        // build colorMap based on max/min
        ColorMap colorMap(maxV);
        float color[3];

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
                    continue;
                }
                colorMap.getColor(color, (n-min)/(min+1)*multi,JET);
        
                texture.m_data[(yDimTiff  - y - 1) * (xDimTiff * 3) + x * 3 + 0] = color[0] * 255;
                texture.m_data[(yDimTiff  - y - 1) * (xDimTiff * 3) + x * 3 + 1] = color[1] * 255;
                texture.m_data[(yDimTiff  - y - 1) * (xDimTiff * 3) + x * 3 + 2] = color[2] * 255;                
            }              
        }
        delete(mat);
            
    }
    else if (bandRange == 3)
    {
        for (int b = firstBand; b <= lastBand; b++)
        {
            cv::Mat *mat = io->readBand(b);
            // get minimum/maximum of band and find multipler that removes comma
            int counter = 0;
            float values[2];
            io->getMaxMinOfBand(values,b);
            
            int multi = 1;
            auto max = values[0];
            auto min = values[1];

            auto dimV = max - min;

            for (ssize_t y = 0; y < yDimTiff; y++)
            {
                for (ssize_t x = 0; x < xDimTiff; x++)
                {                
                    auto n = mat->at<float>((yDimTiff - y - 1) * (xDimTiff) + x);
                    n /= dimV;
                    n = round(n*255);
                    if(n < 0 || n > 255)
                    {
                        n = 0;
                    }
                    texture.m_data[(yDimTiff  - y - 1) * (xDimTiff * 3) + x * 3 + (b - 1)] = n;               
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

template<typename BaseVecT>
MaterializerResult<BaseVecT> projectTexture(const lvr2::HalfEdgeMesh<BaseVecT>& mesh, const ClusterBiMap<FaceHandle>& clusters, const PointsetSurface<Vec>& surface, 
float texelSize, Eigen::MatrixXd affineMatrix, Eigen::MatrixXd fullAffineMatrix, GeoTIFFIO* io,SearchTreeFlann<BaseVecT>& tree)
{
    // =======================================================================
    // Prepare necessary preconditions to create MaterializerResult
    // =======================================================================
    DenseClusterMap<Material> clusterMaterials;
    SparseVertexMap<ClusterTexCoordMapping> vertexTexCoords;
    // the keypoint_map is never utilised in the finalizer and will be ignored henceforth
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
        if(io)
        {            
            tex = readGeoTIFF(io,1,3);             
        }
        else
        {
            tex = generateHeightDifferenceTexture<VecD,double>(surface,tree,mesh,texelSize,affineMatrix);
        }     

        //Rotates the extreme Values to fit the Texture
        if(affineMatrix.size() != 0)
        {
            Eigen::Vector4d pointMax(xMax,yMax,zMax,1);
            Eigen::Vector4d pointMin(xMin,yMin,zMin,1);

            Eigen::Vector4d solution;
            //Remove affine Translation if bool is set
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
        // correct coordinates
        BaseVecT correct(xMin,yMin,0);    

        // Code copied from Materializer.tcc; this part essentially does what the materializer does
        // save Texture as Material so it can be correctly generated by the finalizer
        Material material;
        material.m_texture = textures.push(tex);
        
        std::array<unsigned char, 3> arr = {255, 255, 255};
        
        material.m_color = std::move(arr);            
        clusterMaterials.insert(clusterH, material);

        std::unordered_set<VertexHandle> clusterVertices;
        // get a set of all unique vertices
        for (auto faceH : cluster.handles)
        {
            for (auto vertexH : mesh.getVerticesOfFace(faceH))
            {
                clusterVertices.insert(vertexH);
            }
        }        

        // calculate the Texture Coordinates for all Vertices
        for (auto vertexH : clusterVertices)
        {            
            auto pos = mesh.getVertexPosition(vertexH);
            // correct coordinates
            
            float yPixel = 0;
            float xPixel = 0;
            
            if(io)
            {
                double geoTransform[6];
                int y_dim_tiff = io->getRasterHeight();
                int x_dim_tiff = io->getRasterWidth();
                float values[2];
                io->getMaxMinOfBand(values,1);
                io->getGeoTransform(geoTransform);   
                if(preventTranslation) 
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

template <typename Data>
Data weight(Data distance)
{    
    // calculates inverted distance
    return 1/distance;                
}

template <typename BaseVecT, typename Data>
Data findLowestZ(Data x, Data y, Data lowestZ, Data highestZ, Data searchArea, SearchTreeFlann<BaseVecT>& tree,FloatChannel& points)
{    
    Data bestZ = highestZ;
    Data currentZ = lowestZ;
    bool found = false;

    // =======================================================================
    // Look for Point with the Lowest height in the x/y coordinate
    // =======================================================================
    // utilises radiusSearch
    do
    {
        vector<size_t> neighbors;  
        vector<Data> distances;        
        // look for the closest point whith a z-value lower then our currently best point
        // we increse the radius so we have the whole area the node is affected by covered
        // and later check if the nodes found are inside the square
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
            //Set the grid points height according to its nearest neighbors
            
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
    
    int counter = 0;
    
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
            // Small Window Thresh holding
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

            // compare lowest z with points z
            // if height differen biggern then smallWindowHeight 
            // the point does not belong to the surface area
            if(abs(workGrid[x][y] - lowestDist) > smallWindowHeight)
            {
                ++progressPoints;
                continue;
            }

            // =======================================================================
            // Slope Thresh holding
            // =======================================================================
            // Calculates the Slope between the observed point and its neighbors
            // if the Slope exerts a threshhold, the point is a non-ground point
            bool slopeGood = true;
            
            for (int xwin = -1; xwin <= 1; xwin++)
            {
                for (int ywin = -1; ywin <= 0; ywin++)
                {
                    //Only use points that are we have set b4
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
           
            // Similar too Small Window Thresh holding; used to elimnate large Objects like Trees

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
            
            // if the point passes through the three tests, it gets recoginised as ground point
            double v_x = u_x+xMin;
            double v_y = u_y+yMin;
            double v_z = workGrid[x][y];
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
            counter++;
        }
        
    }

    std::cout << std::endl;

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

template <typename BaseVecT, typename Data>
void nearestNeighborMethod(lvr2::HalfEdgeMesh<VecD>& mesh, FloatChannel& points, PointsetSurfacePtr<Vec>& surface,
SearchTreeFlann<BaseVecT>& tree ,int numNeighbors, Data stepSize, Eigen::MatrixXd& affineMatrix)
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

    // lists used when constructing
    vector<size_t> indices;
    vector<Data> distances;
    Data finalZ = 0;         
    Data avgDistance = 0;
    int trustedNeighbors = 0;

    int numberNeighbors = numNeighbors;

    xMin = xMin * (1/stepSize);
    xMax = xMax * (1/stepSize);
    yMin = yMin * (1/stepSize);
    yMax = yMax * (1/stepSize);      
    
    avgDistance = stepSize;
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle> dict1;
    
    // =======================================================================
    // Calculate Vertice Height and create Hexagonial Net
    // =======================================================================
    ProgressBar progressVert((xDim / stepSize)*(yDim / stepSize), timestamp.getElapsedTime() + "Calculating Grid");
    for (ssize_t x = xMin; x < xMax; x++)
    {        
        for (ssize_t y = yMin; y < yMax; y++)
        {               
            Data u_x = x * stepSize;
            Data u_y = y * stepSize;
            
            indices.clear();
            distances.clear();

            finalZ = 0;  
            Data closeZ = findLowestZ<BaseVecT,Data>(u_x,u_y,zMin,zMax,stepSize/2,tree,points);
            if(closeZ == std::numeric_limits<Data>::max())
            {
                ++progressVert;
                continue;
            }              
            tree.kSearch(BaseVecT(u_x,u_y,closeZ),numberNeighbors,indices,distances);
            
            trustedNeighbors = numberNeighbors;
            for (int i = 0; i < numberNeighbors; i++)
            {
                if(distances[i] > avgDistance)
                {
                    trustedNeighbors--;
                }
                else
                {
                    auto index = indices[i];
                    auto nearest = points[index];
                    finalZ = finalZ + nearest[2];
                }
            }

            // if the center vertice isn't trustworthy, we can't complete
            // any triangle and thus skip the others
            if(trustedNeighbors < numberNeighbors)
            {
                ++progressVert;
                continue;
            }  
            else
            {
                finalZ = finalZ/trustedNeighbors;
            }

            double d_x = u_x;
            double d_y = u_y;
            double d_z = finalZ;

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

template <typename BaseVecT, typename Data>
void improvedMovingAverage(lvr2::HalfEdgeMesh<VecD>& mesh, FloatChannel& points, PointsetSurfacePtr<Vec>& surface,
SearchTreeFlann<BaseVecT>& tree, float minRadius, float maxRadius, int minNeighbors, int maxNeighbors, int radiusSteps, float stepSize, Eigen::MatrixXd& affineMatrix )
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

    xMin = xMin * (1/stepSize);
    xMax = xMax * (1/stepSize);
    yMin = yMin * (1/stepSize);
    yMax = yMax * (1/stepSize);

    Data finalZ = 0;  
    Data addedDistance = 0;
    vector<size_t> indices;  
    vector<Data> distances;  

    // calculate the radius step size
    float radiusStepsize = (maxRadius - minRadius)/radiusSteps;
    float radius = 0;

    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle> dict;

    ProgressBar progressVert((xDim/stepSize)*(yDim/stepSize), timestamp.getElapsedTime() + "Calculating z values "); 

    // =======================================================================
    // Calculate Vertice Height and create Hexagonial Net
    // =======================================================================
    for (ssize_t x = xMin; x < xMax; x++)
    {        
        for (ssize_t y = yMin; y < yMax; y++)
        {           
              
            Data u_x = x * stepSize;
            Data u_y = y * stepSize;

            BaseVecT point;
            
            found = 0;
            finalZ = 0;  
            addedDistance = 0;

            indices.clear();
            distances.clear(); 

            radius = minRadius;
            Data u_z = findLowestZ<BaseVecT,Data>(u_x,u_y,zMin,zMax,stepSize/2,tree,points);
            if(u_z == std::numeric_limits<Data>::max())
            {
                ++progressVert; 
                continue;
            }
            point = BaseVecT(u_x,u_y,u_z);
            // if we don't find enough points in the current radius, we extend the radius
            // if we hit the maximum extension and still find nothing, the point is left blank
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

            if(found == 1)
            {
                for (int i = 0; i < numberNeighbors; i++)
                {
                    size_t pointIdx = indices[i];
                    auto neighbor = points[pointIdx];
                    // when we are exactly on the point, distance is 0 and would divide by 0
                    Data distance = 1;
                    if(distances[i] != 0)
                    {
                        distance = weight<Data>(distances[i]); 
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

            double d_x = u_x;
            double d_y = u_y;
            double d_z = finalZ;

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

int main(int argc, char* argv[])
{  
    // =======================================================================
    // Parse command line parameters
    // =======================================================================
    std::cout << std::fixed;
    lvr2::HalfEdgeMesh<VecD> mesh;

    ground_level_extractor::Options options(argc,argv);

    options.printLogo();

    if (options.printUsage())
    {
        return 0;
    }

    std::cout << options << std::endl;

    //TODO: Falsche Anzahl an Bändern (nicht 1 oder 3) abfangen    

    // =======================================================================
    // Load Pointcloud and create Model + Surface + SearchTree
    // =======================================================================

    ModelPtr baseModel = ModelFactory::readModel(options.getInputFileName());
    if (!baseModel)
    {
        std::cout << timestamp.getElapsedTime() << "IO Error: Unable to parse " << options.getInputFileName() << std::endl;
        return 0;
    }
    auto surface = loadPointCloud<Vec>(options.getInputFileName());    
    PointBufferPtr baseBuffer = baseModel->m_pointCloud;
    auto tree = SearchTreeFlann<VecD> (baseBuffer);
    // get the pointcloud coordinates from the FloatChannel
    FloatChannel arr =  *(baseBuffer->getFloatChannel("points"));   

    PointsetSurfacePtr<Vec> usedSurface = surface;
    FloatChannel usedArr = arr;   
    
    float resolution = options.getResolution();
    float texelSize = resolution/2; 
    
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

    //TODO:abhängig von eingaben setzen
    bool targetDest = false;
    bool gTiff = false;
    string newTiffName = "out.tif";

    //TODO: aus input extrahieren
    string typ1;

    if(!options.getInputReferencePairs().empty())
    {
        ifstream input;
        input.open(options.getInputReferencePairs());
        if(input.fail())
        {
            std::cout << timestamp.getElapsedTime() << "IO Error: Unable to read " << options.getExtractionMethod() << std::endl;
            return 0;
        }

        std::getline(input, typ1);
        VecD s,d;
        char ch;
        for(int i = 0; i < 4; i++){
            input >> s.x >> ch >> s.y >> ch >> s.z;
            srcPoints[i] = s;
            input >> d.x >> ch >> d.y >> ch >> d.z;
            dstPoints[i] = d;
        }       
        
    }
    
    // =======================================================================
    // Read GeoTIFF + ReffPoints and transform
    // =======================================================================
    GeoTIFFIO* io = NULL;
    GDALDataset* set;
    //TODO: gucken ob file existiert
    //"/home/mario/Schreibtisch/field_scans/ortho_austausch/20200807_hs_blang_rgb_ortho.tif";
    //"/home/mario/Schreibtisch/field_scans/ortho_austausch/20200807_hs_blang_multi_ortho_refl.tif";
    //"/home/mario/BA/UpToDatest/Develop/build/20200807_hs_blang_multi_ortho_refl.tif";
    if(!options.getTargetSystem().empty())
    {
        if(!options.getInputReferencePairs().empty())
        {
            //TODO:typ1 extrahiert auf refferenzpunkten
            string typ2 = options.getTargetSystem();
            VecD dstP[4];

            transformPoints(typ1,typ2,dstPoints,dstP);

            std::copy(std::begin(dstP),std::end(dstP),std::begin(dstPoints));

            if(!options.getInputGeoTIFF().empty())
            {
                GDALDatasetH src = GDALOpen(options.getInputGeoTIFF().c_str(),GA_ReadOnly);
                GDALDatasetH dt;
                // creates a new GeoTIFF file with the transformed info of the old one
                warpGeoTIFF(src,dt,typ2,newTiffName);
            }
        }
    
    }

    // =======================================================================
    // Compute Affine Transform Matrix from Transformed Reff Points
    // =======================================================================
    Eigen::MatrixXd affineMatrix, fullAffineMatrix;
    if(!options.getInputReferencePairs().empty())
    {    
        tie(affineMatrix,fullAffineMatrix) = computeAffineGeoRefMatrix(srcPoints,dstPoints,4);
        // Right now, LVR2 doesn't support Large Coordinates and we can't use the Translation fully
        // In Functions where we use the Matrix we need to exclude the Translation
        if(!options.getInputGeoTIFF().empty())
        {
            if(targetDest)
            {                
                io = new GeoTIFFIO(newTiffName);
            }
            else
            {
                io = new GeoTIFFIO(options.getInputGeoTIFF());
            }
        }      
    } 
    // =======================================================================
    // Extract ground from the point cloud
    // =======================================================================
    //TODO: daten aus input hier einfügen
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
    // creating a cluster map made up of one cluster is necessary to use the finalizer   
    ClusterBiMap<FaceHandle> clusterBiMap;
    MeshHandleIteratorPtr<FaceHandle> iterator = mesh.facesBegin();
    auto newCluster = clusterBiMap.createCluster();
    for (size_t i = 0; i < mesh.numFaces(); i++)
    { 
        clusterBiMap.addToCluster(newCluster,*iterator);

        ++iterator;
    }  
    TextureFinalizer<VecD> finalize(clusterBiMap);
    //TODO: add option for choosing which color scale to use
    //TODO: eingabe von welchen bändern gelsen werden sollen
    auto matResult = projectTexture<VecD>(mesh,clusterBiMap,*usedSurface,texelSize,affineMatrix,fullAffineMatrix,io,tree);
    finalize.setMaterializerResult(matResult);  
    auto buffer = finalize.apply(mesh);
    buffer->addIntAtomic(1, "mesh_save_textures");
    buffer->addIntAtomic(1, "mesh_texture_image_extension");
    std::cout << timestamp.getElapsedTime() << " Setting Model" << std::endl;
    auto m = ModelPtr(new Model(buffer)); 
    //TODO: make filenames less messy
    //TODO: ouput nach vorgabe und in eigenem Ordner
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
    auto time = oss.str();

    // =======================================================================
    // Export Files as PLY and OBJ with a JPEG as Texture
    // =======================================================================    
    std::cout << timestamp.getElapsedTime() << "Saving Model as ply" << std::endl;
    ModelFactory::saveModel(m,time +"groundextraction" + ".ply");

    std::cout << timestamp.getElapsedTime() << "Saving Model as obj" << std::endl;
    ModelFactory::saveModel(m,time +"groundextraction" + ".obj");  

    ofstream file;
    file.open ("transformmatrix.txt");
    file << affineMatrix << "\n" << fullAffineMatrix;
    file.close();

    delete(io);
    return 0;
}