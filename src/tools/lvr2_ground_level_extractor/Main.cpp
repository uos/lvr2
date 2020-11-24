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
PointsetSurfacePtr<Vec> loadPointCloud(string &data)
{
    ModelPtr base_model = ModelFactory::readModel(data);
    PointBufferPtr base_buffer = base_model->m_pointCloud;
    PointsetSurfacePtr<Vec> surface;
    surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(base_buffer,"FLANN");
    surface->calculateSurfaceNormals();
    return surface;

}

//creates a surface for K search
template <typename BaseVecT>
PointsetSurfacePtr<Vec> rebuildPointCloud(PointBufferPtr &base_buffer)
{
    PointsetSurfacePtr<Vec> surface;
    surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(base_buffer,"FLANN");
    surface->calculateSurfaceNormals();
    return surface;

}

//TODO: rename this
//FIXME: new way of calculating moves out grid by 0.5, is this bad in any way when calculating the texture?
template <typename BaseVecT, typename Data>
Texture generateHeightDifferenceTexture(const PointsetSurface<Vec>& surface ,const lvr2::HalfEdgeMesh<VecD>& mesh, Data texelSize, int mode)
{
    // =======================================================================
    // Generate Bounding Box and set up Variables
    // =======================================================================
    auto bb = surface.getBoundingBox();
    auto max = bb.getMax();
    auto min = bb.getMin();
    
    ssize_t x_max = (ssize_t)max.x + 1;
    ssize_t x_min = (ssize_t)min.x - 1;
    ssize_t y_max = (ssize_t)max.y + 1;
    ssize_t y_min = (ssize_t)min.y - 1; 
    ssize_t x_dim = (abs(x_max) + abs(x_min))/texelSize; 
    ssize_t y_dim = (abs(y_max) + abs(y_min))/texelSize; 

    ssize_t z_max = (ssize_t)max.z + 1;
    ssize_t z_min = (ssize_t)min.z - 1;

    ssize_t z_mid = (z_max - z_min)/2 + z_min;
    
    // initialise the texture that will contain the height information
    Texture texture(globalTexIndex++, x_dim, y_dim, 3, 1, texelSize);

    // contains the distances from each relevant point in the mesh to its closest neighbor
    Data* distance = new Data[x_dim * y_dim];
    Data max_distance = 0;
    Data min_distance = std::numeric_limits<Data>::max();
    int counter = 0;

    // Initialise every distance 
    for (int y = 0; y < y_dim; y++)
    {
        for (int x = 0; x < x_dim; x++)
        {
            distance[(y_dim - y - 1) * (x_dim) + x] = std::numeric_limits<Data>::min();
        }
    }

    // get the Channel containing the point coordinates
    PointBufferPtr base_buffer = surface.pointBuffer();   
    FloatChannel arr =  *(base_buffer->getFloatChannel("points"));   
    MeshHandleIteratorPtr<FaceHandle> iterator = mesh.facesBegin();

    // =======================================================================
    // Iterate over all faces + calculate which Texel they are represented by
    // =======================================================================

    ProgressBar progress_distance(mesh.numFaces(), timestamp.getElapsedTime() + "Calcing distances ");
    
    for (size_t i = 0; i < mesh.numFaces(); i++)
    {
        BaseVecT correct(x_min,y_min,0);
        auto real_point1 = mesh.getVertexPositionsOfFace(*iterator)[0];
        auto real_point2 = mesh.getVertexPositionsOfFace(*iterator)[1];
        auto real_point3 = mesh.getVertexPositionsOfFace(*iterator)[2];

        auto point1 = real_point1 - correct;
        auto point2 = real_point2 - correct;
        auto point3 = real_point3 - correct;
        //TODO: change to round
        auto max_x = std::max(point1[0],std::max(point2[0],point3[0]));
        ssize_t fmax_x = (ssize_t)max_x + 1;
        auto min_x = std::min(point1[0],std::min(point2[0],point3[0]));
        ssize_t fmin_x = (ssize_t)min_x - 1;
        auto max_y = std::max(point1[1],std::max(point2[1],point3[1]));
        ssize_t fmax_y = (ssize_t)max_y + 1;
        auto min_y = std::min(point1[1],std::min(point2[1],point3[1]));
        ssize_t fmin_y = (ssize_t)min_y - 1;

        // calculate the faces surface; needed for barycentric coordinate calculation
        Data face_surface = 0.5 *((point2[0] - point1[0])*(point3[1] - point1[1])
            - (point2[1] - point1[1]) * (point3[0] - point1[0]));

        fmin_y = fmin_y * (1/texelSize);
        fmax_y = fmax_y * (1/texelSize);
        fmin_x = fmin_x * (1/texelSize);
        fmax_x = fmax_x * (1/texelSize);
        

        #pragma omp parallel for collapse(2)
        for (ssize_t y = fmin_y; y < fmax_y; y++)
        {
            for (ssize_t x = fmin_x; x < fmax_x; x++)
            {
                // we want the information in the center of the pixel
                Data u_x = x * texelSize + texelSize/2;
                Data u_y = y * texelSize + texelSize/2;

                // check, if this face carries the information for texel xy
                Data surface_1 = 0.5 *((point2[0] - u_x)*(point3[1] - u_y)
                - (point2[1] - u_y) * (point3[0] - u_x));

                Data surface_2 = 0.5 *((point3[0] - u_x)*(point1[1] - u_y)
                - (point3[1] - u_y) * (point1[0] - u_x));

                Data surface_3 = 0.5 *((point1[0] - u_x)*(point2[1] - u_y)
                - (point1[1] - u_y) * (point2[0] - u_x));

                surface_1 = surface_1/face_surface;
                surface_2 = surface_2/face_surface;
                surface_3 = surface_3/face_surface;                

                if(surface_1 < 0 || surface_2 < 0 || surface_3 < 0)
                {
                    continue;
                }
                else
                {
                    ssize_t x_tex = (ssize_t)(u_x/texelSize);
                    ssize_t y_tex = (ssize_t)(u_y/texelSize);
                    //FIXME: Is this still the case?
                    // due to me being dumb, this workaround is necesarry
                    // y_tex can become -1, because y can hit -0.5, because, when creating the mesh
                    // we go over every coordinate and add +0.5 and -0.5
                    // i either have to correct the boundaries or catch it here
                    if((y_dim * x_dim * 3 * 1) < ((y_dim - y_tex  - 1) * (x_dim * 3) + 3 * x_tex  + 0)){
                        continue;
                    }     

                    // interpolate point
                    // find nearest point in pointcloud      

                    BaseVecT point = real_point1 * surface_1 + real_point2 * surface_2 + real_point3 * surface_3;
                    vector<size_t> cv;  
                    vector<float> distances;   

                    // mode 1 shows the difference between original pc and the mesh
                    /*if(mode == 1)
                    {                        
                        surface.searchTree()->kSearch(point, 1, cv);
                        // TODO: if I still want to use this, i should use the distance instead of calculating it manually
                        // to make a dynamic gradient, we save all of the points we want to colorize and their distance
                        // after knowing the maximum distance, we can color the texture accordingly
                        for (size_t pointIdx : cv)
                        {
                            auto cp = arr[pointIdx];
                            distance[(y_dim - y_tex  - 1) * (x_dim) + x_tex] =  
                            sqrt(pow(point[0] - cp[0],2) + pow(point[1] - cp[1],2) + pow(point[2] - cp[2],2)),point[2] - cp[2];                        
                            
                            if(max_distance < distance[(y_dim - y_tex  - 1) * (x_dim) + x_tex])
                            {
                                max_distance= distance[(y_dim - y_tex  - 1) * (x_dim) + x_tex];
                            }    

                            if(min_distance > distance[(y_dim - y_tex  - 1) * (x_dim) + x_tex])
                            {
                                min_distance = distance[(y_dim - y_tex  - 1) * (x_dim) + x_tex];
                            }                 
                            
                        }
                    }

                    //TODO: find a word for "Bestandskarte"
                    // mode 2 show the height difference between ground and highest point on one texel
                    else if(mode == 2)
                    { */                       
                        // search from maximum height
                        Vec point_dist; 
                        //sadly there is no other solution to this right now, and i need to convert double to float
                        point_dist[0] = point[0];
                        point_dist[1] = point[1];
                        point_dist[2] = z_max;
                       
                        size_t best_point = -1;
                        Data highest_z = z_min;

                        // search inside the radius for the closes Point
                        // if this doesn't work, search lower
                        // if we reach minimum height, stop looking --> texel is left blank
                        do
                        {
                            if(point_dist[2] <= z_min)
                            {
                                break;
                            }

                            cv.clear();
                            
                            distances.clear();
                            size_t neighbors = surface.searchTree()->radiusSearch(point_dist, 1000, texelSize, cv, distances);
                            for (size_t j = 0; j < neighbors; j++)
                            {
                                size_t pointIdx = cv[j];
                                auto cp = arr[pointIdx];

                                // the point we are looking for is the one with the 
                                // highest z value and the point that is still inside the texel range

                                if(cp[2] >= highest_z)
                                {
                                    if(sqrt(pow(point[0] - cp[0],2)) <= texelSize/2 && sqrt(pow(point[1] - cp[1],2)) <= texelSize/2)
                                    {
                                        highest_z = cp[2];
                                        best_point = pointIdx;                                        
                                    }
                                }
                            }   
                            // we make small steps so we dont accidentally miss points
                            //FIXME: make this grow faster
                            point_dist[2] -= texelSize/4; 

                        } while(best_point == -1);                        
                        
                        if(best_point == -1)
                        {
                            distance[(y_dim - y_tex  - 1) * (x_dim) + x_tex] = std::numeric_limits<Data>::min();
                            continue;
                        }

                        auto p = arr[best_point];
                        // since we only care about the height difference, x and y coordinate are ignored
                        distance[(y_dim - y_tex  - 1) * (x_dim) + x_tex] =  
                        sqrt(pow(point[2] - p[2],2));

                        if(max_distance < distance[(y_dim - y_tex  - 1) * (x_dim) + x_tex])
                        {
                            max_distance = distance[(y_dim - y_tex  - 1) * (x_dim) + x_tex];
                        }   

                        if(min_distance > distance[(y_dim - y_tex  - 1) * (x_dim) + x_tex])
                        {
                            min_distance = distance[(y_dim - y_tex  - 1) * (x_dim) + x_tex];
                        } 
                    //}

                }

            }
            
        }
        ++progress_distance;
        ++iterator;
    }  
    std::cout << std::endl;

    // =======================================================================
    // Color the texels according to the recorded height difference
    // =======================================================================
    // color gradient behaves according to the highest distance
    // the jet color gradient is used

    ProgressBar progress_color(x_dim * y_dim, timestamp.getElapsedTime() + "Setting colors ");     

    ColorMap colorMap(max_distance - min_distance);
    float color[3];

    for (int y = 0; y < y_dim; y++)
    {
        for (int x = 0; x < x_dim; x++)
        {
            if(distance[(y_dim - y - 1) * (x_dim) + x] == std::numeric_limits<Data>::min())
            {
                texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 0] = 0;
                texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 1] = 0;
                texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 2] = 0;
            }
            else
            {
            colorMap.getColor(color,distance[(y_dim - y - 1) * (x_dim) + x] - min_distance,JET);

            texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 0] = color[0] * 255;
            texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 1] = color[1] * 255;
            texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 2] = color[2] * 255;
            }

            ++progress_color;
        }
    }
    delete distance;
    std::cout << std::endl;
    return texture;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> computeAffineGeoRefMatrix(VecD srcPoints[4], VecD destPoints[4])
{
    ///Create one 12 x 12 Matrix and two 12 x 1 Vector, fill one with created Points
    Eigen::MatrixXd src(12,12);
    src << 
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
    0, 0, 0, 0, 0, 0, 0, 0, srcPoints[3].x, srcPoints[3].y, srcPoints[3].z, 1;
    
    Eigen::VectorXd dest(12);
    dest << 
    destPoints[0].x, destPoints[0].y, destPoints[0].z,
    destPoints[1].x, destPoints[1].y, destPoints[1].z,
    destPoints[2].x, destPoints[2].y, destPoints[2].z,
    destPoints[3].x, destPoints[3].y, destPoints[3].z;
    
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
    //std::cout << affineTranslation << std::endl;
   
    auto fullAffineMatrix = affineTranslation * affineMatrix;
    

    return {affineMatrix,fullAffineMatrix};
}

void warpGeoTIFF(GDALDatasetH& src,GDALDatasetH& dt,const std::string& geogCS, const std::string& newTiffName )
{    
    const char *pszSrcWKT = NULL;
    char *pszDstWKT  = NULL;

    ///Initialse Driver
    GDALDriverH hDriver = GDALGetDriverByName( "GTiff" );
    CPLAssert( hDriver != NULL );

    ///Get Coordinate Information from Source
    pszSrcWKT = GDALGetProjectionRef(src);
    CPLAssert( pszSrcWKT != NULL && strlen(src) > 0 );

    GDALDataType eDT = GDALGetRasterDataType(GDALGetRasterBand(src,1));

    ///Create Coordinate Informatiom for Destination    
    OGRSpatialReference oSRS;
    oSRS.SetFromUserInput(geogCS.c_str());
    oSRS.exportToWkt(&pszDstWKT);
    CPLAssert( Og == 0 );

    void *hTransformArg;
    hTransformArg =
        GDALCreateGenImgProjTransformer( src, pszSrcWKT, NULL, pszDstWKT,
                                        FALSE, 0, 1 );
    CPLAssert( hTransformArg != NULL );
    
    /// approximated output
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

    ///Set No-Data Value and Color Info
    
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

    ///Warp Image
    GDALWarpOptions *psWarpOptions = GDALCreateWarpOptions();
    psWarpOptions->hSrcDS = src;
    psWarpOptions->hDstDS = dt;
    psWarpOptions->nBandCount = 0;
    psWarpOptions->pfnProgress = GDALTermProgress;
    psWarpOptions->papszWarpOptions = 
    CSLSetNameValue(psWarpOptions->papszWarpOptions,"OPTIMIZE_SIZE","TRUE");

    ///reprojections transformer
    psWarpOptions->pTransformerArg =
        GDALCreateGenImgProjTransformer( src,
                                        GDALGetProjectionRef(src),
                                        dt,
                                        GDALGetProjectionRef(dt),
                                        FALSE, 0.0, 1 );
    psWarpOptions->pfnTransformer = GDALGenImgProjTransform;

    ///execute
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
        //std::cout << p.getX() << " " << p.getY() << " " << p.getZ() << std::endl;  
        
        destPoints[i] = BaseVecT(p.getX(),p.getY(),p.getZ());
    }

}



Texture readGeoTIFF(GeoTIFFIO* io, int firstBand, int lastBand)
{
    // =======================================================================
    // Read key Information from the TIFF
    // =======================================================================

    //GeoTIFFIO io(filename);   
    
    int yDimTiff = io->getRasterHeight();
    int xDimTiff = io->getRasterWidth();
    int numBands = io->getNumBands();
    double geoTransform[6];
    io->getGeoTransform(geoTransform);

    int bandRange = lastBand - firstBand + 1;
    float texelSize = geoTransform[1];  
 
    
    Texture texture(globalTexIndex++, xDimTiff, yDimTiff, 3, 1, texelSize);
    //std::cout << timestamp.getElapsedTime() << "Num. Bands: " << numBands << " x_dim :" << x_dim << " y_dim: "
    //<< y_dim << " texel size: " << texelSize << std::endl;

    // =======================================================================
    // Insert Band Information into Texture
    // =======================================================================
    //TODO: change so we go through all the chosen bands
    if(bandRange == 1)
    {
        cv::Mat *mat = io->readBand(firstBand);
        //get minimum/maximum of band and find multipler that removes comma
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
        
        std::cout << min << std::endl;
        std::cout << max << std::endl;
        max = (max-min)/(min+1);
        std::cout << max << std::endl;
        min = 0;
        
        size_t maxV = (size_t)(max*multi);
        size_t minV = (size_t)(min*multi);
       

        ColorMap colorMap(maxV- minV);
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
               /* if(n != 0)
                {
                    std::cout << n <<std::endl;
                    std::cout << min <<std::endl;
                    std::cout << (n-min)/(min+1) <<std::endl;
                }*/
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
            //get minimum/maximum of band and find multipler that removes comma
            int counter = 0;
            float values[2];
            io->getMaxMinOfBand(values,b);
            
            int multi = 1;
            auto max = values[0];
            auto min = values[1];
        
            /*if(abs(max) < 1 || abs(min) < 1)
            {
                multi = 1000;
            }*/

            auto dimV = max - min;


            for (ssize_t y = 0; y < yDimTiff; y++)
            {
                for (ssize_t x = 0; x < xDimTiff; x++)
                {                
                    //FIXME: default color should be black and not white
                    auto n = mat->at<float>((yDimTiff - y - 1) * (xDimTiff) + x);
                    n /= dimV;
                    n = round(n*255);
                    if(n < 0)
                    {
                        n = 0;
                    }
                    else if(n > 255)
                    {
                        n = 255;
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

//cluster Materials sind nicht nutzbar für Farbe
//TODO: rename into something more fitting
template<typename BaseVecT>
MaterializerResult<BaseVecT> setColor(const lvr2::HalfEdgeMesh<BaseVecT>& mesh, const ClusterBiMap<FaceHandle>& clusters, const PointsetSurface<Vec>& surface, 
float texelSize, Eigen::MatrixXd affineMatrix, Eigen::MatrixXd fullAffineMatrix, GeoTIFFIO* io)
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

        auto xMax = max.x;
        auto xMin = min.x;
        auto yMax = max.y;
        auto yMin = min.y;
        
        //Rotates the extreme Values to fit the Texture
        if(affineMatrix.size() != 0)
        {
            Eigen::Vector4d pointMax(xMax,yMax,0,1);
            Eigen::Vector4d pointMin(xMin,yMin,0,1);

            Eigen::Vector4d solution;
            //Remove affine Translation if bool is set
            solution = affineMatrix*pointMax;
            
            xMax = solution.coeff(0);
            yMax = solution.coeff(1);

            solution = affineMatrix*pointMin;

            xMin = solution.coeff(0);
            yMin = solution.coeff(1);
            /*
            if(preventTranslation)
            {
                xMax -= affineMatrix(12);
                xMin -= affineMatrix(12);
                yMax -= affineMatrix(13);
                yMax -= affineMatrix(13);
            }*/
                
        }

        ssize_t x_max = (ssize_t)xMax + 1;
        ssize_t x_min = (ssize_t)xMin - 1;
        ssize_t y_max = (ssize_t)yMax + 1;
        ssize_t y_min = (ssize_t)yMin - 1;
        ssize_t x_dim = (abs(x_max) + abs(x_min)); 
        ssize_t y_dim = (abs(y_max) + abs(y_min)); 

        //TODO: implement a way to switch between test,tiff and geländemodell & make mode changeable from outside
        //TODO: implement a way to read a specific tiff

        //THIS PART IF GEOTIFF
        Texture tex;
        if(io)
        {
            tex = readGeoTIFF(io,1,3); 
        }
        else
        {
            tex = generateHeightDifferenceTexture<VecD,double>(surface,mesh,texelSize,3);
        }
            

        // Code copied from Materializer.tcc; this part essentially does what the materializer does
        // save Texture as Material so it can be correctly generated by the finalizer
        Material material;
        
        material.m_texture = textures.push(tex);
        //blank texture color is black
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
            
            //THIS PART IF GEOTIFF
            //Add affine Translation if bool is set
            float yPixel = 0;
            float xPixel = 0;
            
            if(io)
            {
                //double pos0,pos1;
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
                    /*pos0 = pos[0];
                    pos1 = pos[1];
                    pos0 += fullAffineMatrix(12);
                    pos1 += fullAffineMatrix(13);*/
                }
                
                xPixel = (pos[0] - geoTransform[0])/geoTransform[1];
                xPixel /= x_dim_tiff;
                yPixel = (pos[1] - geoTransform[3])/geoTransform[5];
                yPixel /= y_dim_tiff;
            }
            else
            {
                // correct coordinates
                BaseVecT correct(x_min,y_min,0);
                pos = pos - correct;
                xPixel = pos[0]/x_dim;
                yPixel = 1 - pos[1]/y_dim;
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
    //Reziproke Distanz f(d) = 1/d --> Je näher, desto maßgeblicher, linear
    //Gaußsche Klogenkurve f(d) = e(-ad²) --> extremer Fokus auf nahe Punkte
    return 1/distance;                
}
template <typename BaseVecT, typename Data>
Data findLowestZ(float x, float y, float lowestZ, float highestZ, float searchArea, PointsetSurfacePtr<Vec>& surface,FloatChannel& points)
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
        //TODO: check the math behind this --> muss glaube nur größer sqrt(2) sein, also passt 1.5 sogar
        size_t numNeighbors = surface->searchTree()->radiusSearch(BaseVecT(x,y,currentZ), 100, 1.5 * searchArea, neighbors, distances);
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
/*
floatArr transformCloud(FloatChannel& points, Eigen::MatrixXd affine)
{
    floatArr transformed( new float[1000 * 1000 * 3] );
    int y_dim = 1000;
    int x_dim = 1000;
    Eigen::Vector4d solution;
    for(int i = 0; i < points.numElements(); i++)
    {
        auto p = points[i];
        Eigen::Vector4d point(p[0],p[1],p[2],1);
        solution = affine * point;
        transformed[i * 3] = solution(0);
        transformed[i * 3 + 1] = solution(1);
        transformed[i * 3 + 2] = solution(2);
    }
    /*for (int y = 0; y < y_dim; y++)
    {
        for (int x = 0; x < x_dim; x++)
        {
            transformed[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 0] = x;
            transformed[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 1] = y;
            transformed[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 2] = 1;          
        }
    }//
    return transformed;

}
*/

template <typename BaseVecT, typename Data>
std::tuple<floatArr, size_t> extractGround(FloatChannel& points, PointsetSurfacePtr<Vec>& surface, float resolution,
 int smallWindow, float smallWindowHeight, int largeWindow, float largeWindowHeight, float slopeThreshold)
{
    // =======================================================================
    // Generate the Bounding Box and calculate the Grids size
    // =======================================================================
    
    auto bb = surface->getBoundingBox();
    auto max = bb.getMax();
    auto min = bb.getMin();
    
    ssize_t x_max = (ssize_t)max.x + 1;
    ssize_t x_min = (ssize_t)min.x - 1;
    ssize_t y_max = (ssize_t)max.y + 1;
    ssize_t y_min = (ssize_t)min.y - 1;
    ssize_t z_min = (ssize_t)min.z - 1;
    ssize_t z_max = (ssize_t)max.z + 1;
    ssize_t x_dim = abs(x_max) + abs(x_min);
    ssize_t y_dim = abs(y_max) + abs(y_min);

    //TODO: somehow bind this into the density of the pointcloud or make it super high
    int maxNeighbors = 1;

    float averagePointSpacing = 0;
    float averageHeight = 0;

    float xR = x_dim/resolution;
    int xReso = xR;
    if(xR - xReso >= 0.5)
    {
        xReso++;
    }
    float yR = y_dim/resolution;
    int yReso = yR;
    if(yR - yReso >= 0.5)
    {
        yReso++;
    }

    vector<vector<Data>> workGrid;
    workGrid.resize(xReso+1, vector<Data>(yReso+1,0));

    vector<size_t> indices;  
    vector<Data> distances;  
    
    //Point Spacing als Grid größe macht Meshes unmöglich riesig
    //Außerdem Widersprechen bzw. verunsichern die Angaben im Paper
    /*
    ProgressBar progress_ps(points.numElements(), timestamp.getElapsedTime() + "Calculating Point Spacing");
    auto elemnts = points.numElements();
    for(int i = 0; i < points.numElements(); i++)
    {
        //TODO: calculate this differently
        surface->searchTree()->kSearch(points[i],2,indices,distances);
        if(distances[1] == 0)
        {
            elemnts--;
            ++progress_ps;
            continue;
            
        }
        averagePointSpacing += distances[1];
        
        ++progress_ps;
    }
    if(averagePointSpacing != 0)
    {
        averagePointSpacing = averagePointSpacing/elemnts;
        std::cout << std::endl;
        std::cout << "average point spacing is " << averagePointSpacing << std::endl;
    }
    else
    {
        std::cout << "average point spacing is 0" << std::endl;
    }
    */

    // =======================================================================
    // Generate the Grid
    // =======================================================================
    ProgressBar progress_grid(xReso * yReso, timestamp.getElapsedTime() + "Calculating Grid");
    for(ssize_t y = 0; y < yReso; y++)
    {
        for(ssize_t x = 0; x < xReso; x++)
        {
            Data u_x = x * resolution;
            Data u_y = y * resolution;

            indices.clear();
            //Set the grid points height according to its nearest neighbors
            int numberNeighbors = surface->searchTree()->radiusSearch(BaseVecT(u_x + x_min,u_y + y_min,findLowestZ<BaseVecT,Data>(u_x + x_min,u_y + y_min,z_min,z_max,resolution/2,surface,points)),
             maxNeighbors, 1, indices, distances);
            if(numberNeighbors == 0)
            {
                workGrid[x][y] = std::numeric_limits<Data>::max();
                ++progress_grid;
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
            ++progress_grid;
        }
    }
        
    std::cout << std::endl;
    
    // =======================================================================
    // Extraction of Ground Points via Thresholding
    // =======================================================================
    // Three steps that try to filter ground points from non-ground points
    ProgressBar progress_points(x_dim/resolution * y_dim/resolution, timestamp.getElapsedTime() + "Checking Points");
    
    int counter = 0;
    
    list<BaseVecT> pointList;
    
    for (ssize_t y = 0; y < yReso; y++)
    {
        for (ssize_t x = 0; x < xReso; x++)
        {
            if(workGrid[x][y] == std::numeric_limits<Data>::max())
            {
                ++progress_points;
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
            // if height difference bigger than smallWindowHeight 
            // the point does not belong to the surface area
            if(abs(workGrid[x][y] - lowestDist) > smallWindowHeight)
            {
                ++progress_points;
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
                ++progress_points;
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
                ++progress_points;
                continue;
            }
            
            // if the point passes through the three tests, it gets recoginised as ground point
            pointList.push_back(BaseVecT(u_x+x_min, u_y+y_min,workGrid[x][y]));
          
            ++progress_points;
            counter++;
        }
        
    }

    floatArr grid( new float[3 * counter] );
    auto it = pointList.begin();
    for(int i = 0; i < pointList.size(); i++)
    {

        grid[i * 3] = it->x;
        grid[i * 3 + 1] = it->y;
        grid[i * 3 + 2] = it->z;
        ++it;
    }
    std::cout << std::endl;
    
    return {grid, counter};    
}

template <typename BaseVecT, typename Data>
void extractMesh(lvr2::HalfEdgeMesh<VecD>& mesh,FloatChannel& points, PointsetSurfacePtr<Vec>& surface, float resolution,
 int smallWindow, float smallWindowHeight, int largeWindow, float largeWindowHeight, float slopeThreshold, Eigen::MatrixXd affineMatrix)
{
    // =======================================================================
    // Generate the Bounding Box and calculate the Grids size
    // =======================================================================
    
    auto bb = surface->getBoundingBox();
    auto max = bb.getMax();
    auto min = bb.getMin();
    
    ssize_t x_max = (ssize_t)max.x + 1;
    ssize_t x_min = (ssize_t)min.x - 1;
    ssize_t y_max = (ssize_t)max.y + 1;
    ssize_t y_min = (ssize_t)min.y - 1;
    ssize_t z_min = (ssize_t)min.z - 1;
    ssize_t z_max = (ssize_t)max.z + 1;
    ssize_t x_dim = abs(x_max) + abs(x_min);
    ssize_t y_dim = abs(y_max) + abs(y_min);

    //TODO: somehow bind this into the density of the pointcloud or make it super high
    //Text speaks of a singular point
    int maxNeighbors = 1;

    float averagePointSpacing = 0;
    float averageHeight = 0;

    float xR = x_dim/resolution;
    int xReso = xR;
    if(xR - xReso >= 0.5)
    {
        xReso++;
    }
    float yR = y_dim/resolution;
    int yReso = yR;
    if(yR - yReso >= 0.5)
    {
        yReso++;
    }

    vector<vector<Data>> workGrid;
    workGrid.resize(xReso+1, vector<Data>(yReso+1,0));

    vector<size_t> indices;  
    vector<Data> distances;  
    
    //Point Spacing als Grid größe macht Meshes unmöglich riesig
    //Außerdem Widersprechen bzw. verunsichern die Angaben im Paper
    /*
    ProgressBar progress_ps(points.numElements(), timestamp.getElapsedTime() + "Calculating Point Spacing");
    auto elemnts = points.numElements();
    for(int i = 0; i < points.numElements(); i++)
    {
        //TODO: calculate this differently
        surface->searchTree()->kSearch(points[i],2,indices,distances);
        if(distances[1] == 0)
        {
            elemnts--;
            ++progress_ps;
            continue;
            
        }
        averagePointSpacing += distances[1];
        
        ++progress_ps;
    }
    if(averagePointSpacing != 0)
    {
        averagePointSpacing = averagePointSpacing/elemnts;
        std::cout << std::endl;
        std::cout << "average point spacing is " << averagePointSpacing << std::endl;
    }
    else
    {
        std::cout << "average point spacing is 0" << std::endl;
    }
    */

    // =======================================================================
    // Generate the Grid
    // =======================================================================
    ProgressBar progress_grid(xReso * yReso, timestamp.getElapsedTime() + "Calculating Grid");
    for(ssize_t y = 0; y < yReso; y++)
    {
        for(ssize_t x = 0; x < xReso; x++)
        {
            Data u_x = x * resolution;
            /*if(y%2 !=0)
            {
                u_x += resolution/2;
            } */
            Data u_y = y * resolution;

            indices.clear();
            //Set the grid points height according to its nearest neighbors
            int numberNeighbors = surface->searchTree()->radiusSearch(BaseVecT(u_x + x_min,u_y + y_min,findLowestZ<BaseVecT,Data>(u_x + x_min,u_y + y_min,z_min,z_max,resolution/2,surface,points)),
             maxNeighbors, 1, indices, distances);
            if(numberNeighbors == 0)
            {
                workGrid[x][y] = std::numeric_limits<Data>::max();
                ++progress_grid;
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
            ++progress_grid;
        }
    }
        
    std::cout << std::endl;
    
    // =======================================================================
    // Extraction of Ground Points via Thresholding
    // =======================================================================
    // Three steps that try to filter ground points from non-ground points
    ProgressBar progress_points(x_dim/resolution * y_dim/resolution, timestamp.getElapsedTime() + "Checking Points");
    
    int counter = 0;
    
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle> dict;
    
    for (ssize_t y = 0; y < yReso; y++)
    {
        for (ssize_t x = 0; x < xReso; x++)
        {
            if(workGrid[x][y] == std::numeric_limits<Data>::max())
            {
                ++progress_points;
                continue;
            }

            Data u_x = x * resolution;
            /*if(y%2 !=0)
            {
                u_x += resolution/2;
            } */
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
                ++progress_points;
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
                ++progress_points;
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
                ++progress_points;
                continue;
            }
            
            // if the point passes through the three tests, it gets recoginised as ground point
            double v_x = u_x+x_min;
            double v_y = u_y+y_min;
            double v_z = workGrid[x][y];
            if(affineMatrix.size() != 0)
            {
                Eigen::Vector4d point(v_x,v_y,v_z,1);

                Eigen::Vector4d solution;
                solution = affineMatrix*point;
                v_x = solution.coeff(0);
                v_y = solution.coeff(1);
                v_z = solution.coeff(2);
                //Right now, LVR2 doesn't support Large Coordinates and we can't use the Translation fully
                /*
                if(preventTranslation)
                {
                    v_x -= affineMatrix(12);
                    v_y -= affineMatrix(13);
                    v_z -= affineMatrix(14);
                }*/
            }
            VertexHandle v = mesh.addVertex(VecD(v_x,v_y,v_z));
            dict.emplace(std::make_tuple(x,y),v);
            ++progress_points;
            counter++;
        }
        
    }

    std::cout << std::endl;

    ProgressBar progress_grid2(dict.size(), timestamp.getElapsedTime() + "Writing Grid");

    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator pf1;
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator pf2;

    for(auto it = dict.begin(); it != dict.end(); )
    {
        tuple t = it->first;
        ssize_t x = std::get<0>(t);
        ssize_t y = std::get<1>(t);  
        int add = 0;
        /*if(y%2 != 0)
        {
            add = 1;
        }*/    

       
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
        ++progress_grid2;
    }

    std::cout << std::endl;
    
       
}

template <typename BaseVecT, typename Data>
void nearestNeighborNaiv(lvr2::HalfEdgeMesh<VecD>& mesh, FloatChannel& points, PointsetSurfacePtr<Vec>& surface,
int numNeighbors, Data stepSize, Eigen::MatrixXd& affineMatrix)
{
    // =======================================================================
    // Generating Boundingbox and Initialising 
    // =======================================================================
    auto bb = surface->getBoundingBox();
    auto max = bb.getMax();
    auto min = bb.getMin();
    
    ssize_t x_max = (ssize_t)max.x + 1;
    ssize_t x_min = (ssize_t)min.x - 1;
    ssize_t y_max = (ssize_t)max.y + 1;
    ssize_t y_min = (ssize_t)min.y - 1;
    ssize_t z_min = (ssize_t)min.z - 1;
    ssize_t z_max = (ssize_t)max.z + 1;
    ssize_t x_dim = abs(x_max) + abs(x_min);
    ssize_t y_dim = abs(y_max) + abs(y_min);

    // lists used when constructing
    vector<size_t> indices;
    vector<Data> distances;
    Data final_z = 0;
    Data added_distance = 0;          
    Data avg_distance = 0;
    Data avg_distance_point = 0;
    int trusted_neighbors = 0;

    bool trust_center_l = true;
    bool trust_center_r = true;
    bool trust_bottom_l = true;
    bool trust_bottom_r = true;
    bool trust_top_l = true;
    bool trust_top_r = true;

    Data neighbor_weighting = 1;

    int number_neighbors = numNeighbors;

    x_min = x_min * (1/stepSize);
    x_max = x_max * (1/stepSize);
    y_min = y_min * (1/stepSize);
    y_max = y_max * (1/stepSize);      
    
    // =======================================================================
    // Calculate Average Distance between Vertexpoint and closest Neighbor in their Tile
    // ======================================================================= 
    // Naiv approach to creating a DTM directly from a prefilterd Pointcloud
    
    //This part creates more problems than it solves
    /*ProgressBar progress_avg_dist((x_dim / stepSize)*(y_dim / stepSize), timestamp.getElapsedTime() + "Calculating average distance ");

    int numDiffs = 0;
    for (ssize_t x = x_min; x < x_max + 1; x++)
    {        
        for (ssize_t y = y_min; y < y_max + 1; y++)
        { 
            Data u_x = x * stepSize;
            /*if(y%2 !=0)
            {
                u_x += stepSize/2;
            } *
            Data u_y = y * stepSize;

            vector<size_t> indexes;
            vector<Data> distances;
            Data closeZ = findLowestZ<BaseVecT,Data>(u_x,u_y,z_min,z_max,stepSize/2,surface,points);
            if(closeZ == std::numeric_limits<Data>::max())
            {
                ++progress_avg_dist; 
                continue;
            }
            surface->searchTree()->kSearch(BaseVecT(u_x,u_y,closeZ),1,indexes,distances);
            avg_distance+= distances[0];  
            numDiffs++;   
            ++progress_avg_dist;           
        }
        
    }
    std::cout << std::endl;
    
    avg_distance = avg_distance/(numDiffs);
    std::cout << timestamp.getElapsedTime() << "Average Distance : " << avg_distance <<std::endl;
    */
    avg_distance = stepSize;
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle> dict1;
    
    // =======================================================================
    // Calculate Vertice Height and create Hexagonial Net
    // =======================================================================
    ProgressBar progress_vert((x_dim / stepSize)*(y_dim / stepSize), timestamp.getElapsedTime() + "Calculating Grid");
    for (ssize_t x = x_min; x < x_max; x++)
    {        
        for (ssize_t y = y_min; y < y_max; y++)
        {               
            Data u_x = x * stepSize;
            /*if(y%2 !=0)
            {
                u_x += stepSize/2;
            } */
            Data u_y = y * stepSize;
            
            indices.clear();
            distances.clear();

            // Center
            final_z = 0;  
            Data closeZ = findLowestZ<BaseVecT,Data>(u_x,u_y,z_min,z_max,stepSize/2,surface,points);
            if(closeZ == std::numeric_limits<Data>::max())
            {
                ++progress_vert;
                continue;
            }              
            surface->searchTree()->kSearch(BaseVecT(u_x,u_y,closeZ),number_neighbors,indices,distances);
            
            trusted_neighbors = number_neighbors;
            for (int i = 0; i < number_neighbors; i++)
            {
                if(distances[i] > avg_distance)
                {
                    trusted_neighbors--;
                }
                else
                {
                    auto index = indices[i];
                    auto nearest = points[index];
                    final_z = final_z + nearest[2];
                }
            }

            // if the center vertice isn't trustworthy, we can't complete
            // any triangle and thus skip the others
            if(trusted_neighbors < number_neighbors * neighbor_weighting)
            {
                ++progress_vert;
                continue;
            }  
            else
            {
                final_z = final_z/trusted_neighbors;
            }

            double d_x = u_x;
            double d_y = u_y;
            double d_z = final_z;

            if(affineMatrix.size() != 0)
            {
                Eigen::Vector4d point(u_x,u_y,final_z,1);

                Eigen::Vector4d solution;
                solution = affineMatrix*point;
                d_x = solution.coeff(0);
                d_y = solution.coeff(1);
                d_z = solution.coeff(2);
                //Right now, LVR2 doesn't support Large Coordinates and we can't use the Translation fully
                /*
                if(preventTranslation)
                {
                    u_x -= affineMatrix(12);
                    u_y -= affineMatrix(13);
                    final_z -= affineMatrix(14);
                }*/
            }
            VertexHandle v1 = mesh.addVertex(VecD(d_x,d_y,d_z)); 
            dict1.emplace(std::make_tuple(x,y),v1);
            ++progress_vert;
        }

    }
    std::cout << std::endl;

    ProgressBar progress_grid(dict1.size(), timestamp.getElapsedTime() + "Writing Grid");

    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator pf1;
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator pf2;

    for(auto it = dict1.begin(); it != dict1.end(); )
    {
        tuple t = it->first;
        ssize_t x = std::get<0>(t);
        ssize_t y = std::get<1>(t);  
        int add = 0;
        if(y%2 != 0)
        {
            add = 1;
        }    

       
        //pf1 = dict1.find(make_tuple(x+1,y));
        pf1 = dict1.find(make_tuple(x,y+1));
        if(pf1 != dict1.end())
        {   
            //pf2 = dict1.find(make_tuple(x+add,y+1));
            pf2 = dict1.find(make_tuple(x-1,y));
            if(pf2 != dict1.end())
            {
                mesh.addFace(it->second,pf1->second,pf2->second);
            }
        }

        //pf1 = dict1.find(make_tuple(x+1,y));
        pf1 = dict1.find(make_tuple(x,y+1));
        if(pf1 != dict1.end())
        {   
            //pf2 = dict1.find(make_tuple(x+add,y-1));
            pf2 = dict1.find(make_tuple(x+1,y+1));
            if(pf2 != dict1.end())
            {
                mesh.addFace(pf2->second,pf1->second,it->second);
            }
        }


        it++;
        ++progress_grid;
    }

    std::cout << std::endl;
}

template <typename BaseVecT, typename Data>
void movingAverage(lvr2::HalfEdgeMesh<VecD>& mesh, FloatChannel& points, PointsetSurfacePtr<Vec>& surface,
float minRadius, float maxRadius, int minNeighbors, int maxNeighbors, int radiusSteps, float stepSize, Eigen::MatrixXd& affineMatrix )
{
    // =======================================================================
    // Generating Boundingbox and Initialising 
    // =======================================================================
    auto bb = surface->getBoundingBox();
    auto max = bb.getMax();
    auto min = bb.getMin();

    int found = 0;
    //TODO: Change this to round in all functions
    ssize_t x_max = (ssize_t)max.x + 1;
    ssize_t x_min = (ssize_t)min.x - 1;
    ssize_t y_max = (ssize_t)max.y + 1;
    ssize_t y_min = (ssize_t)min.y - 1;
    ssize_t z_min = (ssize_t)min.z - 1;
    ssize_t z_max = (ssize_t)max.z + 1;
    ssize_t x_dim = abs(x_max) + abs(x_min);
    ssize_t y_dim = abs(y_max) + abs(y_min);

    size_t number_neighbors = 0;

    x_min = x_min * (1/stepSize);
    x_max = x_max * (1/stepSize);
    y_min = y_min * (1/stepSize);
    y_max = y_max * (1/stepSize);

    Data final_z = 0;  
    Data added_distance = 0;
    vector<size_t> indices;  
    vector<Data> distances;  

    // calculate the radius step size
    float radiusStepsize = (maxRadius - minRadius)/radiusSteps;
    float radius = 0;

    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle> dict;

    ProgressBar progress_vert((x_dim/stepSize)*(y_dim/stepSize), timestamp.getElapsedTime() + "Calculating z values "); 

    // =======================================================================
    // Calculate Vertice Height and create Hexagonial Net
    // =======================================================================
    for (ssize_t x = x_min; x < x_max; x++)
    {        
        for (ssize_t y = y_min; y < y_max; y++)
        {           
              
            Data u_x = x * stepSize;
            /*if(y%2 !=0)
            {
                u_x += stepSize/2;
            } */
            Data u_y = y * stepSize;

            BaseVecT point;
            
            found = 0;
            final_z = 0;  
            added_distance = 0;

            indices.clear();
            distances.clear(); 

            radius = minRadius;
            Data u_z = findLowestZ<BaseVecT,Data>(u_x,u_y,z_min,z_max,stepSize/2,surface,points);
            point = BaseVecT(u_x,u_y,u_z);
            //TODO: IF NO POINT IS FOUND JUST SKIP THIS
            // if we don't find enough points in the current radius, we extend the radius
            // if we hit the maximum extension and still find nothing, the point is left blank
            while (found == 0)
            {
                number_neighbors = surface->searchTree()->radiusSearch(point, maxNeighbors, radius, indices, distances);
                if(number_neighbors >= minNeighbors)
                {
                    found = 1;
                    break;
                }
                else if(radius <= maxRadius)
                {   
                    //std::cout << radius << " " << maxRadius << std::endl;
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
                for (int i = 0; i < number_neighbors; i++)
                {
                    size_t pointIdx = indices[i];
                    auto neighbor = points[pointIdx];
                    // when we are exactly on the point, distance is 0 and would divide by 0
                    Data distance = 1;
                    if(distances[i] != 0)
                    {
                        distance = weight<Data>(distances[i]); 
                    }                        
                    
                    final_z += neighbor[2] * distance;
                    added_distance += distance;                
                }  
                
                final_z = final_z/added_distance;                                 
                
            }
            else
            {
                ++progress_vert; 
                continue;
            }

            double d_x = u_x;
            double d_y = u_y;
            double d_z = final_z;

            if(affineMatrix.size() != 0)
            {
                Eigen::Vector4d point(u_x,u_y,final_z,1);

                Eigen::Vector4d solution;
                solution = affineMatrix*point;
                d_x = solution.coeff(0);
                d_y = solution.coeff(1);
                d_z = solution.coeff(2);
                //Right now, LVR2 doesn't support Large Coordinates and we can't use the Translation fully
                /*
                if(preventTranslation)
                {
                    u_x -= affineMatrix(12);
                    u_y -= affineMatrix(13);
                    final_z -= affineMatrix(14);
                }*/
            }
            VertexHandle v1 = mesh.addVertex(VecD(d_x,d_y,d_z)); 
            
            dict.emplace(std::make_tuple(x,y),v1);
            
            ++progress_vert;         
        }        
        
    }

    std::cout << std::endl;

    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator pf1;
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator pf2;

    ProgressBar progress_grid(dict.size(), timestamp.getElapsedTime() + "Writing Grid");

    for(auto it = dict.begin(); it != dict.end(); )
    {
        tuple t = it->first;
        ssize_t x = std::get<0>(t);
        ssize_t y = std::get<1>(t);  
        int add = 0;
        /*if(y%2 != 0)
        {
            add = 1;
        }*/    

       
        //pf1 = dict.find(make_tuple(x+1,y));
        pf1 = dict.find(make_tuple(x,y+1));
        if(pf1 != dict.end())
        {   
            //pf2 = dict.find(make_tuple(x+add,y+1));
            pf2 = dict.find(make_tuple(x-1,y));
            if(pf2 != dict.end())
            {
                mesh.addFace(it->second,pf1->second,pf2->second);
            }
        }

        //pf1 = dict.find(make_tuple(x+1,y));
        pf1 = dict.find(make_tuple(x,y+1));
        if(pf1 != dict.end())
        {   
            //pf2 = dict.find(make_tuple(x+add,y-1));
            pf2 = dict.find(make_tuple(x+1,y+1));
            if(pf2 != dict.end())
            {
                mesh.addFace(pf2->second,pf1->second,it->second);
            }
        }


        it++;
        ++progress_grid;
    }

    std::cout << std::endl;
}

int main(int argc, char* argv[])
{  
    // =======================================================================
    // Parse command line parameter
    // =======================================================================
    std::cout << std::fixed;
    lvr2::HalfEdgeMesh<VecD> mesh;
    
    //TODO: Handle this via Options
    //TODO: Make GeoTIFF and Texture Generation Optional
    //TODO: Falsche Anzahl an Bändern (nicht 1 oder 3) abfangen
    float minRadius= atof(argv[1]);
    float maxRadius= atof(argv[2]);
    int radiusSteps = atoi(argv[3]); 
    int minNeighbors = atoi(argv[4]); 
    int maxNeighbors = atoi(argv[5]); 
    float resolution = atof(argv[6]); 
    float texelSize = atof(argv[7]);  
    int activate_ground = atoi(argv[8]);
    int mode = atoi(argv[9]);
    string data(argv[10]);
    //string geoTIFFfile(argv[10]);
    string geoTIFFfile = "/home/mario/Schreibtisch/field_scans/ortho_austausch/20200807_hs_blang_rgb_ortho.tif";
    //"/home/mario/Schreibtisch/field_scans/ortho_austausch/20200807_hs_blang_multi_ortho_refl.tif";
    //"/home/mario/BA/UpToDatest/Develop/build/20200807_hs_blang_multi_ortho_refl.tif";

    //extractMesh<VecD,double>(mesh,usedArr,usedSurface,resolution,minNeighbors,minRadius,maxNeighbors,maxRadius,radiusSteps,affineMatrix);
    /*void extractMesh(lvr2::HalfEdgeMesh<BaseVecT>& mesh,FloatChannel& points, PointsetSurfacePtr<BaseVecT>& surface, float resolution,
    int smallWindow, float smallWindowHeight, int largeWindow, float largeWindowHeight, float slopeThreshold, Eigen::MatrixXd affineMatrix)*/
    bool targetDest = false;
    bool refPoints = false;
    bool gTiff = false;
    string newTiffName = "out.tif";

    /*VecD srcPoints[4] = {VecD(-1.445081 ,11.642192 ,-1.354529),VecD(-82.324432 ,41.830395 ,5.222107),
            VecD(-4.787183 ,82.624680 ,-0.580364),VecD(47.635303, 106.202621, 0.513081)};
    VecD dstPoints[4] = {VecD(32563657.761 ,6019213.520 ,26.518),VecD(32563623.915, 6019134.116, 34.384),
        VecD(32563586.623, 6019213.195, 26.113),VecD(32563566.373,6019266.616,25.408)};*/

    VecD srcPoints[4] = {VecD(-2.924260 ,11.032000 ,-1.123120),VecD(-82.946602,43.682301,5.424430),
        VecD(-4.189860 ,82.130402 ,-0.529224),VecD(48.909199, 103.139999, 0.316265)};
    VecD dstPoints[4] = {VecD(32563657.761 ,6019213.520 ,26.518),VecD(32563623.915, 6019134.116, 34.384),
        VecD(32563586.623, 6019213.195, 26.113),VecD(32563566.373,6019266.616,25.408)};

    GeoTIFFIO* io;
    GDALDataset* set;
    // =======================================================================
    // Read GeoTIFF + ReffPoints and transform
    // =======================================================================
    if(targetDest)
    {
        if(refPoints)
        {
            string typ1 = "EPSG:4647";
            string typ2 = "EPSG:4647";    
            VecD dstP[4];
            //VecD refP[4] = {VecD(32563657.761 ,6019213.520 ,26.518),VecD(32563623.915, 6019134.116, 34.384),
                //VecD(32563586.623, 6019213.195, 26.113),VecD(32563566.373,6019266.616,25.408)};

            transformPoints(typ1,typ2,dstPoints,dstP);

            std::copy(std::begin(dstP),std::end(dstP),std::begin(dstPoints));
            /*for (size_t i = 0; i < 4; i++)
            {
                std::cout << dstPoints[i] << std::endl;
            }*/

            if(gTiff)
            {
                GDALDatasetH src = GDALOpen(geoTIFFfile.c_str(),GA_ReadOnly);
                GDALDatasetH dt;
                //creates a new GeoTIFF file with the transformed info of the old one
                warpGeoTIFF(src,dt,typ2,newTiffName);
            }
        }
    
    }
    
    // =======================================================================
    // Compute Affine Transform Matrix from Transformed Reff Points
    // =======================================================================
    Eigen::MatrixXd affineMatrix, fullAffineMatrix;
    if(refPoints)
    {    
        tie(affineMatrix,fullAffineMatrix) = computeAffineGeoRefMatrix(srcPoints,dstPoints);
        //Right now, LVR2 doesn't support Large Coordinates and we can't use the Translation fully
        //In Functions where we use the Matrix we need to exclude the Translation
        std::cout << affineMatrix << std::endl;  
        if(gTiff)
        {
            if(targetDest)
            {                
                io = new GeoTIFFIO(newTiffName);
            }
            else
            {
                io = new GeoTIFFIO(geoTIFFfile);
            }
        }      
    }

    // =======================================================================
    // Load Pointcloud and create Model + Surface
    // =======================================================================
 
    auto surface = loadPointCloud<Vec>(data);
    ModelPtr base_model = ModelFactory::readModel(data);
    PointBufferPtr base_buffer = base_model->m_pointCloud;

    // get the pointcloud coordinates from the FloatChannel
    FloatChannel arr =  *(base_buffer->getFloatChannel("points"));   

    PointsetSurfacePtr<Vec> usedSurface = surface;
    FloatChannel usedArr = arr;

    // =======================================================================
    // Transform Pointcloud
    // =======================================================================
    /*if(false)
    {
        auto transformedPoints = transformCloud(arr,affineMatrix);
        PointBufferPtr p;
        p = PointBufferPtr( new PointBuffer );
        p->setPointArray(transformedPoints,1000 * 1000);
        usedArr = *(p->getFloatChannel("points")); 
        usedSurface = rebuildPointCloud<BaseVecT>(p);
    }
    std::cout << "works" << std::endl;*/
    
    
    // =======================================================================
    // Extract the Surface Area from the Pointcloud if enabled
    // =======================================================================
    if(activate_ground != 0)
    {
        auto [grid, np] = extractGround<Vec, float>(arr,surface,0.125,21,2,115,2.5,15);
        PointBufferPtr p;
        p = PointBufferPtr( new PointBuffer );
        p->setPointArray(grid,np);
        usedArr = *(p->getFloatChannel("points")); 
        usedSurface = rebuildPointCloud<Vec>(p);
    }
    
    std::cout << timestamp.getElapsedTime() << " Start" << std::endl;
    if(mode == 0)
    {
        std::cout << "Moving Average" << std::endl;
        movingAverage<Vec,float>(mesh,usedArr,usedSurface,minRadius,maxRadius,minNeighbors,maxNeighbors,radiusSteps,resolution,affineMatrix);
    }
    else if(mode == 1)
    {
        std::cout << "Nearest Neighbor" << std::endl;
        nearestNeighborNaiv<Vec,float>(mesh,usedArr,usedSurface,minNeighbors,resolution,affineMatrix);
    }
    else
    {
        std::cout << "Threshold Method"<< std::endl;
        //extractMesh<VecD,double>(mesh,usedArr,usedSurface,resolution,5,0.4,11,0.5,5,affineMatrix);
        extractMesh<Vec,float>(mesh,usedArr,usedSurface,resolution,minNeighbors,minRadius,maxNeighbors,maxRadius,radiusSteps,affineMatrix);
    }
    std::cout << timestamp.getElapsedTime() << "End" << std::endl;
    
    // =======================================================================
    // Setup LVR_2 Function to allow the export of the Mesh as obj/ply
    // =======================================================================
    // creating a cluster map made up of one cluster is necessary to use the finalizer   
    
    //auto faceNormals = calcFaceNormals<Vec>(mesh);
    ClusterBiMap<FaceHandle> clusterBiMap;
    MeshHandleIteratorPtr<FaceHandle> iterator = mesh.facesBegin();
    auto newCluster = clusterBiMap.createCluster();
    for (size_t i = 0; i < mesh.numFaces(); i++)
    { 
        clusterBiMap.addToCluster(newCluster,*iterator);

        ++iterator;
    }  
    //auto vertexNormals = calcVertexNormals<Vec>(mesh, faceNormals, *usedSurface);
    TextureFinalizer<VecD> finalize(clusterBiMap);
    //finalize.setVertexNormals(vertexNormals);

    // this has to be the normal surface and not the extracted ground
    //TODO: add option for choosin which color scale to use
    auto matResult = setColor<VecD>(mesh,clusterBiMap,*surface,texelSize,affineMatrix,fullAffineMatrix,io);
    finalize.setMaterializerResult(matResult);    

    auto buffer = finalize.apply(mesh);

    buffer->addIntAtomic(1, "mesh_save_textures");
    buffer->addIntAtomic(1, "mesh_texture_image_extension");
    std::cout << timestamp.getElapsedTime() << " Setting Model" << std::endl;
    auto m = ModelPtr(new Model(buffer)); 
    
    //TODO: make filenames less messy
    size_t pos = data.find_last_of("/");  
    string name = data.substr(pos+1);
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
    auto time = oss.str();

    // =======================================================================
    // Export Files as PLY and OBJ with a JPEG as Texture
    // =======================================================================    
    std::cout << timestamp.getElapsedTime() << "Saving Model as ply" << std::endl;
    ModelFactory::saveModel(m,time +"_min_neighbors_"+ to_string(minNeighbors)
    + "_max_neighbors_" + to_string(maxNeighbors)+ "_min_radius_" + to_string(minRadius)
    + "_max_radius_" + to_string(maxRadius) + "_radius_steps_" + to_string(radiusSteps)
    + "_groundlevel_" + name);

    std::cout << timestamp.getElapsedTime() << "Saving Model as obj" << std::endl;
    ModelFactory::saveModel(m,time +"_min_neighbors_"+ to_string(minNeighbors)
    + "_max_neighbors_" + to_string(maxNeighbors)+ "_min_radius_" + to_string(minRadius)
    + "_max_radius_" + to_string(maxRadius) + "_radius_steps_" + to_string(radiusSteps)
    + "_groundlevel_" + ".obj");  

    delete(io);
    return 0;
}