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

#include "lvr2/geometry/BVH.hpp"

#include "lvr2/reconstruction/DMCReconstruction.hpp"

#include "lvr2/io/PLYIO.hpp"

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
using PsSurface = lvr2::PointsetSurface<Vec>;

std::map<std::tuple<ssize_t, ssize_t>,float> dict_z;
std::map<std::tuple<ssize_t, ssize_t>,VertexHandle> dict_point;

//creates a surface for K search
template <typename BaseVecT>
PointsetSurfacePtr<BaseVecT> loadPointCloud(string &data)
{
    ModelPtr base_model = ModelFactory::readModel(data);
    PointBufferPtr base_buffer = base_model->m_pointCloud;
    PointsetSurfacePtr<Vec> surface;
    //TODO: what is the difference between FLANN and the other modes and is FLANN the best to use
    surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(base_buffer,"FLANN");
    surface->calculateSurfaceNormals();
    return surface;

}
//TODO: Generate a texture that contains the height difference
//TODO: Make texelsize scaleable

/**Von Texel zu Face/Vertice --> Wie komme ich auf die Faces ??
Finde für das Texel den nächsten Punkt in der Punktwolke
Finde für die X,Y Koordinaten des Punktes ein Face, welches dieser erstellen könnte
Ermittle den Z Koordinatenunterschied und setze danach gehend die Farbe
Nachteil: das dauert bestimmt super fucking lange
Vorteil: kann für beliebige meshes verwendet werden
Wir gehen alle Faces durch und gucken, welche Koordinaten das in der Textur wären
Vorher initialiseren wir alles mit einer Farbe die angibt, das wir keine Ahnung haben was da ist
Vorteil: viel fucking schneller
Nachteil: kriege ich es hin, konsistenz die richtigen Texturkoordinaten auszurechnen??**/

template <typename BaseVecT>
Texture generateHeightDifferenceTexture(const PointsetSurface<BaseVecT>& surface ,const lvr2::HalfEdgeMesh<Vec>& mesh)
{
    float texelSize = 0.5;

    auto bb = surface.getBoundingBox();
    auto max = bb.getMax();
    auto min = bb.getMin();
    
    ssize_t x_max = (ssize_t)max.x + 1;
    ssize_t x_min = (ssize_t)min.x - 1;
    ssize_t y_max = (ssize_t)max.y + 1;
    ssize_t y_min = (ssize_t)min.y - 1; 
    ssize_t z_max = (ssize_t)max.z + 1;
    ssize_t x_dim = (abs(x_max) + abs(x_min))*2; //*2 is there due to texel size 0.5
    ssize_t y_dim = (abs(y_max) + abs(y_min))*2; //*2 is there due to texel size 0.5
    
    //since we only create one texture, the index is 5 TODO: make use of the index
    Texture texture(2, x_dim, y_dim, 3, 1, texelSize);
    //contains the distances from each relevant point in the mesh to its closest neighbor
    float distance[x_dim * y_dim];
    float max_positive_distance = 0;
    float max_negative_distance = 0;
    int counter = 0;

    //Initialise every texel with a neutral color and distance array with neutral distance
    for (int y = 0; y < y_dim; y++)
    {
        for (int x = 0; x < x_dim; x++)
        {
            distance[(y_dim - y - 1) * (x_dim) + x] = std::numeric_limits<float>::min();
        }
    }
    PointBufferPtr base_buffer = surface.pointBuffer();    
    FloatChannel arr =  *(base_buffer->getFloatChannel("points"));   
    MeshHandleIteratorPtr<FaceHandle> iterator = mesh.facesBegin();

    ProgressBar progress_distance(mesh.numFaces(), timestamp.getElapsedTime() + "Calcing distances ");

    //TODO: make texel_size way smaller
    for (size_t i = 0; i < mesh.numFaces(); i++)
    {
        Vec correct(x_min,y_min,0);
        auto real_point1 = mesh.getVertexPositionsOfFace(*iterator)[0];
        auto real_point2 = mesh.getVertexPositionsOfFace(*iterator)[1];
        auto real_point3 = mesh.getVertexPositionsOfFace(*iterator)[2];

        auto point1 = real_point1 - correct;
        auto point2 = real_point2 - correct;
        auto point3 = real_point3 - correct;

        auto max_x = std::max(point1[0],std::max(point2[0],point3[0]));
        ssize_t fmax_x = (ssize_t)max_x + 1;
        auto min_x = std::min(point1[0],std::min(point2[0],point3[0]));
        ssize_t fmin_x = (ssize_t)min_x - 1;
        auto max_y = std::max(point1[1],std::max(point2[1],point3[1]));
        ssize_t fmax_y = (ssize_t)max_y + 1;
        auto min_y = std::min(point1[1],std::min(point2[1],point3[1]));
        ssize_t fmin_y = (ssize_t)min_y - 1;

        float face_surface = 0.5 *((point2[0] - point1[0])*(point3[1] - point1[1])
            - (point2[1] - point1[1]) * (point3[0] - point1[0]));
        
        for (float y = fmin_y; y < fmax_y; y+=texelSize)
        {
            for (float x = fmin_x; x < fmax_x; x+=texelSize)
            {
                float surface_1 = 0.5 *((point2[0] - x)*(point3[1] - y)
                - (point2[1] - y) * (point3[0] - x));

                float surface_2 = 0.5 *((point3[0] - x)*(point1[1] - y)
                - (point3[1] - y) * (point1[0] - x));

                float surface_3 = 0.5 *((point1[0] - x)*(point2[1] - y)
                - (point1[1] - y) * (point2[0] - x));

                surface_1 = surface_1/face_surface;
                surface_2 = surface_2/face_surface;
                surface_3 = surface_3/face_surface;                

                if(surface_1 < 0 || surface_2 < 0 || surface_3 < 0)
                {
                    continue;
                }
                else
                {
                    ssize_t x_tex = (ssize_t)(x *2);
                    ssize_t y_tex = (ssize_t)(y *2);
                    //due to me being dumb, this workaround is necesarry
                    //y_tex can become -1, because y can hit -0.5, because, when creating the mesh
                    //we go over every coordinate and add +0.5 and -0.5
                    //i either have to correct the boundaries or catch it here
                    if((y_dim * x_dim * 3 * 1) < ((y_dim - y_tex  - 1) * (x_dim * 3) + 3 * x_tex  + 0)){
                        continue;
                    }     

                    //interpolate point
                    //find neihbor in pointcloud
                    //HIGHEST POINT
                    //calculate difference      

                    Vec point = real_point1 * surface_1 + real_point2 * surface_2 + real_point3 * surface_3;      
                    Vec highest_point = point;
                    highest_point[2] = z_max;
                    //find the clostest neighbor
                    vector<size_t> cv;
                    surface.searchTree()->kSearch(highest_point, 1, cv);  
                    
                    //to make a dynamic gradient, we should save all of the points we want to colorize and their distance
                    //after knowing the maximum distance, we can color the texture accordingly
                    for (size_t pointIdx : cv)
                    {
                        auto cp = arr[pointIdx];
                        distance[(y_dim - y_tex  - 1) * (x_dim) + x_tex] =  point[2] - cp[2];                        
                        
                        if(max_positive_distance < point[2] - cp[2])
                        {
                            max_positive_distance = point[2] - cp[2];
                        }    

                        if(max_negative_distance > point[2] - cp[2])
                        {
                            max_negative_distance = point[2] - cp[2];
                        }                 
                        
                    }
                                 
                }

            }
            
        }
        ++progress_distance;
        ++iterator;
    }  
    std::cout << std::endl;

    //set color gradient points

    ProgressBar progress_color(x_dim * y_dim, timestamp.getElapsedTime() + "Setting colors ");    

    float max_distance = 0;
    if(max_positive_distance > abs(max_negative_distance))
    {
        max_distance = max_positive_distance;
    }else
    {
        max_distance = abs(max_negative_distance);
    }

    for (int y = 0; y < y_dim; y++)
    {
        for (int x = 0; x < x_dim; x++)
        {
            if(distance[(y_dim - y - 1) * (x_dim) + x] == std::numeric_limits<float>::min())
            {
                texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 0] = 0;
                texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 1] = 0;
                texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 2] = 0;
            }
            else if(distance[(y_dim - y - 1) * (x_dim) + x] >= 0)
            {
                float gradient = distance[(y_dim - y - 1) * (x_dim) + x]/max_distance;

                texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 0] = 255;
                texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 1] = 255 *(1-gradient);
                texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 2] = 255 *(1-gradient);
            }
            else if (distance[(y_dim - y - 1) * (x_dim) + x] < 0)
            {
                float gradient = abs(distance[(y_dim - y - 1) * (x_dim) + x]/max_distance);

                texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 0] = 255 *(1-gradient);
                texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 1] = 255 *(1-gradient);
                texture.m_data[(y_dim - y - 1) * (x_dim * 3) + x * 3 + 2] = 255;
            }
            
            ++progress_color;
            
        }
    }
    std::cout << std::endl;
    return texture;
}

//TODO:Materialiser --> weise Texturdaten so zu, dass das face einfach als Farbe die nächste Textur hat

int main(int argc, char* argv[])
{
    lvr2::HalfEdgeMesh<Vec> mesh;
    
    //TODO: read infos with Options file
    int number_neighbors = atoi(argv[1]);
    int mode = atoi(argv[2]);
    float distance_weighting = atof(argv[3]);
    float neighbor_weighting = atof(argv[4]);
    string data(argv[5]);

    //load the surface
    auto surface = loadPointCloud<Vec>(data);
    ModelPtr base_model = ModelFactory::readModel(data);
    PointBufferPtr base_buffer = base_model->m_pointCloud;
    
    //get the pointcloud coordinates from the FloatChannel
    FloatChannel arr =  *(base_buffer->getFloatChannel("points"));   
    
    //load the boundingbox to specify the size of the mesh
    auto bb = surface->getBoundingBox();
    auto max = bb.getMax();
    auto min = bb.getMin();
    
    ssize_t x_max = (ssize_t)max.x + 1;
    ssize_t x_min = (ssize_t)min.x - 1;
    ssize_t y_max = (ssize_t)max.y + 1;
    ssize_t y_min = (ssize_t)min.y - 1;
    ssize_t z_min = (ssize_t)min.z - 1;
    ssize_t x_dim = abs(x_max) + abs(x_min);
    ssize_t y_dim = abs(y_max) + abs(y_min);

    //lists used when constructing
    vector<size_t> indices;
    vector<float> distances;
    float final_z = 0;
    std::map<std::tuple<ssize_t, ssize_t>,float>::iterator it;
    std::map<std::tuple<ssize_t, ssize_t>,VertexHandle>::iterator it_point;
    float added_distance = 0;
    
    //TODO: ???Filter der Annomalien beseitigt entwerfen
    //TODO: ???Punkte nach Distanz bewerten
    //TODO: make faces scaleable
    
    //+0.5 so that the faces line up 
    float h = 0.5;
    float step_size = 0.5;
    float avg_distance = 0;
    float avg_distance_point = 0;
    bool trust_center_l = true;
    bool trust_center_r = true;
    bool trust_bottom_l = true;
    bool trust_bottom_r = true;
    bool trust_top_l = true;
    bool trust_top_r = true;
    int trusted_neighbors = 0;
    
    //Calculate the average distance 
    ProgressBar progress_avg_dist(x_dim*y_dim / step_size, timestamp.getElapsedTime() + "Calcing avgerage distance ");
   
    for (float x = x_min; x < x_max + 1; x+=step_size)
    {        
        for (float y = y_min; y < y_max + 1; y++)
        { 
            vector<size_t> indexes;
            vector<float> distances;
            surface->searchTree()->kSearch(Vec(x,y,z_min),1,indexes,distances); 
            auto index = indexes[0];
            auto closest = arr[index];  
            surface->searchTree()->kSearch(Vec(x,y,closest[2]),1,indexes,distances);
            avg_distance+= distances[0];     
            ++progress_avg_dist;           
        }
        
    }
    std::cout << std::endl;

    avg_distance = distance_weighting * avg_distance/(x_dim * y_dim);
    printf("Average Distance: %f\n",avg_distance);

        
    //Calculate the Vertice Positions
    ProgressBar progress_vert(x_dim * y_dim / step_size, timestamp.getElapsedTime() + "Calculating vertice positions ");
    for (float x = x_min; x < x_max; x+=step_size)
    {        
        for (float y = y_min; y < y_max; y++)
        {               
            indices.clear();

            //Center
            final_z = 0;                
            surface->searchTree()->kSearch(Vec(x,y,z_min),number_neighbors,indices,distances);
            
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
                    auto nearest = arr[index];
                    final_z = final_z + nearest[2];
                }
            }
            //if the center vertice isn't trustworthy, we can't complete
            //any triangle and thus skip the others
            if(trusted_neighbors < number_neighbors * neighbor_weighting)
            {
                ++progress_vert;
                continue;
            }  
            else
            {
                final_z = final_z/trusted_neighbors;
            }

            VertexHandle center = mesh.addVertex(Vec(x,y,final_z));

            //Center Right               
            final_z = 0;                
            surface->searchTree()->kSearch(Vec(x+0.5,y,z_min),number_neighbors,indices);
            
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
                    auto nearest = arr[index];
                    final_z = final_z + nearest[2];
                }             
            } 

            if(trusted_neighbors < number_neighbors * neighbor_weighting)
            {
                trust_center_r = false;
            }  
            else
            {
                trust_center_r = true;
                final_z = final_z/trusted_neighbors;
            }
            VertexHandle center_right = mesh.addVertex(Vec(x+0.5,y,final_z));

            //Center Left
            final_z = 0;                
            surface->searchTree()->kSearch(Vec(x-0.5,y,z_min),number_neighbors,indices);
            
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
                    auto nearest = arr[index];
                    final_z = final_z + nearest[2];
                }
            }  

            if(trusted_neighbors < number_neighbors * neighbor_weighting)
            {
                trust_center_l = false;
            }  
            else
            {
                trust_center_l = true;
                final_z = final_z/trusted_neighbors;
            }
            VertexHandle center_left = mesh.addVertex(Vec(x-0.5,y,final_z));

            //Bottom Right
            final_z = 0;                
            surface->searchTree()->kSearch(Vec(x+0.25,y-h,z_min),number_neighbors,indices);
            
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
                    auto nearest = arr[index];
                    final_z = final_z + nearest[2];
                }
            } 

            if(trusted_neighbors < number_neighbors * neighbor_weighting)
            {
                trust_bottom_r = false;
            }  
            else
            {
                trust_bottom_r = true;
                final_z = final_z/trusted_neighbors;
            } 
            VertexHandle bottom_right = mesh.addVertex(Vec(x+0.25,y-h,final_z));

            //Bottom Left
            final_z = 0;                
            surface->searchTree()->kSearch(Vec(x-0.25,y-h,z_min),number_neighbors,indices);
            
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
                    auto nearest = arr[index];
                    final_z = final_z + nearest[2];
                }
            } 

            if(trusted_neighbors < number_neighbors * neighbor_weighting)
            {
                trust_bottom_l = false;
            }  
            else
            {
                trust_bottom_l = true;
                final_z = final_z/trusted_neighbors;
            } 
            VertexHandle bottom_left = mesh.addVertex(Vec(x-0.25,y-h,final_z));
            
            //Top_right
            final_z = 0;                
            surface->searchTree()->kSearch(Vec(x+0.25,y+h,z_min),number_neighbors,indices);

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
                    auto nearest = arr[index];
                    final_z = final_z + nearest[2];
                }
            }  

            if(trusted_neighbors < number_neighbors * neighbor_weighting)
            {
                trust_top_r = false;
            }  
            else
            {
                trust_top_r = true;
                final_z = final_z/trusted_neighbors;
            } 
            VertexHandle top_right = mesh.addVertex(Vec(x+0.25,y+h,final_z));

            //Top Left
            final_z = 0;                
            surface->searchTree()->kSearch(Vec(x-0.25,y+h,z_min),number_neighbors,indices);
            
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
                    auto nearest = arr[index];
                    final_z = final_z + nearest[2];
                }
            }  

            if(trusted_neighbors < number_neighbors * neighbor_weighting)
            {
                trust_top_l = false;
            }  
            else
            {
                trust_top_l = true;
                final_z = final_z/trusted_neighbors;
            }  
            VertexHandle top_left = mesh.addVertex(Vec(x-0.25,y+h,final_z));

            if(trust_top_r && trust_top_l)
            {
                mesh.addFace(center,top_right,top_left);
            }

            if(trust_bottom_r && trust_bottom_l)
            {
                mesh.addFace(center,bottom_left,bottom_right);
            }

            if(trust_bottom_r && trust_center_r)
            {
                mesh.addFace(center,bottom_right,center_right);
            }
            
            if(trust_center_r && trust_top_r)
            {
                mesh.addFace(center,center_right,top_right);
            }

            if(trust_top_l && trust_center_l)
            {
                mesh.addFace(center,top_left,center_left);
            }

            if(trust_center_l && trust_bottom_l)
            {
                mesh.addFace(center,center_left,bottom_left);
            }               
            ++progress_vert;
        }
    }
    std::cout << std::endl;

    //creating a cluster map made up of one cluster is necessary to use the finalizer    
    auto faceNormals = calcFaceNormals(mesh);
    ClusterBiMap<FaceHandle> clusterBiMap;
    BaseVector<float> currentCenterPoint;
    MeshHandleIteratorPtr<FaceHandle> iterator = mesh.facesBegin();
    auto newCluster = clusterBiMap.createCluster();
    for (size_t i = 0; i < mesh.numFaces(); i++)
    {
        currentCenterPoint = (mesh.getVertexPositionsOfFace(*iterator)[0] + mesh.getVertexPositionsOfFace(*iterator)[1]
            + mesh.getVertexPositionsOfFace(*iterator)[2])/ 3;        
        clusterBiMap.addToCluster(newCluster,*iterator);

        ++iterator;
    }    
    /*-------------------------------------------------------------------------------------------------------*/
    //TODO: create Texture instead of colormaps
    ClusterPainter painter(clusterBiMap);
    auto clusterColors = boost::optional<DenseClusterMap<Rgb8Color>>(painter.simpsons(mesh));
    DenseClusterMap<Rgb8Color> colorMap;
    colorMap.reserve(clusterBiMap.numHandles() * 3);
    auto vertexNormals = calcVertexNormals(mesh, faceNormals, *surface);
    TextureFinalizer<Vec> finalize(clusterBiMap);
    finalize.setVertexNormals(vertexNormals);
    finalize.setClusterColors(*clusterColors);
    
    //Exchange materliazer and texturizer with selfmade functions
    Materializer<Vec> materializer(
        mesh,
        clusterBiMap,
        faceNormals,
        *surface
    );

    Texturizer<Vec> texturizer(
        1,
        mesh.numFaces(),
        mesh.numFaces()
    );

    materializer.setTexturizer(texturizer);
    MaterializerResult<Vec> matResult = materializer.generateMaterials();
    finalize.setMaterializerResult(matResult);
    
    auto tex = generateHeightDifferenceTexture(*surface,mesh);
    tex.save();

    /*-------------------------------------------------------------------------------------------------------*/
    auto buffer = finalize.apply(mesh);
    /*materializer.saveTextures();
    buffer->addIntAtomic(1, "mesh_save_textures");
    buffer->addIntAtomic(1, "mesh_texture_image_extension");*/
    auto m = ModelPtr(new Model(buffer));    

    size_t pos = data.find_last_of("/");  
    string name = data.substr(pos+1);
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
    auto time = oss.str();
    ModelFactory::saveModel(m,time + "_" + to_string(mode) + "_neighborhood_"+ to_string(number_neighbors)
    + "_distance_weighting_" + to_string(distance_weighting)+ "_neighbor_weighting_" + to_string(neighbor_weighting) 
    + "_groundlevel_" + name);
    ModelFactory::saveModel(m, "data.obj");   

    return 0;
}