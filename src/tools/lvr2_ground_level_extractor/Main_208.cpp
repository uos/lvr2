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

StableVector<TextureHandle, Texture> m_textures;

//creates a surface for K search
//TO_DO: understand how it works exactly
template <typename BaseVecT>
PointsetSurfacePtr<BaseVecT> loadPointCloud(string &data)
{
    ModelPtr base_model = ModelFactory::readModel(data);
    PointBufferPtr base_buffer = base_model->m_pointCloud;
    PointsetSurfacePtr<Vec> surface;
    //TO_DO: what is the difference between FLANN and the other modes and is FLANN the best to use
    surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(base_buffer,"FLANN");
    surface->calculateSurfaceNormals();
    return surface;

}

TextureHandle generateHeightDifferenceTexture(int index)
{
    //TO_DO: actualy calculate the difference and set the colors accordingly
    unsigned short int sizeX = 100;
    unsigned short int sizeY = 100;
    float texelSize = 1;
    Texture texture(index, sizeX, sizeY, 3, 1, texelSize);
    ColorMap cMap(3);
    float colors[3] = {};
    cMap.getColor(colors,2,SIMPSONS);

    for (int y = 0; y < sizeY; y++)
    {
        for (int x = 0; x < sizeX; x++)
        {

            texture.m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 0] = 255;
            texture.m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 1] = 255;
            texture.m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 2] = 0;
        }
    }

    return m_textures.push(texture);
}

//TO_DO: transform into a form that actualy does what i want
template<typename BaseVecT>
MaterializerResult<BaseVecT> generateMaterials(TextureHandle texH, ClusterBiMap<FaceHandle>& m_cluster, BaseMesh<BaseVecT>& m_mesh
, PointsetSurface<BaseVecT>& m_surface, FaceMap<Normal<typename BaseVecT::CoordType>>& m_normals)
{
    // Prepare result
    DenseClusterMap<Material> clusterMaterials;
    SparseVertexMap<ClusterTexCoordMapping> vertexTexCoords;

    std::unordered_map<BaseVecT, std::vector<float>> keypoints_map;

    // Counters used for texturizing
    int numClustersTooSmall = 0;
    int numClustersTooLarge = 0;
    int textureCount = 0;
    int clusterCount = 0;

    for (auto clusterH : m_cluster)
    {
        const Cluster<FaceHandle>& cluster = m_cluster.getCluster(clusterH);
        int numFacesInCluster = cluster.handles.size();

        // Textures

        // Contour
        std::vector<VertexHandle> contour = calculateClusterContourVertices(
            clusterH,
            m_mesh,
            m_cluster
        );

        // Bounding rectangle
        BoundingRectangle<typename BaseVecT::CoordType> boundingRect = calculateBoundingRectangle(
            contour,
            m_mesh,
            cluster,
            m_normals,
            1,
            clusterH
        );

        
        // Create texture

        printf("Generating Texturizer works\n");
        //Was macht cv hier und brauche ich das
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        //welchen feature detectot will ich ??
        cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
        /* m_texturizer.findKeyPointsInTexture(texH,
                boundingRect, detector, keypoints, descriptors);*/
        /*----------------------------------------------------------------------------*/
        const Texture texture = m_textures[texH];
        /*if (texture.m_height <= 32 && texture.m_width <= 32)
        {
            return;
        }*/
        const unsigned char* img_data = texture.m_data;
        cv::Mat image(texture.m_height, texture.m_width, CV_8UC3, (void*)img_data);

        detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        printf("Generating Keypoints works\n");
        /*----------------------------------------------------------------------------*/
        
        Texturizer<Vec> texturizer(
        1,
        1,
        1
        );
        
        /*std::vector<BaseVecT> features3d =
            m_texturizer.keypoints23d(keypoints, boundingRect, texH);
        printf("Generating keypoints23d works\n");*/
        const size_t N = keypoints.size();
        std::vector<BaseVecT> keypoints3d(N);
        const int width            = m_textures[texH].m_width;
        const int height           = m_textures[texH].m_height;



        for (size_t p_idx = 0; p_idx < N; ++p_idx)
        {
            const cv::Point2f keypoint = keypoints[p_idx].pt;
            // Calculate texture coordinates from pixel locations and then calculate backwards
            // to 3D coordinates
            const float u = keypoint.x / width;
            // I'm not sure why we need to mirror this coordinate, but it works like
            // this
            const float v      = 1 - keypoint.y / height;
            BaseVecT location  = texturizer.calculateTexCoordsInv(texH, boundingRect, TexCoords(u, v));
            keypoints3d[p_idx] = location;
        }
        std::vector<BaseVecT> features3d = keypoints3d;

        /*----------------------------------------------------------------------------*/
        // Transform descriptor from matrix row to float vector
        for (unsigned int row = 0; row < features3d.size(); ++row)
        {
            keypoints_map[features3d[row]] =
                std::vector<float>(descriptors.ptr(row), descriptors.ptr(row) + descriptors.cols);
        }
        printf("Trasnforming descriptor works\n");

        // Create material with default color and insert into face map
        Material material;
        material.m_texture = texH;
        std::array<unsigned char, 3> arr = {255, 255, 255};

        material.m_color = std::move(arr);            
            clusterMaterials.insert(clusterH, material);
        printf("creating default material works works\n");
        // Calculate tex coords
        // Insert material into face map for each face
        // Find unique vertices in cluster
        std::unordered_set<VertexHandle> verticesOfCluster;
        for (auto faceH : cluster.handles)
        {
            for (auto vertexH : m_mesh.getVerticesOfFace(faceH))
            {
                verticesOfCluster.insert(vertexH);
                // (doesnt insert duplicate vertices)
            }
        }
        printf("calculating tex coords works\n");
        // For each unique vertex in this cluster
        for (auto vertexH : verticesOfCluster)
        {
            // Calculate tex coords
            /**TexCoords texCoords = m_texturizer.calculateTexCoords(
                texH,
                boundingRect,
                m_mesh.getVertexPosition(vertexH)
            );*/
            auto point = m_mesh.getVertexPosition(vertexH);
            auto texelSize = m_textures[texH].m_texelSize;
            auto width = m_textures[texH].m_width;
            auto height = m_textures[texH].m_height;

            BaseVecT w =  point - ((boundingRect.m_vec1 * boundingRect.m_minDistA) + (boundingRect.m_vec2 * boundingRect.m_minDistB)
                    + boundingRect.m_supportVector);
            float u = (boundingRect.m_vec1 * (w.dot(boundingRect.m_vec1))).length() / texelSize / width;
            float v = (boundingRect.m_vec2 * (w.dot(boundingRect.m_vec2))).length() / texelSize / height;
            TexCoords texCoords = TexCoords(u,v);

            // Insert into result map
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
            printf("works1\n");
        }
        textureCount++;
    }
    printf("works2\n");
    return MaterializerResult<BaseVecT>(
        clusterMaterials,
        m_textures,
        vertexTexCoords,
        keypoints_map
    );
}

int main(int argc, char* argv[])
{
    //used to build the ply mesh
    //SimpleFinalizer<Vec> finalizer;
    lvr2::HalfEdgeMesh<Vec> mesh;
    //read, which ply file to analyse, what mode to use and how big the neighborhood should be
    //TO_DO: figure out, what the option file does and how i can use it in this project
    int number_neighbors = atoi(argv[1]);
    int mode = atoi(argv[2]);
    float distance_weighting = atof(argv[3]);
    float neighbor_weighting = atof(argv[4]);
    string data(argv[5]);

    TextureHandle tex = generateHeightDifferenceTexture(1);
    

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
    float avg_distance_point = 0;
    float added_distance = 0;
    
    // mode 1 --> "sechsecke"
    //mode 0 will be deleted after testing
    //TO_DO: make faces scaleable
    if(mode == 1){
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
            }
        }

        avg_distance = distance_weighting * avg_distance/(x_dim * 2 * y_dim);
        printf("Average Distance: %f\n",avg_distance);

        
        //Calculate the Vertice Positions
        for (float x = x_min; x < x_max + 1; x+=step_size)
        {        
            for (float y = y_min; y < y_max + 1; y++)
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
                
            }
        }
    }


    //Differenz zwischen Punktwolke und Mesh als Textur generieren
    //Farbe aus ColorMap nutzen
    //Channels sind data


    //1. Ziel --> alles in einer Farbe machen CHECK
    //Maybe Ziel --> Textur
    
    auto faceNormals = calcFaceNormals(mesh);
    ClusterBiMap<FaceHandle> clusterBiMap;
    BaseVector<float> currentCenterPoint;
    MeshHandleIteratorPtr<FaceHandle> iterator = mesh.facesBegin();
    auto newCluster = clusterBiMap.createCluster();
    for (size_t i = 0; i < mesh.numFaces(); i++)
    {
              
        clusterBiMap.addToCluster(newCluster,*iterator);
        ++iterator;
    }    
    //3. Ziel --> Farbe aus Ziel lesen
    auto vertexNormals = calcVertexNormals(mesh, faceNormals, *surface);
    TextureFinalizer<Vec> finalize(clusterBiMap);
    finalize.setVertexNormals(vertexNormals);
    
    
    MaterializerResult<Vec> matResult = generateMaterials(tex,clusterBiMap,mesh,*surface,faceNormals);
    printf("works3\n");
    finalize.setMaterializerResult(matResult);
    printf("works4\n");
    auto buffer = finalize.apply(mesh);
    printf("works5\n");
    //buffer->setTextures();
    buffer->addIntAtomic(1, "mesh_save_textures");
    buffer->addIntAtomic(1, "mesh_texture_image_extension");
    auto m = ModelPtr(new Model(buffer));
    printf("setting model works\n");    

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
    printf("save works\n");    
    

    return 0;
}