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
#include <math.h>

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
std::map<std::tuple<float, float>,VertexHandle> dict_point_float;

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
    return surface;
}

int main(int argc, char* argv[])
{
    //used to build the ply mesh
    SimpleFinalizer<Vec> finalizer;
    lvr2::HalfEdgeMesh<Vec> mesh;
    //read, which ply file to analyse, what mode to use and how big the neighborhood should be
    //TO_DO: figure out, what the option file does and how i can use it in this project
    int number_neighbors = atoi(argv[1]);
    int mode = atoi(argv[2]);
    string data(argv[3]);

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
    std::map<std::tuple<float, float>,VertexHandle>::iterator it_point_float;
    float avg_distance_point = 0;
    float added_distance = 0;
    
    //TO_DO: Anzahl an Punkten reduzieren --> schlaueres Plazieren der Faces
    //Entweder Reskursiv oder mit einem Dictionary
    //TO_DO: Filter der Annomalien beseitigt entwerfen
    //TO_DO: Punkte nach Distanz bewerten
    //mode 0 --> squares, mode 1 --> "sechsecke"
    //mode 0 will be deleted after testing
    //TO_DO: make faces scaleable
    if(mode == 0){
        float avg_distance = 0;
        float avg_z = 0;
        //iterate over every point that will be set and calculate the average distance from these points to their refference points
        //also calculate the average z value of the points that will be set
        for (ssize_t x = x_min; x < x_max + 1; x++)
        {        
            for (ssize_t y = y_min; y < y_max + 1; y++)
            {
                //TO_DO: Testen, ob das einen Unterschied macht
                //Antwort: ist so besser, average z sollte bessere abschätzung geben und average distance ist kleiner
                vector<size_t> indexes;
                vector<float> distances;
                surface->searchTree()->kSearch(Vec(x,y,z_min),number_neighbors,indexes,distances); 
                auto index = indexes[0];
                auto closest = arr[index];  
                surface->searchTree()->kSearch(Vec(x,y,closest[2]),number_neighbors,indexes,distances);
                
                /*float avg_distance_point = 0;
                for(float f : distances){
                    avg_distance_point += f;
                }
                avg_distance += avg_distance_point/number_neighbors;*/

                avg_distance += distances[0];
                float point_avg_z = 0;
                for (int i = 0; i < number_neighbors; i++)
                {
                    auto index = indexes[i];
                    auto nearest = arr[index];
                    point_avg_z += nearest[2]/number_neighbors;
                }  
                avg_z += point_avg_z;
            }
        }
        avg_z = avg_z/(x_dim*y_dim);
        //Hier mit 0.x multiplizieren oder nicht average über alle Nachbarn nehmen
        //TO_DO: how do i find a beter estiamte for this
        //0.2/.3 looked good 
        avg_distance = avg_distance/(x_dim * y_dim); 

        printf("Average Distance: %f\n",avg_distance);
        printf("Average z-Value: %f\n",avg_z);

        for (ssize_t x = x_min; x < x_max + 1; x++)
        {        
            for (ssize_t y = y_min; y < y_max + 1; y++)
            {
                //Vertice z Koordinate soll abhängig von den umliegenden Punkten gesetzt werden
                //Naiv zum Testen: für jeden Punkt aus dem Grid den nächstliegenden Punkt aus der Punktwolke ermitteln
                //und nach diesme Punkt die Z-Koordinate wählen

                //Weitere Ideen --> in einer Umgebung die ungefähre "Bodenhöhe" mittels der bereits gesetzten Punkte ermitteln
                //Anomalien erkennen?
                //probieren, für welches k das am besten funktioniert
                
                indices.clear();
                distances.clear();
                it_point = dict_point.find(make_tuple(x,y));
                if(it_point == dict_point.end()){ 
                    final_z = 0;
                    //calculate if the distance to other points is below average
                    //TO_DO --> should be more about dencity and less about distance
                    //should also be more about difference in the z coordinate
                    surface->searchTree()->kSearch(Vec(x,y,z_min),number_neighbors,indices,distances);                
                    added_distance = 0;
                    avg_distance_point = 0;
                    for(float f : distances){
                        added_distance += f;
                        //printf("float %f",f);
                    }
                    avg_distance_point = added_distance/number_neighbors;
                    //printf("%f > %f\n",avg_distance_point,avg_distance);
                    //printf("Zähler: %d",counter);
                    
                    //if the average distance is below average, we just use the average z value calculated earlier
                    if(avg_distance_point > avg_distance){
                        // final_z = avg_z;
                        /* Wir gucken uns das viereck links darunter an, mit uns als top right*/
                        it = dict_z.find(make_tuple(x-1,y));
                        if(it != dict_z.end()){
                            final_z += it->second * 1/3;
                        }
                        else{
                            final_z += avg_z * 1/3;
                        }
                        it = dict_z.find(make_tuple(x-1,y-1));
                        if(it != dict_z.end()){
                            final_z += it->second * 1/3;
                        }
                        else{
                            final_z += avg_z * 1/3;
                        }
                        it = dict_z.find(make_tuple(x,y-1));
                        if(it != dict_z.end()){
                            final_z += it->second * 1/3;
                        }
                        else{
                            final_z += avg_z * 1/3;
                        }
                    }else{
                        for (int i = 0; i < number_neighbors; i++)
                        {
                            auto index = indices[i];
                            auto nearest = arr[index];
                            final_z = final_z + nearest[2] * distances[i]/added_distance;
                        }  
                    }
                    //do this for every vertex handle
                    VertexHandle bottom_left = mesh.addVertex(Vec(x,y,final_z));
                    dict_point.emplace(make_tuple(x,y),bottom_left);
                    dict_z.emplace(make_tuple(x,y),final_z);
                }
                
                it_point = dict_point.find(make_tuple(x,y+1));
                if(it_point == dict_point.end()){ 
                    final_z = 0;
                    surface->searchTree()->kSearch(Vec(x,y + 1,z_min),number_neighbors,indices,distances);
                    avg_distance_point = 0;
                    added_distance = 0;
                    for(float f : distances){
                        added_distance += f;
                    }
                    avg_distance_point = added_distance/number_neighbors;
                    if(avg_distance_point > avg_distance){
                        //final_z = avg_z;
                        it = dict_z.find(make_tuple(x-1,y+1));
                        if(it != dict_z.end()){
                            final_z += it->second * 1/3;
                        }
                        else{
                            final_z += avg_z * 1/3;
                        }
                        it = dict_z.find(make_tuple(x-1,y));
                        if(it != dict_z.end()){
                            final_z += it->second * 1/3;
                        }
                        else{
                            final_z += avg_z * 1/3;
                        }
                        it = dict_z.find(make_tuple(x,y));
                        if(it != dict_z.end()){
                            final_z += it->second * 1/3;
                        }
                        else{
                            final_z += avg_z * 1/3;
                        }
                    }else{
                        for (int i = 0; i < number_neighbors; i++)
                        {
                            auto index = indices[i];
                            auto nearest = arr[index];
                            final_z = final_z + nearest[2] * distances[i]/added_distance;
                        }  
                    } 
                    
                    VertexHandle top_left = mesh.addVertex(Vec(x,y + 1,final_z));
                    dict_point.emplace(make_tuple(x,y+1),top_left);
                    dict_z.emplace(make_tuple(x,y+1),final_z);
                }

                it_point = dict_point.find(make_tuple(x+1,y+1));
                if(it_point == dict_point.end()){ 
                    final_z = 0;
                    surface->searchTree()->kSearch(Vec(x + 1,y + 1,z_min),number_neighbors,indices,distances);
                    avg_distance_point = 0;
                    added_distance = 0;
                    for(float f : distances){
                        added_distance += f;
                    }
                    avg_distance_point = added_distance/number_neighbors;
                    if(avg_distance_point > avg_distance){
                        //final_z = avg_z;
                        it = dict_z.find(make_tuple(x,y+1));
                        if(it != dict_z.end()){
                            final_z += it->second * 1/3;
                        }
                        else{
                            final_z += avg_z * 1/3;
                        }
                        it = dict_z.find(make_tuple(x,y));
                        if(it != dict_z.end()){
                            final_z += it->second * 1/3;
                        }
                        else{
                            final_z += avg_z * 1/3;
                        }
                        it = dict_z.find(make_tuple(x+1,y));
                        if(it != dict_z.end()){
                            final_z += it->second * 1/3;
                        }
                        else{
                            final_z += avg_z * 1/3;
                        }
                    }else{
                        for (int i = 0; i < number_neighbors; i++)
                        {
                            auto index = indices[i];
                            auto nearest = arr[index];
                            final_z = final_z + nearest[2] * distances[i]/added_distance;
                        }  
                    }
                
                    VertexHandle top_right = mesh.addVertex(Vec(x + 1,y + 1,final_z));
                    dict_point.emplace(make_tuple(x+1,y+1),top_right);
                    dict_z.emplace(make_tuple(x+1,y+1),final_z);
                }
                
                it_point = dict_point.find(make_tuple(x+1,y));
                if(it_point == dict_point.end()){ 
                    final_z = 0;
                    surface->searchTree()->kSearch(Vec(x + 1,y,z_min),number_neighbors,indices,distances);
                    avg_distance_point = 0;
                    added_distance = 0;
                    for(float f : distances){
                        added_distance += f;
                    }
                    avg_distance_point = added_distance/number_neighbors;
                    if(avg_distance_point > avg_distance){
                        //final_z = avg_z;
                        it = dict_z.find(make_tuple(x,y));
                        if(it != dict_z.end()){
                            final_z += it->second * 1/3;
                        }
                        else{
                            final_z += avg_z * 1/3;
                        }
                        it = dict_z.find(make_tuple(x,y-1));
                        if(it != dict_z.end()){
                            final_z += it->second * 1/3;
                        }
                        else{
                            final_z += avg_z * 1/3;
                        }
                        it = dict_z.find(make_tuple(x+1,y-1));
                        if(it != dict_z.end()){
                            final_z += it->second * 1/3;
                        }
                        else{
                            final_z += avg_z * 1/3;
                        }
                    }else{
                        for (int i = 0; i < number_neighbors; i++)
                        {
                            auto index = indices[i];
                            auto nearest = arr[index];
                            final_z = final_z + nearest[2] * distances[i]/added_distance;
                        }  
                    }
                    
                    VertexHandle bottom_right = mesh.addVertex(Vec(x + 1,y,final_z)); 
                    dict_point.emplace(make_tuple(x+1,y),bottom_right);
                    dict_z.emplace(make_tuple(x+1,y),final_z);
                }
                //Combine the Handles to faces
                VertexHandle bottom_left = dict_point.find(make_tuple(x,y))->second;
                VertexHandle top_left = dict_point.find(make_tuple(x,y+1))->second;
                VertexHandle top_right = dict_point.find(make_tuple(x+1,y+1))->second;
                VertexHandle bottom_right =  dict_point.find(make_tuple(x+1,y))->second;
                mesh.addFace(bottom_left,bottom_right,top_left);
                mesh.addFace(top_left,bottom_right,top_right);
            }       
            
        }
    }
    else if(mode == 1){
        //+0.5 so that the faces line up 
        ssize_t y_counter = 0;
        ssize_t x_counter = 0;
        float h = 0.5;
        for (float x = x_min*2; x < (x_max+1)*2; x+= 1)
        {        
            for (float y = y_min; y < y_max+1; y++)
            {
                //height of the triangle faces
                //used to calculate where the coordinates of the vertices are
                

                indices.clear();
                //Center
                float x_coordinate = 4 * x;
                float y_coordinate = 4 * y;
                printf("x-Coord = %f, y-Coord = %f \n",x_coordinate,y_coordinate);
                it_point_float = dict_point_float.find(make_tuple(x_coordinate,y_coordinate));
                if(it_point_float == dict_point_float.end()){
                    final_z = 0;                
                    surface->searchTree()->kSearch(Vec(x/2,y,z_min),number_neighbors,indices);
                    for (int i = 0; i < number_neighbors; i++)
                    {
                        auto index = indices[i];
                        auto nearest = arr[index];
                        final_z = final_z + nearest[2] * 1/number_neighbors;
                    }  
                    VertexHandle center = mesh.addVertex(Vec(x/2,y,final_z));
                    dict_point_float.emplace(make_tuple(x_coordinate,y_coordinate),center);
                }
                
                //Center Right
                x_coordinate = 4 * (x+0.5);
                y_coordinate = 4 * y;
                printf("x-Coord = %f, y-Coord = %f \n",x_coordinate,y_coordinate);
                it_point_float == dict_point_float.find(make_tuple(x_coordinate,y_coordinate));
                if(it_point_float == dict_point_float.end()){
                    final_z = 0;                
                    surface->searchTree()->kSearch(Vec(x/2+0.5,y,z_min),number_neighbors,indices);
                    for (int i = 0; i < number_neighbors; i++)
                    {
                        auto index = indices[i];
                        auto nearest = arr[index];
                        final_z = final_z + nearest[2] * 1/number_neighbors;
                    }  
                    VertexHandle center_right = mesh.addVertex(Vec(x/2+0.5,y,final_z));
                    dict_point_float.emplace(make_tuple(x_coordinate,y_coordinate),center_right);
                }

                //Center Left
                x_coordinate = 4 * (x-0.5);
                y_coordinate = 4 * y;
                printf("x-Coord = %f, y-Coord = %f \n",x_coordinate,y_coordinate);
                it_point_float == dict_point_float.find(make_tuple(x_coordinate,y_coordinate));
                if(it_point_float == dict_point_float.end()){
                    final_z = 0;                
                    surface->searchTree()->kSearch(Vec(x/2-0.5,y,z_min),number_neighbors,indices);
                    for (int i = 0; i < number_neighbors; i++)
                    {
                        auto index = indices[i];
                        auto nearest = arr[index];
                        final_z = final_z + nearest[2] * 1/number_neighbors;
                    }  
                    VertexHandle center_left = mesh.addVertex(Vec(x/2-0.5,y,final_z));
                    dict_point_float.emplace(make_tuple(x_coordinate,y_coordinate),center_left);
                }

                //Bottom Right
                x_coordinate = 4 * (x+0.25);
                y_coordinate = 4 * (y-h);
                printf("x-Coord = %f, y-Coord = %f \n",x_coordinate,y_coordinate);
                it_point_float == dict_point_float.find(make_tuple(x_coordinate,y_coordinate));
                if(it_point_float == dict_point_float.end()){
                    final_z = 0;                
                    surface->searchTree()->kSearch(Vec(x/2+0.25,y-h,z_min),number_neighbors,indices);
                    for (int i = 0; i < number_neighbors; i++)
                    {
                        auto index = indices[i];
                        auto nearest = arr[index];
                        final_z = final_z + nearest[2] * 1/number_neighbors;
                    }  
                    VertexHandle bottom_right = mesh.addVertex(Vec(x/2+0.25,y-h,final_z));
                    dict_point_float.emplace(make_tuple(x_coordinate,y_coordinate),bottom_right);
                }

                //Bottom Left
                x_coordinate = 4 * (x-0.25);
                y_coordinate = 4 * (y-h);
                printf("x-Coord = %f, y-Coord = %f \n",x_coordinate,y_coordinate);
                it_point_float == dict_point_float.find(make_tuple(x_coordinate,y_coordinate));
                if(it_point_float == dict_point_float.end()){
                    final_z = 0;                
                    surface->searchTree()->kSearch(Vec(x/2-0.25,y-h,z_min),number_neighbors,indices);
                    for (int i = 0; i < number_neighbors; i++)
                    {
                        auto index = indices[i];
                        auto nearest = arr[index];
                        final_z = final_z + nearest[2] * 1/number_neighbors;
                    }  
                    VertexHandle bottom_left = mesh.addVertex(Vec(x/2-0.25,y-h,final_z));
                    dict_point_float.emplace(make_tuple(x_coordinate,y_coordinate),bottom_left);
                }
                
                //Top_right
                x_coordinate = 4 * (x+0.25);
                y_coordinate = 4 * (y+h);
                printf("x-Coord = %f, y-Coord = %f \n",x_coordinate,y_coordinate);
                it_point_float == dict_point_float.find(make_tuple(x_coordinate,y_coordinate));
                if(it_point_float == dict_point_float.end()){
                    final_z = 0;                
                    surface->searchTree()->kSearch(Vec(x/2+0.25,y+h,z_min),number_neighbors,indices);
                    for (int i = 0; i < number_neighbors; i++)
                    {
                        auto index = indices[i];
                        auto nearest = arr[index];
                        final_z = final_z + nearest[2] * 1/number_neighbors;
                    }  
                    VertexHandle top_right = mesh.addVertex(Vec(x/2+0.25,y+h,final_z));
                    dict_point_float.emplace(make_tuple(x_coordinate,y_coordinate),top_right);
                }

                //Top Left
                x_coordinate = 4 * (x-0.25);
                y_coordinate = 4 * (y+h);
                printf("x-Coord = %f, y-Coord = %f \n",x_coordinate,y_coordinate);
                it_point_float == dict_point_float.find(make_tuple(x_coordinate,y_coordinate));
                if(it_point_float == dict_point_float.end()){
                    final_z = 0;                
                    surface->searchTree()->kSearch(Vec(x/2-0.25,y+h,z_min),number_neighbors,indices);
                    for (int i = 0; i < number_neighbors; i++)
                    {
                        auto index = indices[i];
                        auto nearest = arr[index];
                        final_z = final_z + nearest[2] * 1/number_neighbors;
                    }  
                    VertexHandle top_left = mesh.addVertex(Vec(x/2-0.25,y+h,final_z));
                    dict_point_float.emplace(make_tuple(x_coordinate,y_coordinate),top_left);
                }
                
                VertexHandle center = dict_point_float.find(make_tuple(x*4,y*4))->second;
                VertexHandle center_right = dict_point_float.find(make_tuple((x+0.5)*4,y*4))->second;
                VertexHandle center_left = dict_point_float.find(make_tuple((x-0.5)*4,y*4))->second;
                VertexHandle bottom_right = dict_point_float.find(make_tuple((x+0.25)*4,(y-h)*4))->second;
                VertexHandle bottom_left = dict_point_float.find(make_tuple((x-0.25)*4,(y-h)*4))->second;
                VertexHandle top_right = dict_point_float.find(make_tuple((x+0.25)*4,(y+h)*4))->second;
                VertexHandle top_left = dict_point_float.find(make_tuple((x-0.25)*4,(y+h)*4))->second;
                printf("Start: y-Counter = %ld, x-Counter = %ld \n",y_counter,x_counter);
                mesh.addFace(center,top_right,top_left);
                printf("center,top_right,top_left works\n");
                mesh.addFace(center,bottom_left,bottom_right);
                printf("center,bottom_left,bottom_right\n");
                mesh.addFace(center,bottom_right,center_right);
                printf("center,bottom_right,center_right\n");
                mesh.addFace(center,center_right,top_right);
                printf("center,center_right,top_right\n");
                mesh.addFace(center,top_left,center_left);
                printf("center,top_left,center_left\n");
                mesh.addFace(center,center_left,bottom_left);
                printf("center,center_left,bottom_left\n");
                y_counter++;
            }   
            x_counter++;
        }
    }
    auto buffer = finalizer.apply(mesh);
    auto m = ModelPtr(new Model(buffer));
    
    PLYIO io;
    size_t pos = data.find_last_of("/");  
    string name = data.substr(pos+1);
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
    auto time = oss.str();
    io.save(m, time + "_" + to_string(mode) + "_neighborhood_"+ to_string(number_neighbors) + "_groundlevel_" + name);

    return 0;
}