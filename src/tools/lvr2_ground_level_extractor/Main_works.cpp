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

template <typename BaseVecT>
PointsetSurfacePtr<BaseVecT> loadPointCloud(string &data)
{
    ModelPtr base_model = ModelFactory::readModel(data);
    PointBufferPtr base_buffer = base_model->m_pointCloud;
    PointsetSurfacePtr<Vec> surface;
    surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(base_buffer,"test");
    return surface;
}

int main(int argc, char* argv[])
{
    //Wird genutzt um Mesh aufzuspannen
    SimpleFinalizer<Vec> finalizer;
    lvr2::HalfEdgeMesh<Vec> mesh;
    //Punktwolke einlesen
    string data(argv[1]);
    auto surface = loadPointCloud<Vec>(data);
    ModelPtr base_model = ModelFactory::readModel(data);
    PointBufferPtr base_buffer = base_model->m_pointCloud;
    //Koordinaten aus FloatChannel lesen
    FloatChannel arr =  *(base_buffer->getFloatChannel("points"));   
    //Wenn selber machen muss --> alle Punkte durchgehen, kleinste und größte x,y,z Koordinaten finden
    auto bb = surface->getBoundingBox();
    auto max = bb.getMax();
    auto min = bb.getMin();
    ssize_t x_max = (ssize_t)max.x + 1;
    ssize_t x_min = (ssize_t)min.x - 1;
    ssize_t y_max = (ssize_t)max.y + 1;
    ssize_t y_min = (ssize_t)min.y - 1;
    ssize_t z_min = (ssize_t)min.z - 1;
    vector<size_t> indices;

    //Bounding Box von Umgebung ermitteln und richtige Auflösung festlegen
    for (ssize_t x = x_min; x < x_max + 1; x++)
    {        
        for (ssize_t y = y_min; y < y_max + 1; y++)
        {           

            indices.clear();

            surface->searchTree()->kSearch(Vec(x,y,z_min), 1, indices);
            auto index = indices[0];
            auto closest = arr[index];
            auto z = closest[2];          
            VertexHandle bottom_left = mesh.addVertex(Vec(x,y,z));

            surface->searchTree()->kSearch(Vec(x,y+1,z_min), 1, indices);
            index = indices[0];
            closest = arr[index];
            z = closest[2];   
            VertexHandle top_left = mesh.addVertex(Vec(x,y+1,z));

            surface->searchTree()->kSearch(Vec(x+1,y+1,z_min), 1, indices);
            index = indices[0];
            closest = arr[index];
            z = closest[2];  
            VertexHandle top_right = mesh.addVertex(Vec(x+1,y+1,z));

            surface->searchTree()->kSearch(Vec(x+1,y,z_min), 1, indices);
            index = indices[0];
            closest = arr[index];
            z = closest[2];  
            VertexHandle bottom_right = mesh.addVertex(Vec(x+1,y,z)); 

            mesh.addFace(bottom_left,bottom_right,top_left);
            mesh.addFace(top_left,bottom_right,top_right);
        }       
        
    }
    
    auto buffer = finalizer.apply(mesh);
    auto m = ModelPtr(new Model(buffer));

    PLYIO io;
    io.save(m,"mdp_3.ply");

    return 0;
}


/**
 * Pointbuffer in IO angucken
 * AN geometry attribute anhängen mit Maps aus basebuffer
 * über get channel channel zurückkriegen der attribute für die punktwolke
 * map mit name auf array mappen
 * variant channel map
 * wie sehen die buffer aus mit farben/echos
 * alle dateiformate bilden auf basebuffer ab
 * beispiel --> ios geben modelpointer zurück --> mashes und punktwolken
 * mash und pointbuffer angucken
 * attribe an faces UND vertices
 * gdel und geotiff nochmal ansehen
 * höherenrefferenzen als geotiff exportieren
 * 
 * aufgabenstellung stichpunkte für expose
 */