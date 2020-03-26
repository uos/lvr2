#include "LVRChunkedMeshBridge.hpp"

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkActor.h>
#include <vtkTriangle.h>
#include <vtkProperty.h>
#include <vtkImageData.h>
#include <vtkTexture.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkCellData.h>


#include "LVRBoundingBoxBridge.hpp"
#include <vtkPolyDataMapper.h>


#include <chrono>
#include <queue>
using namespace lvr2;


LVRChunkedMeshBridge::LVRChunkedMeshBridge(std::string file, vtkSmartPointer<vtkRenderer> renderer, std::vector<std::string> layers, size_t cache_size) : m_chunkManager(file, cache_size), m_renderer(renderer), m_layers(layers), m_cacheSize(cache_size)
{
    getNew_ = false;
    running_ = true;
    worker = std::thread(&LVRChunkedMeshBridge::highResWorker, this);

}


void LVRChunkedMeshBridge::highResWorker()
{
    while(running_)
    {
       std::unique_lock<std::mutex> l(mutex);
       while(!getNew_) 
       {
           cond_.wait(l);
       }
       getNew_ = false;
       std::vector<size_t> visible_indices = m_highResIndices;
       std::vector<BaseVector<float> > visible_centroids = m_highResCentroids;
       // delta for new fetch
       // basically prevents an endless loop of reloads.
       BaseVector<float> diff = m_region.getCentroid() - m_lastRegion.getCentroid();
       if(!(std::abs(diff[0]) > 1.0 || std::abs(diff[1]) > 1.0 || std::abs(diff[2]) > 1.0))
       {
           l.unlock();
           continue;
       }
       m_lastRegion = m_region;
       l.unlock();

       // Needed for consistency for underlying buffers of actors.
       // see below.
       auto old_highRes = m_highRes;
       m_highRes.clear();

       // check if there is only one layer.
       // if yes use the chunks from the "lowRes" layer.
       if(m_layers.size() > 1)
       {
            m_chunkManager.extractArea(m_region, m_highRes, m_layers[0]);
       }
       else
       {
            for(auto& index : visible_indices)
            {
                m_highRes.insert({index, m_chunks[index]});
            }
       }

       // Consistency for underlying buffers of actors.
       // The chunkmanagers may have loaded new meshbuffers while we use old ones
       // Use the old ones otherwise we need to rebuild and copy the actors.
       // New ones are ignored in that case.
       for(auto& it : m_highRes)
       {
           if(old_highRes.find(it.first) != old_highRes.end())
           {
               m_highRes.at(it.first) = old_highRes.at(it.first);
               // TODO Not clearing may result in duplicates.
               //old_highRes.erase(it.first);
           }
       }
        
       // This should be a cache based on 2 queues 
       // and the distance of the centroids of the chunk to the current frustum centroid.
       size_t numCopy =  m_cacheSize - m_highResIndices.size();
       if(numCopy > 0)
       {
           // This is the caching based on the distance of the centroids
           // to the centroids from the current visible region.
           // ALL this is far from optimal.
           typedef std::pair<float, size_t> IndexPair;
           typedef std::pair<float, BaseVector<float> > CentroidPair;
           std::priority_queue<IndexPair, std::vector<IndexPair>, CompareDistancePair<size_t> > index_queue;
           std::priority_queue<CentroidPair, std::vector<CentroidPair>, CompareDistancePair<BaseVector<float> > > centroid_queue;
           BaseVector<float> center = m_region.getCentroid();
           if(m_lastCentroids.size() > 0)
           {
               float distance = m_lastCentroids[0].distance(center);
               index_queue.push({distance, m_lastIndices[0]});
               centroid_queue.push({distance, m_lastCentroids[0]});
               for(size_t i = 1; i < m_lastCentroids.size(); ++i)
               {

                   float distance = m_lastCentroids[i].distance(center);
                   if(index_queue.size() < numCopy)
                   {
                       centroid_queue.push({distance, m_lastCentroids[i]});
                       index_queue.push({distance, m_lastIndices[i]});
                   }
                   else if(index_queue.top().first > distance)
                   {
                       index_queue.pop();
                       index_queue.push({distance, m_lastIndices[i]});
                       centroid_queue.pop();
                       centroid_queue.push({distance, m_lastCentroids[i]});
                   }
               }
           }

           m_lastIndices.clear();
           m_lastCentroids.clear();

           // add current visible to cache
           for(size_t i = 0; i < visible_indices.size(); ++i)
           {
                m_lastIndices.push_back(visible_indices[i]);
                m_lastCentroids.push_back(visible_centroids[i]);
            
           }
            
           // Build new cache.
           while(!index_queue.empty())
           {
                size_t index = index_queue.top().second;
                // here is the error it maybe already present in highres
                //  and deleted from old_highres....
                index_queue.pop();
                m_highRes.insert({index, old_highRes[index]});
                m_lastIndices.push_back(index);
                auto centroid = centroid_queue.top().second;
                centroid_queue.pop();
                m_lastCentroids.push_back(centroid);
           }
       }

       // compute and copy actors
       std::unordered_map<size_t, vtkSmartPointer<vtkActor>> tmp_highResActors;
       for(auto it = m_highRes.begin(); it != m_highRes.end(); ++it)
       { 
           auto chunk = *it;
           size_t id = chunk.first;
           lvr2::MeshBufferPtr meshbuffer = chunk.second;

           if(m_highResActors.find(id) == m_highResActors.end())
           {
               tmp_highResActors.insert({id, computeMeshActor(id, meshbuffer)});
           }
           else
           {
               tmp_highResActors.insert({id, m_highResActors[id]});
           }
       }


       // add highres dummies because their maybe lowres chunks
       // which do not exist in the highres map
       for(auto& it: visible_indices)
       {
            if(tmp_highResActors.find(it) == tmp_highResActors.end())
            {
                auto mptr = MeshBufferPtr();
                tmp_highResActors.insert({it, computeMeshActor(it, mptr)});
            }
       }

       // Syncing with main thread.
       actorMap remove_actors;
       actorMap new_actors;
       for(auto& it: m_highResActors)
       {
           if(tmp_highResActors.find(it.first) == tmp_highResActors.end())
           {
               //vtkSmartPointer<vtkActor> copy = vtkSmartPointer<vtkActor>::New();
               //it.second->ShallowCopy(copy);
               remove_actors.insert({it.first, it.second});
           }
       }

       for(auto& it: tmp_highResActors)
       {
           if(m_highResActors.find(it.first) == m_highResActors.end())
           { 
               //vtkSmartPointer<vtkActor> copy = vtkSmartPointer<vtkActor>::New();
               //it.second->ShallowCopy(copy);
               new_actors.insert({it.first, it.second});
           }
       }

       Q_EMIT updateHighRes(remove_actors, new_actors);
//       while(!release)
//       {
//            mw_cond.wait(main_lock);
//       }
       m_highResActors = tmp_highResActors; 
       old_highRes.clear();
       release = false;
//       l.unlock();
    }

}

void LVRChunkedMeshBridge::fetchHighRes(BoundingBox<BaseVector<float> > bb,
                                        std::vector<size_t> indices,
                                        std::vector<BaseVector<float> > centroids)
{
    // FUCKING SYNCHRONIZATION
    std::unique_lock<std::mutex> l(mutex);
    m_region = bb;
    m_highResIndices = indices;
    m_highResCentroids = centroids;
    getNew_ = true;
    l.unlock();
    cond_.notify_all();

}

void LVRChunkedMeshBridge::addInitialActors(vtkSmartPointer<vtkRenderer> renderer)
{
    std::cout << "Generating actors " << std::endl;
    auto bb = m_chunkManager.getBoundingBox();
    BaseVector<float> centroid = bb.getCentroid();

    LVRBoundingBoxBridge bbridge(bb);
    renderer->AddActor(bbridge.getActor());
    bbridge.setVisibility(true);
    
    auto min = bb.getLongestSide(); 
    auto cam_position = centroid + BaseVector<float>(min, min, min);
//    Vector3d cam_origin(cam_position[0], cam_position[1], cam_position[2]);
    Vector3d cam_origin(0, 0, 0);
    Vector3d view_up(1.0, 0.0, 0.0);
 //   Vector3d focal_point(centroid[0], centroid[1], centroid[2]);
    Vector3d focal_point(centroid[0], centroid[1], centroid[2]);

    renderer->GetActiveCamera()->SetPosition(cam_origin.x(), cam_origin.y(), cam_origin.z());
    renderer->GetActiveCamera()->SetFocalPoint(focal_point.x(), focal_point.y(), focal_point.z());
    renderer->GetActiveCamera()->SetViewUp(view_up.x(), view_up.y(), view_up.z());


    // maybe use pop back..
    // maybe copy it..
    if(m_layers.size() == 1)
    {
    
        m_chunkManager.extractArea(bb, m_chunks, m_layers[0]);
    }
    else{
    
        m_chunkManager.extractArea(bb, m_chunks, m_layers[1]);
    }

    std::vector<size_t> hashes;

    // THIS MAY NOT GIVE THE EXPECTED RESULT
    // THE CENTROID OF THE CHUNK...
    std::vector<BaseVector<float> > centroids;
    for(auto& chunk:m_chunks)
    {
        hashes.push_back(chunk.first);
        FloatChannel vertices = *(chunk.second->getFloatChannel("vertices"));
        BaseVector<float> p = vertices[0];
        float minX = p.x;
        float minY = p.y;
        float minZ = p.z;
        float maxX = p.x;
        float maxY = p.y;
        float maxZ = p.z;

        for(size_t i = 0; i < vertices.numElements(); ++i)
        {
            p = vertices[i];
            minX = std::min(minX, p.x);
            minY = std::min(minY, p.y);
            minZ = std::min(minZ, p.z);

            maxX = std::max(maxX, p.x);
            maxY = std::max(maxY, p.y);
            maxZ = std::max(maxZ, p.z);
        }
        BaseVector<float> v1(minX, minY, minZ);
        BaseVector<float> v2(maxX, maxY, maxZ);

        BoundingBox<BaseVector<float> > chunk_bb(v1, v2);
        centroids.push_back(chunk_bb.getCentroid());
    }
    
    m_oct = std::make_unique<MeshOctree<BaseVector<float> >> (m_chunkManager.getChunkSize(),
            hashes, centroids, bb);


    if(m_layers.size() > 1)
    {
        std::cout << "Computing actors " << std::endl;
        computeMeshActors();
        std::cout << "Adding actors." << std::endl;
        for(auto& actor: m_chunkActors)
        {
            renderer->AddActor(actor.second);
            actor.second->VisibilityOff();
        }
        std::cout << "Added " << m_chunkActors.size() << " actors" << std::endl;
    }
}

void LVRChunkedMeshBridge::getActors(double planes[24],
        std::vector<BaseVector<float> >& centroids,
        std::vector<size_t> & indices)
{

    m_oct->intersect(planes, centroids, indices);

}

void LVRChunkedMeshBridge::computeMeshActors()
{

    size_t i = 0;
    for(auto it = m_chunks.begin(); it != m_chunks.end(); ++it)
    {
        auto chunk = *it;
        size_t id = chunk.first;
        lvr2::MeshBufferPtr meshbuffer = chunk.second;
        vtkSmartPointer<vtkActor> actor = computeMeshActor(id, meshbuffer);
        m_chunkActors.insert({id, actor});

    }


    std::cout << lvr2::timestamp << "Done actor computation" << std::endl;
}

vtkSmartPointer<vtkActor> LVRChunkedMeshBridge::computeMeshActor(size_t& id, MeshBufferPtr& meshbuffer)
{

    vtkSmartPointer<vtkActor> meshActor = vtkSmartPointer<vtkActor>::New();
    if(meshbuffer)
    {
//        meshActor         
        vtkSmartPointer<vtkPolyData> mesh = vtkSmartPointer<vtkPolyData>::New();
        // Parse vertex and index buffer
        size_t n_v, n_i, n_c;
        size_t w_color;
        n_v = meshbuffer->numVertices();
        floatArr vertices = meshbuffer->getVertices();
        n_i = meshbuffer->numFaces();
        indexArray indices = meshbuffer->getFaceIndices();
        n_c = n_v;
        ucharArr colors = meshbuffer->getVertexColors(w_color);


        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkFloatArray> pts_data = vtkSmartPointer<vtkFloatArray>::New();
        pts_data->SetNumberOfComponents(3);
        pts_data->SetVoidArray(meshbuffer->getVertices().get(), n_v * 3, 1);
        points->SetData(pts_data);


        vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
        scalars->SetNumberOfComponents(3);
        scalars->SetName("Colors");
//        if(colors)
//        {
//            scalars->SetVoidArray(colors.get(), n_v * w_color, 0,);
//        }


        // Triangle indices
        vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();
        vtkSmartPointer<vtkIdTypeArray> tri_data = vtkSmartPointer<vtkIdTypeArray>::New();
//        tri_data->Resize(n_i * 4);
        vtkIdType* tri_buf = new vtkIdType[n_i * 4];
        for(size_t i = 0; i < n_i; i++)
        {
            size_t index = 3 * i;
            size_t i2    = 4 * i;
            tri_buf[i2 + 0] = static_cast<vtkIdType>(3);
            tri_buf[i2 + 1] = static_cast<vtkIdType>(indices[index + 0]);
            tri_buf[i2 + 2] = static_cast<vtkIdType>(indices[index + 1]);
            tri_buf[i2 + 3] = static_cast<vtkIdType>(indices[index + 2]);
//            (*tri_data).InsertValue(i2 + 0 , static_cast<vtkIdType>(3));
//            (*tri_data).InsertValue(i2 + 1 , static_cast<vtkIdType>(indices[index + 0]));
//            (*tri_data).InsertValue(i2 + 2 , static_cast<vtkIdType>(indices[index + 1]));
//            (*tri_data).InsertValue(i2 + 3 , static_cast<vtkIdType>(indices[index + 2]));
        }

        tri_data->SetVoidArray(tri_buf, n_i * 4, 0, vtkIdTypeArray::VTK_DATA_ARRAY_DELETE);
        //tri_data->SetVoidArray(tri_buf, n_i * 4, 1);
        triangles->SetCells(n_i, tri_data);

        // add points, triangles and if given colors
        mesh->SetPoints(points);
        mesh->SetPolys(triangles);
        if(colors)
        {
            mesh->GetPointData()->SetScalars(scalars);
        }

        vtkSmartPointer<vtkPolyDataMapper> mesh_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();

        #ifdef LVR2_USE_VTK5
        mesh_mapper->SetInput(mesh);
        #else
        mesh_mapper->SetInputData(mesh);
        #endif
        meshActor->SetMapper(mesh_mapper);
        meshActor->GetProperty()->BackfaceCullingOff();
        vtkSmartPointer<vtkPolyDataMapper> wireframe_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();

        #ifdef LVR2_USE_VTK5
        wireframe_mapper->SetInput(mesh);
        #else
        wireframe_mapper->SetInputData(mesh);
        #endif

        // TODO add wireframe stuff
        //m_wireframeActor = vtkSmartPointer<vtkActor>::New();
        //m_wireframeActor->ShallowCopy(meshActor);
        //m_wireframeActor->SetMapper(wireframe_mapper);
        //vtkSmartPointer<vtkProperty> p = vtkSmartPointer<vtkProperty>::New();
        //p->DeepCopy(meshActor->GetProperty());
        //p->SetRepresentationToWireframe();
        //m_wireframeActor->SetProperty(p);

        // TODO THIS WAS A FUNCTION SETBASECOLOR
        vtkSmartPointer<vtkProperty> p = meshActor->GetProperty();
        p->SetColor(0.9, 0.9, 0.9);
        meshActor->SetProperty(p);
        // TODO add wireframe stuff.
        //p = m_wireframeActor->GetProperty();
        //float inv_r = (float)1 - r;
        //float inv_g = (float)1 - g;
        //float inv_b = (float)1 - b;
        //p->SetColor(inv_r, inv_g, inv_b);
        //m_wireframeActor->SetProperty(p);
        mesh_mapper->Update();

        //setBaseColor(0.9, 0.9, 0.9);
        //meshActor->setID(id);

    }

    return meshActor;
}

