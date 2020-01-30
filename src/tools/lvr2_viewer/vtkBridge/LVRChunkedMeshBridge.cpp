#include "LVRChunkedMeshBridge.hpp"

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkPoints.h>
#include <vtkActor.h>
#include <vtkTriangle.h>
#include <vtkProperty.h>
#include <vtkImageData.h>
#include <vtkTexture.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkCellData.h>

#include <omp.h>

using namespace lvr2;

LVRChunkedMeshBridge::LVRChunkedMeshBridge(std::string file) : m_chunkManager(file)
{
    getNew_ = false;
    dist_ = 40.0;
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

//       std::cout << "Get new worker" << std::endl;
//       omp_lock_t writelock;
//
//       omp_init_lock(&writelock);

       std::cout << lvr2::timestamp << "Chunkmanager bb " << m_chunkManager.getBoundingBox() << std::endl;
       BaseVector<float> diff = m_region.getCentroid() - m_lastRegion.getCentroid();
//       std::cout << "Centroid diff: " << diff << std::endl;
       if(!(std::abs(diff[0]) > 1.0 || std::abs(diff[1]) > 1.0 || std::abs(diff[2]) > 1.0))
       {
           getNew_ = false;
           l.unlock();
           continue;
       }
    
       std::cout << "Last region " << m_lastRegion << std::endl;
       std::cout << "New region " << m_region << std::endl;
       m_lastRegion = m_region;
       auto old_highRes = m_highRes;
       m_highRes.clear();

       std::cout << lvr2::timestamp << "Request from cm " << m_region << std::endl;
       m_chunkManager.extractArea(m_region, m_highRes, "mesh1");

       for(auto& it : m_highRes)
       {
            if(old_highRes.find(it.first) != old_highRes.end())
            {
                m_highRes.at(it.first) = old_highRes.at(it.first);
            }
       }
       old_highRes.clear();
       std::cout << lvr2::timestamp << "got from cm " << m_highRes.size() << std::endl;
    
//      m_highResActors.clear();

        std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor>> tmp_highResActors;

//       m_highResActors = std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor>>();
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
                tmp_highResActors.insert({id, m_highResActors.at(id)});
               }

       }
       auto old_actors = m_highResActors;
       m_highResActors = tmp_highResActors; 
       std::cout << lvr2::timestamp << "Got " << m_highResActors.size() << " highres" << std::endl;
       getNew_ = false;
    
       Q_EMIT updateHighRes(old_actors, m_highResActors);
    
       l.unlock();
    }

}

//void LVRChunkedMeshBridge::fetchHighRes(double position[3],
//                                        double dir[3],
//                                        double up[3])
//        
void LVRChunkedMeshBridge::fetchHighRes(BoundingBox<BaseVector<float> > bb)
{   // std::cout << "Up vec " << up_vec << "\n" <<
    //             "projec " << proj_vec << "\n" <<
    //             "perp   " << perp_vec    << "\n" << 
    //             "posit  " << eye    << "\n" << std::endl;

//    std::cout << "get Highres" << std::endl;
//    BaseVector<float> eye(position[0], position[1], position[2]);
//    BaseVector<float> up_vec(up[0], up[1], up[2]);
//    up_vec.normalize();
//    BaseVector<float> proj_vec(dir[0], dir[1], dir[2]);
//    proj_vec.normalize();
//    BaseVector<float> perp_vec = proj_vec.cross(up_vec);
//    perp_vec.normalize();
//    eye += (proj_vec * -128);
//    BoundingBox<BaseVector<float> > n_bb;
    
//    for(int i = 0; i < 2; ++i)
//    {
//        BaseVector<float> perp;
//        if(i)
//        {
//            perp = (perp_vec * dist_ * (-1));
//        }
//        else{
//            perp = (perp_vec * dist_);
//        }
//
//        for(int j = 0; j < 2; ++j)
//        {
//            BaseVector<float> n_up;
//            if(i)
//            {
//                n_up = (up_vec * dist_ * (-1));
//            }
//            else{
//                n_up = (up_vec * dist_);
//            }
//            for(int k = 0; k < 2; ++k)
//            { 
//                
//                BaseVector<float> n_proj;
//                if(k)
//                {
//                    n_proj = (proj_vec * dist_ * (-1));
//                }
//                else{
//                    n_proj = (proj_vec * dist_);
//                }
//
//                BaseVector<float> tmp = eye + perp + n_up + n_proj;
//                n_bb.expand(tmp);
//            }
//        }
//    }
//    auto max = n_bb.getMax();
//    auto min = n_bb.getMin();

    //float min_val = std::min(min[0], std::min(min[1], min[2]));
    //float max_val = std::max(max[0], std::max(max[1], max[2]));
    //min = BaseVector<float>(min_val, min_val, min_val);
    //max = BaseVector<float>(max_val, max_val, max_val);
    //n_bb.expand(min);
    //n_bb.expand(max);

    std::unique_lock<std::mutex> l(mutex);
    m_region = bb;
    getNew_ = true;
    cond_.notify_all();
    //l.unlock();

}

void LVRChunkedMeshBridge::addInitialActors(vtkSmartPointer<vtkRenderer> renderer)
{
    std::cout << "Generating actors " << std::endl;
    auto bb = m_chunkManager.getBoundingBox();
    BaseVector<float> centroid = bb.getCentroid();

    Vector3d cam_origin(centroid[0], centroid[1], centroid[2]);
    Vector3d view_up(1.0, 0.0, 0.0);
    Vector3d focal_point(0.0, 0.0, 0.0);

    renderer->GetActiveCamera()->SetPosition(cam_origin.x(), cam_origin.y(), cam_origin.z());
    renderer->GetActiveCamera()->SetFocalPoint(focal_point.x(), focal_point.y(), focal_point.z());
    renderer->GetActiveCamera()->SetViewUp(view_up.x(), view_up.y(), view_up.z());

    //    lvr2::BoundingBox<BaseVector<float> > bb(max * (-1), max);
    //    std::vector<vtkSmartPointer<MeshChunkActor> > actors;
    //    bb = BoundingBox<BaseVector<float> >(centroid, max);
    m_chunkManager.extractArea(bb, m_chunks, "mesh3");
    std::vector<size_t> hashes;
    std::vector<BaseVector<float> > centroids;
    for(auto& chunk:m_chunks)
    {
        hashes.push_back(chunk.first);
        FloatChannel vertices = *(chunk.second->getFloatChannel("vertices"));
        BoundingBox<BaseVector<float>> bb;
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
    m_oct = new MeshOctree<BaseVector<float> > (m_chunkManager.getChunkSize(),
            hashes, centroids, bb);

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

void LVRChunkedMeshBridge::getActors(double planes[24],
        std::vector<BaseVector<float> >& centroids,
        std::vector<size_t> & indices)

    // std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> >& actors)
{

    m_oct->intersect(planes, centroids, indices);

    //    indices.resize(centroids.size());
    //
    //    #pragma omp parallel for
    //    for(int i = 0; i < indices.size(); ++i)
    //    {
    //
    //        if(m_chunkActors.find(indices[i]) == m_chunkActors.end())
    //        {
    //            std::cout << "KEY NOT FOUND: " << i << std::endl;
    //        }
    //    }


    //    m_chunkManager.extractArea(bb, m_chunks);

    //    for(auto& chunk:m_chunks)
    //    {
    //        indices.push_back(chunk.first);
    //    }

    //computeMeshActors();
    //    actors = m_chunkActors;
    //    for(const auto& chunks: m_chunkActors)
    //    {
    //        actors.push_back(chunks.second);
    //    }

}

//void LVRChunkedMeshBridge::getHighDetailActor(std::vector<size_t>& indices)
//{
//}



void LVRChunkedMeshBridge::computeMeshActors()
{

    omp_lock_t writelock;

    omp_init_lock(&writelock);

    std::cout << lvr2::timestamp << "Start actor computation" << std::endl;
    //    for(const auto& chunk: m_chunks)
#pragma omp parallel
    {
#pragma omp single
        { 
            for(auto it = m_chunks.begin(); it != m_chunks.end(); ++it)
#pragma omp task
            {
                auto chunk = *it;
                size_t id = chunk.first;
                lvr2::MeshBufferPtr meshbuffer = chunk.second;

                omp_set_lock(&writelock);
                m_chunkActors.insert({id, computeMeshActor(id, meshbuffer)});
                omp_unset_lock(&writelock);

            }
        }
    }


    std::cout << lvr2::timestamp << "Done actor computation" << std::endl;
    // clear the chunks array
    //m_chunks.clear();
    //std::unordered_map<size_t, MeshBufferPtr>().swap(m_chunks);

}

vtkSmartPointer<MeshChunkActor> LVRChunkedMeshBridge::computeMeshActor(size_t& id, MeshBufferPtr& meshbuffer)
{
    vtkSmartPointer<MeshChunkActor> meshActor;

    if(meshbuffer)
    {
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

        //vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkFloatArray> pts_data = vtkSmartPointer<vtkFloatArray>::New();
        pts_data->SetNumberOfComponents(3);
        //            pts_data->SetNumberOfTuples(n_v);
        pts_data->SetVoidArray(meshbuffer->getVertices().get(), n_v * 3, 1);
        points->SetData(pts_data);


        vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
        scalars->SetNumberOfComponents(3);
        scalars->SetName("Colors");


        // TODO COLORS
        //            for(size_t i = 0; i < n_v; i++){
        //                size_t index = 3 * i;
        //                points->InsertNextPoint(
        //                        vertices[index    ],
        //                        vertices[index + 1],
        //                        vertices[index + 2]);
        //
        //                if(colors)
        //                {
        //                    size_t color_index = w_color * i;
        //                    unsigned char color[3];
        //                    color[0] = colors[color_index    ];
        //                    color[1] = colors[color_index + 1];
        //                    color[2] = colors[color_index + 2];
        //#if VTK_MAJOR_VERSION < 7
        //                    scalars->InsertNextTupleValue(color);
        //#else
        //                    scalars->InsertNextTypedTuple(color);
        //#endif
        //                }
        //            }

        vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();
        vtkSmartPointer<vtkIdTypeArray> tri_data = vtkSmartPointer<vtkIdTypeArray>::New();
        //            tri_data->SetNumberOfComponents(3);
        //            tri_data->SetNumberOfTuples(n_i);

        vtkIdType* tri_buf = new vtkIdType[n_i * 4];
        //vtkSmartPointer<vtkIdTypeArray> offsets;
        //offsets->SetNumberOfComponents(1);
        //offsets->SetNumberOfTuples(n_i);
        //vtkIdType* off_buf = new vtkIdType[n_i];
        //tri_buf[0] = static_cast<vtkIdType>(n_i);
        for(size_t i = 0; i < n_i; i++)
        {
            size_t index = 3 * i;
            size_t i2    = 4 * i;
            tri_buf[i2 + 0 ] = static_cast<vtkIdType>(3);
            //off_buf[i] = 3;
            tri_buf[i2 + 1 ] = static_cast<vtkIdType>(indices[index + 0]);
            tri_buf[i2 + 2 ] = static_cast<vtkIdType>(indices[index + 1]);
            tri_buf[i2 + 3 ] = static_cast<vtkIdType>(indices[index + 2]);
            //vtkSmartPointer<vtkTriangle> t = vtkSmartPointer<vtkTriangle>::New();
            //t->GetPointIds()->SetId(0, indices[index]);
            //t->GetPointIds()->SetId(1, indices[index + 1]);
            //t->GetPointIds()->SetId(2, indices[index + 2]);
            //triangles->InsertNextCell(t);
        }

        tri_data->SetVoidArray(tri_buf, n_i * 4, 0);
        //offsets->SetVoidArray(off_buf, n_i, 1);
        //triangles->SetData(offsets, tri_buf);
        triangles->SetCells(n_i, tri_data);
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
        meshActor = vtkSmartPointer<MeshChunkActor>::New();
        meshActor->SetMapper(mesh_mapper);
        meshActor->GetProperty()->BackfaceCullingOff();
        vtkSmartPointer<vtkPolyDataMapper> wireframe_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
#ifdef LVR2_USE_VTK5
        wireframe_mapper->SetInput(mesh);
#else
        wireframe_mapper->SetInputData(mesh);
#endif
        // TODO add wireframe stuff
        //m_wireframeActor = vtkSmartPointer<MeshChunkActor>::New();
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

        //setBaseColor(0.9, 0.9, 0.9);
        meshActor->setID(id);

        return meshActor;
    }
}

