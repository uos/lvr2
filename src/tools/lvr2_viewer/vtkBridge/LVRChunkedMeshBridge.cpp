#include "LVRChunkedMeshBridge.hpp"

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
//#include <PreloadOpenGLPolyDataMapper.h>
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
#include <vtkXOpenGLRenderWindow.h>
#include "PreloadOpenGLPolyDataMapper.h"
#include <omp.h>

#include "GL/glx.h"

using namespace lvr2;

GLXContext m_workerContext;
XVisualInfo* vinfo;
Window x_window;
Display* display;

LVRChunkedMeshBridge::LVRChunkedMeshBridge(std::string file, vtkSmartPointer<vtkRenderer> renderer, size_t cache_size, double highResDistance) : m_chunkManager(file, cache_size), m_renderer(renderer)
{
    getNew_ = false;
    dist_ = 40.0;
    running_ = true;

    // get context opengl
    // createSharedcontext  from current.
    //
    //
   vtkRenderWindow* window = m_renderer->GetRenderWindow();
    if(window->IsA("vtkXOpenGLRenderWindow"))
    {
        std::cout << "VTKXOPENGL" << std::endl;
        vtkXOpenGLRenderWindow* ogl_window = static_cast<vtkXOpenGLRenderWindow*>(window);
        vinfo = ogl_window->GetDesiredVisualInfo();
        if(vinfo != nullptr)
        {
            std::cout << "got visualinfo" << std::endl;
        }

        display = ogl_window->GetDisplayId();
        if(display != nullptr)
        {
            std::cout << "got display" << std::endl;
        }   
        x_window = ogl_window->GetParentId();
        //    x_window = ogl_window->GetParentId();
    }
    GLXContext currentContext = glXGetCurrentContext();
    if(currentContext!= nullptr)
    {
        std::cout << "Got the context" << std::endl;
    }

    m_workerContext = glXCreateContext(display, vinfo, currentContext, true);

    worker = std::thread(&LVRChunkedMeshBridge::highResWorker, this);

}


void LVRChunkedMeshBridge::highResWorker()
{
    bool success = glXMakeCurrent(display, x_window, m_workerContext);
    if(success)
    {
        std::cout << "MAKE CURRENT SUCCESS" << std::endl;
    }

    while(running_)
    {
       std::unique_lock<std::mutex> l(mutex);
       while(!getNew_) 
       {
           cond_.wait(l);
       }
       getNew_ = false;
       l.unlock();
       std::vector<size_t> visible_indices = highResIndices;

       BaseVector<float> diff = m_region.getCentroid() - m_lastRegion.getCentroid();
       if(!(std::abs(diff[0]) > 1.0 || std::abs(diff[1]) > 1.0 || std::abs(diff[2]) > 1.0))
       {
//           l.unlock();
           continue;
       }

       m_lastRegion = m_region;


       auto old_highRes = m_highRes;
       if(m_highRes.size() > 1000)
       {
           m_highRes.clear();
       }

       m_chunkManager.extractArea(m_region, m_highRes, "mesh0");


       for(auto& it : m_highRes)
       {
            if(old_highRes.find(it.first) != old_highRes.end())
            {
                m_highRes.at(it.first) = old_highRes.at(it.first);
            }
       }
       old_highRes.clear();
    

        std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor>> tmp_highResActors;

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

       actorMap remove_actors;
       actorMap new_actors;
       for(auto& it: m_highResActors)
       {
           if(tmp_highResActors.find(it.first) == tmp_highResActors.end())
           {
            remove_actors.insert({it.first, it.second});
           }
       }

       for(auto& it: tmp_highResActors)
       {
           if(m_highResActors.find(it.first) == m_highResActors.end())
           {
                new_actors.insert({it.first, it.second});
           }
       }


       m_highResActors = tmp_highResActors; 
    
       Q_EMIT updateHighRes(remove_actors, new_actors);
    
//       l.unlock();
    }

}

void LVRChunkedMeshBridge::fetchHighRes(BoundingBox<BaseVector<float> > bb,
                                        std::vector<size_t> indices)
{
    // FUCK SYNCHRONIZATION
//    std::unique_lock<std::mutex> l(mutex);
    m_region = bb;
    highResIndices = indices;
    getNew_ = true;
//    l.unlock();
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
    Vector3d cam_origin(cam_position[0], cam_position[1], cam_position[2]);
    Vector3d view_up(1.0, 0.0, 0.0);
    Vector3d focal_point(centroid[0], centroid[1], centroid[2]);

    renderer->GetActiveCamera()->SetPosition(cam_origin.x(), cam_origin.y(), cam_origin.z());
    renderer->GetActiveCamera()->SetFocalPoint(focal_point.x(), focal_point.y(), focal_point.z());
    renderer->GetActiveCamera()->SetViewUp(view_up.x(), view_up.y(), view_up.z());


    m_chunkManager.extractArea(bb, m_chunks, "mesh1");
    

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

        m_chunkActors.insert({id, computeMeshActor(id, meshbuffer)});

    }


    std::cout << lvr2::timestamp << "Done actor computation" << std::endl;
}

vtkSmartPointer<MeshChunkActor> LVRChunkedMeshBridge::computeMeshActor(size_t& id, MeshBufferPtr& meshbuffer)
{
    vtkSmartPointer<MeshChunkActor> meshActor;

    if(meshbuffer)
    {
        vtkSmartPointer<vtkPolyData> mesh = vtkSmartPointer<vtkPolyData>::New();
        meshActor = vtkSmartPointer<MeshChunkActor>::New();
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
        if(colors)
        {
            scalars->SetVoidArray(colors.get(), n_v * w_color, 1);
        }


        // Triangle indices
        vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();
        vtkSmartPointer<vtkIdTypeArray> tri_data = vtkSmartPointer<vtkIdTypeArray>::New();
        vtkIdType* tri_buf = new vtkIdType[n_i * 4];
        for(size_t i = 0; i < n_i; i++)
        {
            size_t index = 3 * i;
            size_t i2    = 4 * i;
            tri_buf[i2 + 0 ] = static_cast<vtkIdType>(3);
            tri_buf[i2 + 1 ] = static_cast<vtkIdType>(indices[index + 0]);
            tri_buf[i2 + 2 ] = static_cast<vtkIdType>(indices[index + 1]);
            tri_buf[i2 + 3 ] = static_cast<vtkIdType>(indices[index + 2]);
        }

        tri_data->SetVoidArray(tri_buf, n_i * 4, 0);
        triangles->SetCells(n_i, tri_data);

        // add points, triangles and if given colors
        mesh->SetPoints(points);
        mesh->SetPolys(triangles);
        if(colors)
        {
            mesh->GetPointData()->SetScalars(scalars);
        }

        vtkSmartPointer<PreloadOpenGLPolyDataMapper> mesh_mapper = vtkSmartPointer<PreloadOpenGLPolyDataMapper>::New();
        #ifdef LVR2_USE_VTK5
        mesh_mapper->SetInput(mesh);
        #else
        mesh_mapper->SetInputData(mesh);
        #endif
        meshActor->SetMapper(mesh_mapper);
        meshActor->GetProperty()->BackfaceCullingOff();
        vtkSmartPointer<PreloadOpenGLPolyDataMapper> wireframe_mapper = vtkSmartPointer<PreloadOpenGLPolyDataMapper>::New();
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
        mesh_mapper->CopyToMem(m_renderer.Get(), meshActor.Get());

    }

    return meshActor;
}

