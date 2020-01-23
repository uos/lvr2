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
    m_chunkManager.extractArea(bb, m_chunks, "mesh0");
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
    for(int i = 0; i < indices.size(); ++i)
    {

        if(m_chunkActors.find(indices[i]) == m_chunkActors.end())
        {
            std::cout << "KEY NOT FOUND: " << i << std::endl;
        }
    }


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
    for(const auto& chunk: m_chunks)
    {
        size_t id = chunk.first;
        lvr2::MeshBufferPtr meshbuffer = chunk.second;
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

            vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
            vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();

            vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
            scalars->SetNumberOfComponents(3);
            scalars->SetName("Colors");

            for(size_t i = 0; i < n_v; i++){
                size_t index = 3 * i;
                points->InsertNextPoint(
                        vertices[index    ],
                        vertices[index + 1],
                        vertices[index + 2]);

                if(colors)
                {
                    size_t color_index = w_color * i;
                    unsigned char color[3];
                    color[0] = colors[color_index    ];
                    color[1] = colors[color_index + 1];
                    color[2] = colors[color_index + 2];
#if VTK_MAJOR_VERSION < 7
                    scalars->InsertNextTupleValue(color);
#else
                    scalars->InsertNextTypedTuple(color);
#endif
                }
            }

            for(size_t i = 0; i < n_i; i++)
            {
                size_t index = 3 * i;
                vtkSmartPointer<vtkTriangle> t = vtkSmartPointer<vtkTriangle>::New();
                t->GetPointIds()->SetId(0, indices[index]);
                t->GetPointIds()->SetId(1, indices[index + 1]);
                t->GetPointIds()->SetId(2, indices[index + 2]);
                triangles->InsertNextCell(t);

            }

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
            m_chunkActors.insert({id, meshActor});
        }
    }
    // clear the chunks array
    m_chunks.clear();
    std::unordered_map<size_t, MeshBufferPtr>().swap(m_chunks);

}
