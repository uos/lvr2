#ifndef LVR2_MESH_CHUNK_ACTOR_HPP_
#define LVR2_MESH_CHUNK_ACTOR_HPP_

#include <vtkActor.h>

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkObjectFactory.h>
#include <vtkRenderingCoreModule.h>
#include <vtkProperty.h>
#include <vtkTexture.h>

class VTKRENDERINGCORE_EXPORT MeshChunkActor : public vtkActor
{
    public:
        vtkTypeMacro(MeshChunkActor, vtkActor);

        static MeshChunkActor* New();

        size_t getID() { return id_; }
        void setID(size_t id) { id_ = id; }
    protected:
        size_t id_;

    public:
        virtual void ReleaseGraphicsResources(vtkWindow *window) override;

        virtual int RenderOpaqueGeometry(vtkViewport *viewport) override;

        virtual int RenderTranslucentPolygonalGeometry(vtkViewport *viewport) override;

        virtual void Render(vtkRenderer *ren);

        void ShallowCopy(vtkProp *prop) override; 

    protected:
        vtkActor* Device;

        MeshChunkActor() {
            this->Device = vtkActor::New();
        }

        ~MeshChunkActor() {
            this->Device -> Delete();
        }
};

#endif
