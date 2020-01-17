#include "MeshChunkActor.hpp"

vtkStandardNewMacro(MeshChunkActor);

void MeshChunkActor::ReleaseGraphicsResources(vtkWindow *window) {
    this->Device->ReleaseGraphicsResources(window);
    this->vtkActor::ReleaseGraphicsResources(window);
}

int MeshChunkActor::RenderOpaqueGeometry(vtkViewport *viewport){
    if ( ! this->Mapper ) {
        return 0;
    }
    if (!this->Property) {
        this->GetProperty();
    }
    if (this->GetIsOpaque()) {
        vtkRenderer *ren = static_cast<vtkRenderer *>(viewport);
        this->Render(ren);
        return 1;
    }
    return 0;
}

int MeshChunkActor::RenderTranslucentPolygonalGeometry(vtkViewport *viewport){
    if ( ! this->Mapper ) {
        return 0;
    }
    if (!this->Property) {
        this->GetProperty();
    }
    if (!this->GetIsOpaque()) {
        vtkRenderer *ren = static_cast<vtkRenderer *>(viewport);
        this->Render(ren);
        return 1;
    }
    return 0;
}

void MeshChunkActor::Render(vtkRenderer *ren){
    this->Property->Render(this, ren);
    this->Device->SetProperty (this->Property);
    this->Property->Render(this, ren);
    if (this->BackfaceProperty) {
        this->BackfaceProperty->BackfaceRender(this, ren);
        this->Device->SetBackfaceProperty(this->BackfaceProperty);
    }
    if (this->Texture) {
        this->Texture->Render(ren);
    }
    this->ComputeMatrix();
    this->Device->SetUserMatrix(this->Matrix);
    this->Device->Render(ren,this->Mapper);
}

void MeshChunkActor::ShallowCopy(vtkProp *prop) {
    MeshChunkActor *f = MeshChunkActor::SafeDownCast(prop);
    this->vtkActor::ShallowCopy(prop);
}
