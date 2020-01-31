#include <QObject>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>

#include "MeshChunkActor.hpp"

#include <unordered_map>


class AddHighResolution: public QObject
{
    public:
        AddHighResolution();

    public slots:
        void drawHighRes(std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> > low_res,
                    std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> > high_res);
    signals:
        void highResReady(std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> > low_res,
                    std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> > high_res);


};
