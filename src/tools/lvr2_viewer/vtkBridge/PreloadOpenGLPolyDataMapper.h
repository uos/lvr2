#include <vtkNew.h>

#include <vtkOpenGLPolyDataMapper.h>
#include <vtkRenderingOpenGL2Module.h>
#include <vtkObjectFactory.h>
#include <vtkRenderer.h>
#include <vtkOpenGLRenderWindow.h>
#include "vtkOpenGLResourceFreeCallback.h"


class VTKRENDERINGOPENGL2_EXPORT PreloadOpenGLPolyDataMapper : public vtkOpenGLPolyDataMapper
{
    public:
        static PreloadOpenGLPolyDataMapper* New();
        vtkTypeMacro(PreloadOpenGLPolyDataMapper, vtkOpenGLPolyDataMapper)

        void CopyToMem(vtkRenderer* ren, vtkActor* act) { 
            this->ResourceCallback->RegisterGraphicsResources(
    static_cast<vtkOpenGLRenderWindow *>(ren->GetRenderWindow()));

            this->CurrentInput = this->GetInput();
            this->BuildBufferObjects(ren, act);
        }

        virtual bool GetNeedToRebuildBufferObjects(vtkRenderer *ren, vtkActor *act)
        {
            return false;
        }

        void BuildBufferObjects(vtkRenderer *ren, vtkActor *act) override ;


    protected:
        PreloadOpenGLPolyDataMapper() ;
        virtual ~PreloadOpenGLPolyDataMapper() ;

};
       
