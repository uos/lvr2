#include "PreloadOpenGLPolyDataMapper.h"

#include <vtkOpenGLPolyDataMapper.h>

#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCommand.h>
#include <vtkFloatArray.h>
#include <vtkHardwareSelector.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkLightingMapPass.h>
#include <vtkMath.h>
#include <vtkMatrix3x3.h>
#include <vtkMatrix4x4.h>
#include <vtkObjectFactory.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLBufferObject.h>
#include <vtkOpenGLCamera.h>
#include <vtkOpenGLError.h>
#include <vtkOpenGLHelper.h>
#include <vtkOpenGLIndexBufferObject.h>
#include <vtkOpenGLRenderPass.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOpenGLRenderer.h>
#include <vtkOpenGLRenderTimer.h>
#include <vtkOpenGLShaderCache.h>
#include <vtkOpenGLState.h>
#include <vtkOpenGLTexture.h>
#include <vtkOpenGLVertexArrayObject.h>
#include <vtkOpenGLVertexBufferObject.h>
#include <vtkOpenGLVertexBufferObjectCache.h>
#include <vtkOpenGLVertexBufferObjectGroup.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkScalarsToColors.h>
#include <vtkShaderProgram.h>
#include <vtkTextureObject.h>
#include <vtkTransform.h>
#include <vtkUnsignedIntArray.h>

// Bring <n our fragment lit shader symbols.
//#include <vtkPolyDataFS.h>
//#include <vtkPolyDataWideLineGS.h>

#include <algorithm>
#include <sstream>


vtkStandardNewMacro(PreloadOpenGLPolyDataMapper)


PreloadOpenGLPolyDataMapper::PreloadOpenGLPolyDataMapper() : vtkOpenGLPolyDataMapper()
{}
PreloadOpenGLPolyDataMapper::~PreloadOpenGLPolyDataMapper() 
{}


//void PreloadOpenGLPolyDataMapper::BuildBufferObjects(vtkRenderer *ren, vtkActor *act)
//{
//  vtkPolyData *poly = this->CurrentInput;
//
//  if (poly == nullptr)
//  {
//      std::cout << "NO CURRENT INPUUT" << std::endl;
//    return;
//  }
//
//  // For vertex coloring, this sets this->Colors as side effect.
//  // For texture map coloring, this sets ColorCoordinates
//  // and ColorTextureMap as a side effect.
//  // I moved this out of the conditional because it is fast.
//  // Color arrays are cached. If nothing has changed,
//  // then the scalars do not have to be regenerted.
//  this->MapScalars(1.0);
//
//  // If we are coloring by texture, then load the texture map.
//  if (this->ColorTextureMap)
//  {
//    if (this->InternalColorTexture == nullptr)
//    {
//      this->InternalColorTexture = vtkOpenGLTexture::New();
//      this->InternalColorTexture->RepeatOff();
//    }
//    this->InternalColorTexture->SetInputData(this->ColorTextureMap);
//  }
//
//  this->HaveCellScalars = false;
//  vtkDataArray *c = this->Colors;
//  if (this->ScalarVisibility)
//  {
//    // We must figure out how the scalars should be mapped to the polydata.
//    if ( (this->ScalarMode == VTK_SCALAR_MODE_USE_CELL_DATA ||
//          this->ScalarMode == VTK_SCALAR_MODE_USE_CELL_FIELD_DATA ||
//          this->ScalarMode == VTK_SCALAR_MODE_USE_FIELD_DATA ||
//          !poly->GetPointData()->GetScalars() )
//         && this->ScalarMode != VTK_SCALAR_MODE_USE_POINT_FIELD_DATA
//         && this->Colors)
//    {
//      this->HaveCellScalars = true;
//      c = nullptr;
//    }
//  }
//
//  this->HaveCellNormals = false;
//  // Do we have cell normals?
//  vtkDataArray *n =
//    (act->GetProperty()->GetInterpolation() != VTK_FLAT) ? poly->GetPointData()->GetNormals() : nullptr;
//  if (n == nullptr && poly->GetCellData()->GetNormals())
//  {
//    this->HaveCellNormals = true;
//  }
//
//  int representation = act->GetProperty()->GetRepresentation();
//
//  // check if this system is subject to the apple/amd primID bug
//  this->HaveAppleBug =
//    static_cast<vtkOpenGLRenderer *>(ren)->HaveApplePrimitiveIdBug();
//  if (this->HaveAppleBugForce == 1)
//  {
//    this->HaveAppleBug = false;
//  }
//  if (this->HaveAppleBugForce == 2)
//  {
//    this->HaveAppleBug = true;
//  }
//
//  vtkCellArray *prims[4];
//  prims[0] =  poly->GetVerts();
//  prims[1] =  poly->GetLines();
//  prims[2] =  poly->GetPolys();
//  prims[3] =  poly->GetStrips();
//
//  // only rebuild what we need to
//  // if the data or mapper or selection state changed
//  // then rebuild the cell arrays
//  this->TempState.Clear();
//  this->TempState.Append(
//    prims[0]->GetNumberOfCells() ? prims[0]->GetMTime() : 0, "prim0 mtime");
//  this->TempState.Append(
//    prims[1]->GetNumberOfCells() ? prims[1]->GetMTime() : 0, "prim1 mtime");
//  this->TempState.Append(
//    prims[2]->GetNumberOfCells() ? prims[2]->GetMTime() : 0, "prim2 mtime");
//  this->TempState.Append(
//    prims[3]->GetNumberOfCells() ? prims[3]->GetMTime() : 0, "prim3 mtime");
//  this->TempState.Append(representation, "representation");
//  this->TempState.Append(this->LastSelectionState, "last selection state");
//  this->TempState.Append(poly->GetMTime(), "polydata mtime");
//  this->TempState.Append(this->GetMTime(), "this mtime");
//  if (this->CellTextureBuildState != this->TempState)
//  {
//    this->CellTextureBuildState = this->TempState;
//    this->BuildCellTextures(ren, act, prims, representation);
//  }
//
//  // on Apple Macs with the AMD PrimID bug <rdar://20747550>
//  // we use a slow painful approach to work around it (pre 10.11).
//  this->AppleBugPrimIDs.resize(0);
//  if (this->HaveAppleBug &&
//      (this->HaveCellNormals || this->HaveCellScalars))
//  {
//    if (!this->AppleBugPrimIDBuffer)
//    {
//      this->AppleBugPrimIDBuffer = vtkOpenGLBufferObject::New();
//    }
//    poly = this->HandleAppleBug(poly, this->AppleBugPrimIDs);
//    this->AppleBugPrimIDBuffer->Bind();
//    this->AppleBugPrimIDBuffer->Upload(
//     this->AppleBugPrimIDs, vtkOpenGLBufferObject::ArrayBuffer);
//    this->AppleBugPrimIDBuffer->Release();
//
//#ifndef NDEBUG
//    static bool warnedAboutBrokenAppleDriver = false;
//    if (!warnedAboutBrokenAppleDriver)
//    {
//      vtkWarningMacro("VTK is working around a bug in Apple-AMD hardware related to gl_PrimitiveID.  This may cause significant memory and performance impacts. Your hardware has been identified as vendor "
//        << (const char *)glGetString(GL_VENDOR) << " with renderer of "
//        << (const char *)glGetString(GL_RENDERER) << " and version "
//        << (const char *)glGetString(GL_VERSION));
//      warnedAboutBrokenAppleDriver = true;
//    }
//#endif
//    if (n)
//    {
//      n = (act->GetProperty()->GetInterpolation() != VTK_FLAT) ?
//            poly->GetPointData()->GetNormals() : nullptr;
//    }
//    if (c)
//    {
//      this->Colors->Delete();
//      this->Colors = nullptr;
//      this->MapScalars(poly,1.0);
//      c = this->Colors;
//    }
//  }
//
//  // Set the texture if we are going to use texture
//  // for coloring with a point attribute.
//  vtkDataArray *tcoords = nullptr;
//  if (this->HaveTCoords(poly))
//  {
//    if (this->InterpolateScalarsBeforeMapping && this->ColorCoordinates)
//    {
//      tcoords = this->ColorCoordinates;
//    }
//    else
//    {
//      tcoords = poly->GetPointData()->GetTCoords();
//    }
//  }
//
//  vtkOpenGLRenderWindow *renWin = vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow());
//  vtkOpenGLVertexBufferObjectCache *cache = renWin->GetVBOCache();
//
//  // rebuild VBO if needed
//  for (auto &itr : this->ExtraAttributes)
//  {
//    vtkDataArray *da = poly->GetPointData()->GetArray(itr.second.DataArrayName.c_str());
//    this->VBOs->CacheDataArray(itr.first.c_str(), da, cache, VTK_FLOAT);
//  }
//
//  this->VBOs->CacheDataArray("vertexMC", poly->GetPoints()->GetData(), cache, VTK_FLOAT);
//  vtkOpenGLVertexBufferObject *posVBO = this->VBOs->GetVBO("vertexMC");
//  if (posVBO)
//  {
//    posVBO->SetCoordShiftAndScaleMethod(
//      static_cast<vtkOpenGLVertexBufferObject::ShiftScaleMethod>(this->ShiftScaleMethod));
//  }
//
//  this->VBOs->CacheDataArray("normalMC", n, cache, VTK_FLOAT);
//  this->VBOs->CacheDataArray("scalarColor", c, cache, VTK_UNSIGNED_CHAR);
//  this->VBOs->CacheDataArray("tcoord", tcoords, cache, VTK_FLOAT);
//  this->VBOs->BuildAllVBOs(cache);
//
//  // get it again as it may have been freed
//  posVBO = this->VBOs->GetVBO("vertexMC");
//  if (posVBO && posVBO->GetCoordShiftAndScaleEnabled())
//  {
//    std::vector<double> shift = posVBO->GetShift();
//    std::vector<double> scale = posVBO->GetScale();
//    this->VBOInverseTransform->Identity();
//    this->VBOInverseTransform->Translate(shift[0], shift[1], shift[2]);
//    this->VBOInverseTransform->Scale(1.0/scale[0], 1.0/scale[1], 1.0/scale[2]);
//    this->VBOInverseTransform->GetTranspose(this->VBOShiftScale);
//  }
//
//  // now create the IBOs
//  this->BuildIBO(ren, act, poly);
//
//  // free up polydata if allocated due to apple bug
//  if (poly != this->CurrentInput)
//  {
//    poly->Delete();
//  }
//
//  vtkOpenGLCheckErrorMacro("failed after BuildBufferObjects");
//
//  this->VBOBuildTime.Modified(); // need to call all the time or GetNeedToRebuild will always return true;
//}
