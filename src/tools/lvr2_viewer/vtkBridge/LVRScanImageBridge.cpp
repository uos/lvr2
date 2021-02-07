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

/**
 * LVRModel.cpp
 *
 *  @date Dec 10, 2020
 *  @author Arthur Schreiber
 */
#include "LVRScanImageBridge.hpp"

#include "lvr2/geometry/Matrix4.hpp"

#include <vtkTransform.h>
#include <vtkActor.h>
#include <vtkProperty.h>

namespace lvr2
{

class LVRScanImageBridge;

LVRScanImageBridge::LVRScanImageBridge(ScanImagePtr img)
{
    image = img;
}
void LVRScanImageBridge::addActors(vtkSmartPointer<vtkRenderer> renderer)
{
    renderer->AddActor(imageActor);
}

void LVRScanImageBridge::removeActors(vtkSmartPointer<vtkRenderer> renderer)
{
    renderer->RemoveActor(imageActor);
    imageActor = nullptr;
    imageData = nullptr;
}

void LVRScanImageBridge::setImage(const cv::Mat& img)
{
    imageData = vtkSmartPointer<vtkImageData>::New();
    int numOfChannels = img.channels();
     // dimension set to 1 for z since it's 2D
    imageData->SetDimensions(img.cols, img.rows, 1);
    // NOTE: if your image isn't uchar for some reason you'll need to change this type
    imageData->AllocateScalars(VTK_UNSIGNED_CHAR, numOfChannels);
    // the flipped image data gets put into tempCVImage
    cv::Mat tempCVImage;
    cv::flip(img, tempCVImage, 0);

    // the number of byes in the cv::Mat, assuming the data type is uchar
    size_t byte_count = img.cols * img.rows * numOfChannels * sizeof(unsigned char);

    // copy the internal cv::Mat data into the vtkImageData pointer
    memcpy(imageData->GetScalarPointer(), tempCVImage.data, byte_count);

    imageData->Modified();

    imageActor = vtkSmartPointer<vtkActor2D>::New();

    vtkSmartPointer<vtkImageMapper> mapper = vtkSmartPointer<vtkImageMapper>::New();
    mapper->SetInputData(imageData);
    mapper->SetColorWindow(255);
    mapper->SetColorLevel(128);
    imageActor->SetMapper(mapper);
                
    std::cout << "Loading Image succeeded!" << std::endl;
}

void LVRScanImageBridge::setVisibility(bool visible){
    //TODO IMPLEMENT ME
    
    std::cout << "setting visibility to " << visible << image->imageFile.string() <<std::endl;
}

void LVRScanImageBridge::addPosActor(vtkSmartPointer<vtkRenderer> renderer ,vtkSmartPointer<vtkActor> actor, std::vector<LVRVtkArrow*> arrows)
{
    if (posActor != nullptr)
    {
        return;
    }
    renderer->AddActor(actor);
    posActor = actor;
    m_arrows = arrows;
    for (auto arrow : arrows)
    {
        renderer->AddActor(arrow->getArrowActor());
        renderer->AddActor(arrow->getStartActor());
        renderer->AddActor(arrow->getEndActor());
    }
}

void LVRScanImageBridge::removePosActor(vtkSmartPointer<vtkRenderer> renderer)
{
    renderer->RemoveActor(posActor);
    posActor = nullptr;
    for (auto arrow : m_arrows)
    {
        renderer->RemoveActor(arrow->getArrowActor());
        renderer->RemoveActor(arrow->getStartActor());
        renderer->RemoveActor(arrow->getEndActor());
        delete arrow;
    }
    m_arrows.clear();

}
LVRScanImageBridge::~LVRScanImageBridge()
{
    
}



} /* namespace lvr2 */
