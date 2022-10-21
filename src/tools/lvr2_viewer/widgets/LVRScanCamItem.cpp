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
 * LVRScanCamItem.cpp
 *
 *  @date Dec 10, 2020
 *  @author Arthur Schreiber
 */
#include "LVRScanCamItem.hpp"
#include "LVRPointCloudItem.hpp"
#include "LVRMeshItem.hpp"
#include "LVRItemTypes.hpp"
#include "LVRTextureMeshItem.hpp"
#include "LVRScanImageItem.hpp"
#include "LVRCameraModelItem.hpp"


#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <QTextStream>

namespace lvr2
{

LVRScanCamItem::LVRScanCamItem(ScanCamBridgePtr bridge, QString name) :
    QTreeWidgetItem(LVRScanCamItemType), m_scanCamBridge(bridge), m_name(name)
{
    // Setup tree widget icon
    QIcon icon;
    icon.addFile(QString::fromUtf8(":/qv_model_tree_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
    setIcon(0, icon);

    // Setup item properties
    setText(0, m_name);

    int camNo = name.right(name.size() - name.lastIndexOf('_') - 1).toInt();
    setData(0, Qt::UserRole, camNo);

    setCheckState(0, Qt::Checked);

    // Create a ScanImageItem for every image in the camera
    // To corrextly resemble the structure of the project 
    // the images should be grouped in the future
    size_t c = 0;
    for(auto g : m_scanCamBridge->m_cam->groups)
    {
        for (auto img : g->images)
        {
            ScanImageBridgePtr imgBridge(new LVRScanImageBridge(img));

            /// Fixme: How to lazy-load?==
            QString imgName = "Image_" + c;
            LVRScanImageItem *imgItem = new LVRScanImageItem(imgBridge, imgName);
            addChild(imgItem);
            c++;
        }
    }

    //add model for camera as child
    LVRCameraModelItem* model = new LVRCameraModelItem(*(m_scanCamBridge->m_cam));
    addChild(model);
}

LVRScanCamItem::LVRScanCamItem(const LVRScanCamItem& item)
{
    m_scanCamBridge   = item.m_scanCamBridge;
    m_name          = item.m_name;
}


QString LVRScanCamItem::getName()
{
    return m_name;
}

void LVRScanCamItem::setName(QString name)
{
    m_name = name;
    setText(0, m_name);
}


bool LVRScanCamItem::isEnabled()
{
    return this->checkState(0);
}


LVRScanCamItem::~LVRScanCamItem()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr2 */
