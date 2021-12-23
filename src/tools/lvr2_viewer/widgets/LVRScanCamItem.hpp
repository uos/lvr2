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
 * LVRScanCamItem.hpp
 *
 *  @date Dec 10, 2020
 *  @author Arthur Schreiber
 */
#ifndef LVRSCANCAMITEM_H_
#define LVRSCANCAMITEM_H_

#include "../vtkBridge/LVRScanCamBridge.hpp"
#include "../vtkBridge/LVRScanImageBridge.hpp"
#include "LVRPoseItem.hpp"

#include <QString>
#include <QColor>
#include <QTreeWidgetItem>

namespace lvr2
{

class LVRScanCamItem : public QTreeWidgetItem
{
public:
    /**
     *  @brief          Constructs an item for a ScanCamera
     *  @param bridge   bridge for the new item
     *  @param name     name for the new item
     */
    LVRScanCamItem(ScanCamBridgePtr bridge, QString name = "");

    /**
     *  @brief          Copy constructor for ScanCamItem
     */
    LVRScanCamItem(const LVRScanCamItem& item);

    /**
     *  @brief          Destructor.
     */
    virtual ~LVRScanCamItem();

    /**
     *  @brief          Getter for the item name
     */
    QString         getName();

    /**
     *  @brief          Setter for the item name
     *  @param name     name to set
     */
    void            setName(QString name);

    /**
     *  @brief          returns whether the item is enabled (checked)
     */
    bool            isEnabled();


protected:
    ScanCamBridgePtr  m_scanCamBridge;
    QString         m_name;
};

} /* namespace lvr2 */

#endif /* LVRSCANCAMITEM_H_ */