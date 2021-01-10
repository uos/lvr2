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
 * LVRModelItem.h
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRSCANPROJECTITEM_H_
#define LVRSCANPROJECTITEM_H_

#include "../vtkBridge/LVRScanProjectBridge.hpp"
#include "LVRPoseItem.hpp"

#include <QString>
#include <QColor>
#include <QTreeWidgetItem>

namespace lvr2
{

class LVRScanProjectItem : public QTreeWidgetItem
{
public:
    LVRScanProjectItem(ScanProjectBridgePtr bridge, QString name = "");
    LVRScanProjectItem(const LVRScanProjectItem& item);
    virtual ~LVRScanProjectItem();
    ScanProjectBridgePtr	getScanProjectBridge();
    QString         getName();
    void            setName(QString name);
    bool            isEnabled();
    void            setBridge(ScanProjectBridgePtr bridge);

public Q_SLOTS:
    void			setVisibility(bool visible);

protected:
    ScanProjectBridgePtr  m_scanProjectBridge;
    QString         m_name;
};

} /* namespace lvr2 */

#endif /* LVRSCANPROJECTITEM_H_ */
