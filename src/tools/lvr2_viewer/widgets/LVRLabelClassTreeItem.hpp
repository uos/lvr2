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

#ifndef LVRLABELClassTREEITEM_H_
#define LVRLABELClassTREEITEM_H_


#include <QString>
#include <QColor>
#include <QTreeWidgetItem>
#include "lvr2/types/ScanTypes.hpp"

#define LABEL_NAME_COLUMN 0
#define LABELED_POINT_COLUMN 1
#define LABEL_VISIBLE_COLUMN 2
#define LABEL_ID_COLUMN 3
#define LABEL_EDITABLE_COLUMN 4
#define LABEL_ID_GROUP 0
#define LABEL_COLOR_GROUP 1


namespace lvr2
{

class LVRLabelClassTreeItem : public QTreeWidgetItem
{
public:
    LVRLabelClassTreeItem(std::string labelClass, int labeledPointCount, bool visible, bool editable, QColor color);
    LVRLabelClassTreeItem(LabelClassPtr classptr);
    LVRLabelClassTreeItem(const LVRLabelClassTreeItem& item);
    virtual ~LVRLabelClassTreeItem();
    void setColor(QColor);
    QColor getDefaultColor();
    bool isVisible();
    bool isEditable();
    int getNumberOfLabeledPoints();
    void addChild(QTreeWidgetItem *child);
    void removeChild(QTreeWidgetItem *child);
    void addChildnoChanges(QTreeWidgetItem *child);
    std::string getName();
    LabelClassPtr getLabelClassPtr();
    QStringList getChildNames();
private:
    LabelClassPtr m_labelClassPtr;
};

} /* namespace lvr2 */

#endif /* LVRLABELCLASSTREEITEM_H_ */
