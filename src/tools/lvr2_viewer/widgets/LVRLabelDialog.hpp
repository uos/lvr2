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
 * LVRCorrespondanceDialog.hpp
 *
 *  @date Feb 18, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRLABELDIALOG_HPP_
#define LVRLABELDIALOG_HPP_

#include "ui_LVRLabelDialogUI.h"
#include <QTreeWidget>

#include <lvr2/geometry/Matrix4.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/registration/EigenSVDPointAlign.hpp>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include <boost/shared_array.hpp>

#include <iostream>

#define LABEL_NAME_COLUMN 0
#define LABELED_POINT_COLUMN 1
#define LABEL_VISIBLE_COLUMN 2
#define LABEL_ID_COLUMN 3

using namespace std;

namespace lvr2
{


class LVRLabelDialog : public QObject
{

    Q_OBJECT
public:

    LVRLabelDialog(QTreeWidget* parent);
    virtual ~LVRLabelDialog();
    void showEvent();
    void setPoints(const std::string, const vtkSmartPointer<vtkPolyData> points);

public Q_SLOTS:
    void addNewLabel();
    void addNewInstance();
    void responseLabels(std::vector<uint16_t>);
    void updatePointCount(uint16_t, int);
    void cellSelected(QTreeWidgetItem*, int);
    void comboBoxIndexChanged(int index);
    void visibilityChanged(QTreeWidgetItem*, int);
    void loadLabels();

Q_SIGNALS:
    void labelRemoved(QPair<int, QColor>);
    void labelAdded(QTreeWidgetItem*);
    void labelChanged(uint16_t);
    void labelLoaded(int, std::vector<int>);
    void hidePoints(int, bool);

public:
    Ui_LabelDialog*    m_ui;
    QDialog*                    m_dialog;

private:
    QTreeWidget*                m_treeWidget;
    QColor                      m_dataSelectionColor;
    QColor                      m_modelSelectionColor;
    int                         m_id_hack = 1;
    std::map<std::string, vtkSmartPointer<vtkPolyData>> m_points;
};

} /* namespace lvr2 */

#endif /* LVRLABELDIALOG_HPP_ */
