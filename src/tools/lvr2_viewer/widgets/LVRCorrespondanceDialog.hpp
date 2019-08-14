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
#ifndef LVRCORRESPONDANCEDIALOG_HPP_
#define LVRCORRESPONDANCEDIALOG_HPP_

#include "ui_LVRRegistrationPickCorrespondancesDialogUI.h"
#include "../vtkBridge/LVRVtkArrow.hpp"

#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/registration/EigenSVDPointAlign.hpp"

#include <iostream>
using namespace std;

#include <boost/optional.hpp>
#include <Eigen/Dense>

namespace lvr2
{

class LVRCorrespondanceDialog : public QObject
{
    Q_OBJECT
public:

    LVRCorrespondanceDialog(QTreeWidget* parent);
    virtual ~LVRCorrespondanceDialog();
    void fillComboBoxes();
    boost::optional<Transformf> getTransformation();
    QString  getModelName();
    QString  getDataName();

    bool    doICP();
    double  getEpsilon();
    double  getMaxDistance();
    int     getMaxIterations();

public Q_SLOTS:
    void updateModelSelection(int);
    void updateDataSelection(int);
    void firstPointPicked(double*);
    void secondPointPicked(double*);
    void insertNewItem();
    void deleteItem();
    void swapItemPositions();
    void saveCorrespondences();
    void loadCorrespondences();
    void treeItemSelected(QTreeWidgetItem*, QTreeWidgetItem*);
    void clearAllItems();


Q_SIGNALS:
    void render();
    void removeArrow(LVRVtkArrow *);
    void addArrow(LVRVtkArrow *);
    void disableCorrespondenceSearch();
    void enableCorrespondenceSearch();

public:
    Ui_CorrespondenceDialog*    m_ui;
    QDialog*                    m_dialog;

private:
    QTreeWidget*                m_treeWidget;
    QColor                      m_dataSelectionColor;
    QColor                      m_modelSelectionColor;
    QColor                      m_defaultColor;
    QString                     m_dataSelection;
    QString                     m_modelSelection;
};

} /* namespace lvr2 */

#endif /* LVRCORRESPONDANCEDIALOG_HPP_ */
