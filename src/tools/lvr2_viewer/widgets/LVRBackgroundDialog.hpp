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
 * LVRBackgroundDialog.hpp
 *
 *  @date Sep 18, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRBACKGROUNDDIALOG_HPP_
#define LVRBACKGROUNDDIALOG_HPP_

#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>


#include <QColorDialog>

#include "ui_LVRBackgroundDialogUI.h"

using Ui::BackgroundDialogUI;

namespace lvr2
{

class LVRBackgroundDialog : public QDialog
{
    Q_OBJECT
public:
    LVRBackgroundDialog(vtkSmartPointer<vtkRenderWindow> renderWindow);
    virtual ~LVRBackgroundDialog();

    void getColor1(float &r, float &g, float &b);
    void getColor2(float &r, float &g, float &b);
    bool renderGradient();

public Q_SLOTS:
    void color1Changed();
    void color2Changed();

private:

    void updateColorBox(QFrame* box, QColor color);

    vtkSmartPointer<vtkRenderWindow>    m_renderWindow;
    BackgroundDialogUI*                 m_ui;
    QColor                              m_color1;
    QColor                              m_color2;
};

} /* namespace lvr2 */

#endif /* LVRBACKGROUNDDIALOG_HPP_ */
