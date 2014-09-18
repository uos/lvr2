/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
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

#include "LVRBackgroundDialogUI.h"

using Ui::BackgroundDialogUI;

namespace lvr
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
    QColorDialog                        m_colorDialog1;
    QColorDialog                        m_colorDialog2;
};

} /* namespace lvr */

#endif /* LVRBACKGROUNDDIALOG_HPP_ */
