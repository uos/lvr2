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
 * LVRBackgroundDialog.cpp
 *
 *  @date Sep 18, 2014
 *  @author Thomas Wiemann
 */
#include "LVRBackgroundDialog.hpp"

namespace lvr
{

LVRBackgroundDialog::LVRBackgroundDialog(vtkSmartPointer<vtkRenderWindow> renderWindow)
    : m_renderWindow(renderWindow)
{
    m_ui = new BackgroundDialogUI;
    m_ui->setupUi(this);

    updateColorBox(m_ui->colorFrame1, QColor(0, 0, 255));
    updateColorBox(m_ui->colorFrame1, QColor(255, 255, 255));

    connect(m_ui->buttonChange1, SIGNAL(pressed()), this, SLOT(color1Changed()));
    connect(m_ui->buttonChange2, SIGNAL(pressed()), this, SLOT(color2Changed()));

}

void LVRBackgroundDialog::updateColorBox(QFrame* box, QColor color)
{
    QPalette pal(palette());
    pal.setColor(QPalette::Background, color);
    box->setAutoFillBackground(true);
    box->setPalette(pal);
}

void LVRBackgroundDialog::color1Changed()
{
    //m_colorDialog1.exec();
    QColorDialog d;
    m_color1 = d.getColor();
    updateColorBox(m_ui->colorFrame1, m_color1);
}

void LVRBackgroundDialog::color2Changed()
{
   // m_colorDialog2.exec();
    QColorDialog d;
    m_color2 = d.getColor();
    updateColorBox(m_ui->colorFrame2, m_color2);
}

void LVRBackgroundDialog::getColor1(float &r, float &g, float &b)
{
    r = m_color1.redF();
    g = m_color1.greenF();
    b = m_color1.blueF();
}

void LVRBackgroundDialog::getColor2(float &r, float &g, float &b)
{
    r = m_color2.redF();
    g = m_color2.greenF();
    b = m_color2.blueF();
}

bool LVRBackgroundDialog::renderGradient()
{
    return !(m_ui->checkBoxUniformRendering->isChecked());
}

LVRBackgroundDialog::~LVRBackgroundDialog()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr */
