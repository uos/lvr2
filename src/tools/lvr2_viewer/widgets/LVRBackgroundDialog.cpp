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
 * LVRBackgroundDialog.cpp
 *
 *  @date Sep 18, 2014
 *  @author Thomas Wiemann
 */
#include "LVRBackgroundDialog.hpp"

namespace lvr2
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

} /* namespace lvr2 */
