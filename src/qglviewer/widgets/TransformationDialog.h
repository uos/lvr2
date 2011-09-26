/**
 * TransformationDialog.h
 *
 *  @date 01.08.2011
 *  @author Thomas Wiemann
 */

#ifndef TRANSFORMATIONDIALOG_H_
#define TRANSFORMATIONDIALOG_H_

#include "TransformationDialogUI.h"

#include "display/Renderable.hpp"

using lssr::Renderable;
using lssr::Vertex;
using lssr::Matrix4;

using Ui::TransformationDialogUI;

class TransformationDialog : public QObject
{
    Q_OBJECT

public:
    TransformationDialog(QWidget* parent, Renderable* r);
    virtual ~TransformationDialog();

public Q_SLOTS:

    void rotationXSlided(int value);
    void rotationYSlided(int value);
    void rotationZSlided(int value);

    void rotationXEntered(double value);
    void rotationYEntered(double value);
    void rotationZEntered(double value);

    void translationXEntered(double value);
    void translationYEntered(double value);
    void translationZEntered(double value);

    void reset();
private:

    void connectSignalsAndSlots();
    void transformLocal();
    void transformGlobal();

    double                      m_rotX;
    double                      m_rotY;
    double                      m_rotZ;

    double                      m_posX;
    double                      m_posY;
    double                      m_posZ;

    Renderable*                 m_renderable;
    TransformationDialogUI*     m_dialog;
    QWidget*                    m_parent;

};

#endif /* TRANSFORMATIONDIALOG_H_ */
