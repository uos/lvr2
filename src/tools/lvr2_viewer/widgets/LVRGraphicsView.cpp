#include "LVRGraphicsView.hpp"

#include <QMouseEvent>
#include <QApplication>
#include <QScrollBar>
#include <qmath.h>

namespace lvr2 {

void LVRGraphicsView::init()
{
    viewport()->installEventFilter(this);
    setMouseTracking(true);
    m_modifiers = Qt::ControlModifier;
    m_zoom_factor_base = 1.0015;
}

void LVRGraphicsView::gentle_zoom(double factor)
{
    scale(factor, factor);
    centerOn(m_target_scene_pos);
    QPointF delta_viewport_pos = m_target_viewport_pos - QPointF(viewport()->width() / 2.0, viewport()->height() / 2.0);
    QPointF viewport_center = mapFromScene(m_target_scene_pos) - delta_viewport_pos;
    centerOn(mapToScene(viewport_center.toPoint()));
    Q_EMIT(zoomed());
}

void LVRGraphicsView::set_modifiers(Qt::KeyboardModifiers modifiers)
{
    m_modifiers = modifiers;
}

void LVRGraphicsView::set_zoom_factor_base(double value)
{
    m_zoom_factor_base = value;
}

bool LVRGraphicsView::eventFilter(QObject* object, QEvent* event)
{
    
    if (event->type() == QEvent::MouseMove) {
        QMouseEvent* mouse_event = static_cast<QMouseEvent*>(event);
        QPointF delta = m_target_viewport_pos - mouse_event->pos();
        if (qAbs(delta.x()) > 5 || qAbs(delta.y()) > 5)
        {
            m_target_viewport_pos = mouse_event->pos();
            m_target_scene_pos = mapToScene(mouse_event->pos());
        }
    } else if (event->type() == QEvent::Wheel) {
        QWheelEvent* wheel_event = static_cast<QWheelEvent*>(event);
        if (QApplication::keyboardModifiers() == m_modifiers) {
            if (wheel_event->orientation() == Qt::Vertical) {
                double angle = wheel_event->angleDelta().y();
                double factor = qPow(m_zoom_factor_base, angle);
                gentle_zoom(factor);
                return true;
            }
        }
    }

    Q_UNUSED(object)
    return false;
}

void LVRGraphicsView::closeEvent(QCloseEvent *event)
{
    Q_EMIT(closed());
}

} // namespace lvr2