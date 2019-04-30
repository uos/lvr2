#ifndef LVR2_TOOLS_VIEWER_WIDGETS_LVRGraphicsView_HPP
#define LVR2_TOOLS_VIEWER_WIDGETS_LVRGraphicsView_HPP

#include <QObject>
#include <QGraphicsView>
#include <QtGui>
#include <QCloseEvent>
#include <iostream>

namespace lvr2 {

class LVRGraphicsView : public QGraphicsView {
    Q_OBJECT
public:
    using QGraphicsView::QGraphicsView;

    void init();

    void gentle_zoom(double factor);
    void set_modifiers(Qt::KeyboardModifiers modifiers);
    void set_zoom_factor_base(double value);

private:
    bool eventFilter(QObject* object, QEvent* event);
    void closeEvent(QCloseEvent *event);

    Qt::KeyboardModifiers           m_modifiers;
    double                          m_zoom_factor_base;
    QPointF                         m_target_scene_pos;
    QPointF                         m_target_viewport_pos;

Q_SIGNALS:
    void zoomed();
    void closed();
};

} // namespace lvr2

#endif