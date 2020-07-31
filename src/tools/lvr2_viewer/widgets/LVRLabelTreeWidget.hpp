#ifndef LVRLABELTREEWIDGET_HPP
#define LVRLABELTREEWIDGET_HPP
#include <QTreeWidget>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/types/MatrixTypes.hpp"
class LVRLabelTreeWidget : public QTreeWidget
{
public:
    LVRLabelTreeWidget(QWidget *parent = nullptr);
    //using QTreeWidget::QTreeWidget;
    void addTopLevelItem(QTreeWidgetItem *item);
    void setLabelRoot(lvr2::LabelRootPtr labelRoot);

    lvr2::LabelRootPtr getLabelRoot();
private:
    lvr2::LabelRootPtr m_root;
};

#endif //LVRLABELTREEWIDGET_HPP
