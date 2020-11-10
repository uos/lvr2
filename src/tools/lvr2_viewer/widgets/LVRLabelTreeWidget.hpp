#ifndef LVRLABELTREEWIDGET_HPP
#define LVRLABELTREEWIDGET_HPP
#include <QTreeWidget>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include <QComboBox>
#include "../vtkBridge/LVRPickingInteractor.hpp"
class LVRLabelTreeWidget : public QTreeWidget
{
public:
    LVRLabelTreeWidget(QWidget *parent = nullptr);
    //using QTreeWidget::QTreeWidget;
    void addTopLevelItem(QTreeWidgetItem *item);
    void itemSelected(int);
    void setLabelRoot(lvr2::LabelRootPtr labelRoot, lvr2::LVRPickingInteractor*, QComboBox*);
    int getNextId();

    lvr2::LabelRootPtr getLabelRoot();
private:
    int m_id = 1;
    lvr2::LabelRootPtr m_root;
    lvr2::LVRLabelInstanceTreeItem * m_selectedItem = nullptr;
};

#endif //LVRLABELTREEWIDGET_HPP
