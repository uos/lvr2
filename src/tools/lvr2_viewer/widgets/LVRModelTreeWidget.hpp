#ifndef LVRMODELTREEWIDGET_HPP
#define LVRMODELTREEWIDGET_HPP
#include <QTreeWidget>
#include "LVRModelItem.hpp"
#include "LVRLabeledScanProjectEditMarkItem.hpp"
#include "LVRScanProjectItem.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/types/MatrixTypes.hpp"
class LVRModelTreeWidget : public QTreeWidget
{
public:
    LVRModelTreeWidget(QWidget *parent = nullptr);
    //using QTreeWidget::QTreeWidget;
    void addTopLevelItem(QTreeWidgetItem *item);

    void addScanProject(lvr2::ScanProjectPtr scanProject, std::string ="");
    void addLabeledScanProjectEditMark(lvr2::LabeledScanProjectEditMarkPtr labeledScanProject, std::string ="");

    //std::vector<> get**ModelItems();
};

#endif //LVRMODELTREEWIDGET_HPP
