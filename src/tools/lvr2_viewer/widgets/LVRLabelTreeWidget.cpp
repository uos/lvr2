#include "LVRLabelTreeWidget.hpp"
#include <iostream>
#include "LVRItemTypes.hpp"
#include "LVRLabelClassTreeItem.hpp"
#include "LVRLabelInstanceTreeItem.hpp"

LVRLabelTreeWidget::LVRLabelTreeWidget(QWidget *parent)
{
    m_root = lvr2::LabelRootPtr(new lvr2::LabelRoot()); 
}

void LVRLabelTreeWidget::addTopLevelItem(QTreeWidgetItem *item)
{
    if(item->type() == lvr2::LVRLabelClassItemType)
    {
        lvr2::LVRLabelClassTreeItem *classItem = static_cast<lvr2::LVRLabelClassTreeItem *>(item);

        m_root->labelClasses.push_back(classItem->getLabelClassPtr());
        QTreeWidget::addTopLevelItem(item);
    }
}
void LVRLabelTreeWidget::setLabelRoot(lvr2::LabelRootPtr labelRoot)//, lvr2::LVRPickingInteractor interactor, QComboBox *comboBox)
{
    int id = 0;
    for(lvr2::LabelClassPtr classPtr: labelRoot->labelClasses)
    {

        QColor a;
        lvr2::LVRLabelClassTreeItem* classItem = new lvr2::LVRLabelClassTreeItem(classPtr->className, 0, true, 0,a );
        QTreeWidget::addTopLevelItem(classItem);
        for (lvr2::LabelInstancePtr instancePtr: classPtr->instances)
        {

            lvr2::Vector3d loadedColor = instancePtr->color;
            QColor color(loadedColor[0],loadedColor[1],loadedColor[2]);
            lvr2::LVRLabelInstanceTreeItem * instanceItem = new lvr2::LVRLabelInstanceTreeItem(instancePtr->instanceName, id++, instancePtr->labeledIDs.size(), true, true,color);

            /*
            classItem->addChild(instanceItem);

            comboBox->addItem(QString::fromStdString(instanceItem->getName()), id);
            interactor->newLabel(instanceItem);
            int comboBoxPos = comboBox->findData(instanceItem->getId());
            ComboBox->setCurrentIndex(comboBoxPos);
            */
        }
    }

}

lvr2::LabelRootPtr LVRLabelTreeWidget::getLabelRoot()
{
    return m_root;
}
