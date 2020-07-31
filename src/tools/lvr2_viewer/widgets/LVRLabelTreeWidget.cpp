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

void LVRLabelTreeWidget::itemSelected(int selectedId)
{

    QPalette pal;
    pal.setColor(QPalette::Background, Qt::black);
    if(m_selectedItem != nullptr)
    {
    //    m_selectedItem->setPalette(pal);
    }
    
    lvr2::LVRLabelInstanceTreeItem *foundItem = nullptr;
    //find item
    for(int i = 0; i < this->topLevelItemCount(); i++)
    {
        QTreeWidgetItem* item = this->topLevelItem(i);
        if(item->type() == lvr2::LVRLabelClassItemType)
        {
            lvr2::LVRLabelClassTreeItem *classItem = static_cast<lvr2::LVRLabelClassTreeItem *>(item);
            for (int j = 0; j < classItem->childCount(); j++)
            {
                if(classItem->child(j)->type() == lvr2::LVRLabelInstanceItemType)
                {
                    lvr2::LVRLabelInstanceTreeItem* instanceItem = static_cast<lvr2::LVRLabelInstanceTreeItem *>(classItem->child(j));
                    if(instanceItem->getId() == selectedId)
                    {
                        foundItem = instanceItem;
                        break;
                    }
                }
            }
        }
    }

    if (foundItem != nullptr)
    {
        if(m_selectedItem != nullptr)
        {
            for (int col = 0; col < m_selectedItem->columnCount(); col++)
            { 
                m_selectedItem->setBackground(col, QBrush());
            }
        }
 
        for (int col = 0; col < foundItem->columnCount(); col++)
        { 
            foundItem->setBackgroundColor(col, QColor(255, 255, 0, 100)); 
        }
        m_selectedItem = foundItem;
    }
}

void LVRLabelTreeWidget::setLabelRoot(lvr2::LabelRootPtr labelRoot, lvr2::LVRPickingInteractor* interactor, QComboBox *comboBox)
{
    int currid = 0;
    for(lvr2::LabelClassPtr classPtr: labelRoot->labelClasses)
    {

        bool first = true;

        QColor tmp;
        lvr2::LVRLabelClassTreeItem* classItem = new lvr2::LVRLabelClassTreeItem(classPtr->className, 0, true, true, tmp);
        QTreeWidget::addTopLevelItem(classItem);
        for (lvr2::LabelInstancePtr instancePtr: classPtr->instances)
        {
            lvr2::Vector3i loadedColor = instancePtr->color;
            QColor color(loadedColor[0],loadedColor[1],loadedColor[2]);
            if(first)
            {
                classItem->setColor(color);
            }
            int id = currid++;
            lvr2::LVRLabelInstanceTreeItem * instanceItem = new lvr2::LVRLabelInstanceTreeItem(instancePtr->instanceName, id,0 , true, true,color);

            classItem->addChild(instanceItem);

            comboBox->addItem(QString::fromStdString(instanceItem->getName()), id);
            interactor->newLabel(instanceItem);
            int comboBoxPos = comboBox->findData(instanceItem->getId());
            comboBox->setCurrentIndex(comboBoxPos);
            
            interactor->setLabel(id, instancePtr->labeledIDs);
        }
    }

}

lvr2::LabelRootPtr LVRLabelTreeWidget::getLabelRoot()
{
    return m_root;
}
