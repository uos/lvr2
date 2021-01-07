#include "LVRLabelTreeWidget.hpp"
#include <iostream>
#include "LVRItemTypes.hpp"
#include "LVRLabelClassTreeItem.hpp"
#include "LVRLabelInstanceTreeItem.hpp"
#include <QInputDialog>

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
    m_root = labelRoot;
    QStringList instanceNames;
    for(lvr2::LabelClassPtr classPtr: labelRoot->labelClasses)
    {
        for (lvr2::LabelInstancePtr instancePtr: classPtr->instances)
        {
            instanceNames << QString::fromStdString(instancePtr->instanceName);
        }
    }

    QInputDialog dialog;
    dialog.setOptions(QInputDialog::UseListViewForComboBoxItems);
    dialog.setComboBoxItems(instanceNames);
    dialog.setWindowTitle("Choose the instance Name for Points without label");
    if (!dialog.exec())
    {
        return;
    }

    for(lvr2::LabelClassPtr classPtr: labelRoot->labelClasses)
    {
        bool first = true;

        QColor tmp;
        lvr2::LVRLabelClassTreeItem* classItem = new lvr2::LVRLabelClassTreeItem(classPtr);
        QTreeWidget::addTopLevelItem(classItem);
        for (lvr2::LabelInstancePtr instancePtr: classPtr->instances)
        {
            QColor color(instancePtr->color[0],instancePtr->color[1],instancePtr->color[2]);
            if(first)
            {
                classItem->setColor(color);
            }
            int id = 0;
            if(dialog.textValue().toStdString() != instancePtr->instanceName)
            {
                id = getNextId();
            }
            lvr2::LVRLabelInstanceTreeItem * instanceItem = new lvr2::LVRLabelInstanceTreeItem(instancePtr, id);

            classItem->addChildnoChanges(instanceItem);

            interactor->newLabel(instanceItem);
            comboBox->addItem(QString::fromStdString(instanceItem->getName()), id);
            int comboBoxPos = comboBox->findData(instanceItem->getId());
            interactor->setLabel(id, instancePtr->labeledIDs);
            comboBox->setCurrentIndex(comboBoxPos);
        }
    }

    interactor->refreshActors();

}
int LVRLabelTreeWidget::getNextId()
{
    return m_id++;
}

lvr2::LabelRootPtr LVRLabelTreeWidget::getLabelRoot()
{
    return m_root;
}

QStringList LVRLabelTreeWidget::getTopLevelItemNames()
{
    QStringList out;
    for (int i = 0; i < topLevelItemCount(); i++)
    {
        out << topLevelItem(i)->text(0);
    }
    return out;
}
