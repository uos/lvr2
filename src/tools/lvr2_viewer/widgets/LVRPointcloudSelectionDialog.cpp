#include "LVRPointcloudSelectionDialog.hpp"

LVRPointcloudSelectionDialog::LVRPointcloudSelectionDialog(QStringList strList,QWidget *parent): QDialog(parent)
{
    setWindowTitle("Choose Pointclouds which shall be labeled");

    widget = new QListWidget;

    widget->addItems(strList);

    QListWidgetItem* item = 0;
    for(int i = 0; i < widget->count(); ++i){
        item = widget->item(i);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Unchecked);
    }
    viewBox = new QGroupBox(tr("Required components"));
    buttonBox = new QDialogButtonBox;
    saveButton = buttonBox->addButton(QDialogButtonBox::Save);
    closeButton = buttonBox->addButton(QDialogButtonBox::Close);
QVBoxLayout* viewLayout = new QVBoxLayout;
viewLayout->addWidget(widget);
    viewBox->setLayout(viewLayout);

    QHBoxLayout* horizontalLayout = new QHBoxLayout;
    horizontalLayout->addWidget(buttonBox);

    QVBoxLayout* mainLayout = new QVBoxLayout;
    mainLayout->addWidget(viewBox);
    mainLayout->addLayout(horizontalLayout);

    setLayout(mainLayout);
    QObject::connect(widget, SIGNAL(itemChanged(QListWidgetItem*)),this, SLOT(highlightChecked(QListWidgetItem*)));
    QObject::connect(saveButton, SIGNAL(clicked()), this, SLOT(save()));
    QObject::connect(closeButton, SIGNAL(clicked()), this, SLOT(close()));
}

QStringList LVRPointcloudSelectionDialog::getChecked()
{
    QStringList out;
    QListWidgetItem* item = 0;
    for(int i = 0; i < widget->count(); i++)
    {
        item = widget->item(i);
        if(item->checkState() == Qt::Checked)
        {
            out << item->text();
        }

    }
    return out;
}

void LVRPointcloudSelectionDialog::highlightChecked(QListWidgetItem *item){
    if(item->checkState() == Qt::Checked)
        item->setBackgroundColor(QColor("#ffffb2"));
    else
        item->setBackgroundColor(QColor("#ffffff"));
}

void LVRPointcloudSelectionDialog::save(){

        QFile file("required_components.txt");
            if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
                        return;

                QTextStream out(&file);
                    out << "Choose Pointclouds to label:" << "\n";

                        QListWidgetItem* item = 0;
                            for(int i = 0; i < widget->count(); ++i){
                                        item = widget->item(i);
                                                if(item->checkState() == Qt::Checked)
                                                                out << item->text() << "\n";
                                                    }

                                QMessageBox::information(this, tr("Checkable list in Qt"),
                                                                           tr("Required components were saved."),
                                                                                                              QMessageBox::Ok);
}
