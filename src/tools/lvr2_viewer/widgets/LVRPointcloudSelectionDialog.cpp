#include "LVRPointcloudSelectionDialog.hpp"

LVRPointcloudSelectionDialog::LVRPointcloudSelectionDialog(QStringList strList,QWidget *parent): QDialog(parent)
{
    setWindowTitle("Choose Pointclouds");

    widget = new QListWidget;

    widget->addItems(strList);

    QListWidgetItem* item = 0;
    for(int i = 0; i < widget->count(); ++i){
        item = widget->item(i);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked);
    }
    viewBox = new QGroupBox(tr("To be labeled poointclouds"));
    buttonBox = new QDialogButtonBox;
    acceptButton = buttonBox->addButton(QDialogButtonBox::Ok);
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
    QObject::connect(acceptButton, SIGNAL(clicked()), this, SLOT(accept()));
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

