/*
 * ObjectHandler.cpp
 *
 *  Created on: 27.08.2008
 *      Author: twiemann
 */

#include "ObjectHandler.h"
#include "ObjectHandlerMoc.cpp"

ObjectHandler::ObjectHandler(QListWidget* l) {
	listWidget = l;
	selectedObject = 0;
}

void ObjectHandler::objectEdited(QListWidgetItem * item){
	int index = listWidget->currentRow();
	if(item->checkState() == Qt::Checked){
		objects[index]->setVisible(true);
	} else {
		objects[index]->setVisible(false);
	}
}

void ObjectHandler::objectSelected(){
	objects[selectedObject]->showAxes(false);
	objects[selectedObject]->setActive(false);
	selectedObject = listWidget->currentRow();
	objects[selectedObject]->showAxes(true);
	objects[selectedObject]->setActive(true);
}

void ObjectHandler::transformSelectedObject(Matrix4 m){
	cout << "TRANSFORM: " << m;
	objects[selectedObject]->setTransformationMatrix(m);
}

void ObjectHandler::transformSelectedObject(int mode, int d){

	Renderable* obj = objects[selectedObject];

	switch(mode){
	case 0 : d > 0 ? obj->yaw() : obj->yaw(true); break;
	case 1 : d > 0 ? obj->pitch() : obj->pitch(true); break;
	case 2 : d > 0 ? obj->roll() : obj->roll(true); break;
	case 3 : d > 0 ? obj->strafe(): obj->strafe(true); break;
	case 4 : d > 0 ? obj->lift(): obj->lift(true); break;
	case 5 : d > 0 ? obj->accel(): obj->accel(true); break;
	case 6 : d > 0 ? obj->rotX() : obj->rotX(true); break;
	case 7 : d > 0 ? obj->rotY() : obj->rotY(true); break;
	case 8 : d > 0 ? obj->rotZ() : obj->rotZ(true); break;
	case 9 : d > 0 ? obj->moveX() : obj->moveX(true); break;
	case 10: d > 0 ? obj->moveY() : obj->moveY(true); break;
	case 11: d > 0 ? obj->moveZ() : obj->moveZ(true); break;
	default: cout << "Warning: Transformation undefined: " << mode << endl;
	}

}

void ObjectHandler::addObject(Renderable* o){

	//Save pointer
	objects.push_back(o);

	//Add to list
	QListWidgetItem* item = new QListWidgetItem;
	item->setCheckState(Qt::Checked);
	item->setText(QString(o->Name().c_str()));
	listWidget->insertItem(objects.size() - 1, item);
}

ObjectHandler::~ObjectHandler() {
	for(size_t i = 0; i < objects.size(); i++) delete objects[i];
}


