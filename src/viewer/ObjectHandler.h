/*
 * ObjectHandler.h
 *
 *  Created on: 27.08.2008
 *      Author: Thomas Wiemann
 */

#ifndef OBJECTHANDLER_H_
#define OBJECTHANDLER_H_


#include <QtGui>

#include <iostream>
#include <vector>

using namespace std;

#include "../model3d/Renderable.h"
#include "../model3d/GroundPlane.h"

#include "objectdialog.h"

class ObjectHandler: public QObject{

	Q_OBJECT

public:
	ObjectHandler(QListWidget*);
	virtual ~ObjectHandler();

	void addObject(Renderable* o);
	void showObjectDialog();
	void objectEdited(QListWidgetItem* item);
	void objectSelected();

	void transformSelectedObject(Matrix4 m);
	void transformSelectedObject(int mode, double d);

	void printTransformation(){cout << objects[selectedObject]->getTransformation();};

	inline void renderObjects();

private:

	int selectedObject;

	vector<Renderable*> objects;
	QListWidget* listWidget;

};

inline void ObjectHandler::renderObjects(){
	for(size_t i = 0; i < objects.size(); i++) objects[i]->render();
}

#endif /* OBJECTHANDLER_H_ */
