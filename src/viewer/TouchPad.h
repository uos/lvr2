/*
 * TouchPad.h
 *
 *  Created on: 04.09.2008
 *      Author: twiemann
 */

#ifndef TOUCHPAD_H_
#define TOUCHPAD_H_

#include <iostream>
using namespace std;

#include <QtGui>

#include "MoveDock.cpp"
using namespace Ui;

enum TransformMode{YAW_ROLL, PITCH_ROLL, ACCEL, LIFT, STRAFE,
	               ROT_XY, ROT_Z, MOVE_XY, MOVE_Z};

class TouchPad : public QFrame{

	Q_OBJECT

public:
	TouchPad(QWidget* w, MoveDock*);
	virtual ~TouchPad();

	virtual void mouseMoveEvent(QMouseEvent* e);

public slots:
	void indexChanged(int);

signals:
	void transform(int, double);



private:

	bool checkInput(int d);

	MoveDock* ui;
	int transformation_mode;

	int old_x;
	int old_y;
	int sensity;
};

#endif /* TOUCHPAD_H_ */
