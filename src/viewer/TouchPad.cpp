/*
 * TouchPad.cpp
 *
 *  Created on: 04.09.2008
 *      Author: twiemann
 */

#include "TouchPad.h"
//#include "TouchPadMoc.cpp"

TouchPad::TouchPad(QWidget* w, MoveDock* md) : QFrame(w){

	ui = md;

	//Setup user interface
	ui->comboBoxAction->addItem("Yaw / Roll");
	ui->comboBoxAction->addItem("Pitch / Roll");
	ui->comboBoxAction->addItem("Accel");
	ui->comboBoxAction->addItem("Lift");
	ui->comboBoxAction->addItem("Strafe");
	ui->comboBoxAction->addItem("ROT XY");
	ui->comboBoxAction->addItem("ROT Z");
	ui->comboBoxAction->addItem("MOVE XY");
	ui->comboBoxAction->addItem("MOVE Z");

	connect(ui->comboBoxAction, SIGNAL(currentIndexChanged(int)),
			this, SLOT(indexChanged(int)));

	transformation_mode = YAW_ROLL;
	sensity = 1;

	old_x = 0;
	old_y = 0;
}

void TouchPad::indexChanged(int index){
	transformation_mode = index;
}

bool TouchPad::checkInput(int d){
	return (fabs(d) > sensity && d < 10 * sensity);
}

void TouchPad::mouseMoveEvent(QMouseEvent* e){

	double dx = (double)old_x - e->x();
	double dy = (double)old_y - e->y();

	switch(transformation_mode){
	case YAW_ROLL:
		if(checkInput(dx)) emit(transform( 2, dx));
		if(checkInput(dy)) emit(transform( 0, dy));
		break;
	case PITCH_ROLL:
		if(checkInput(dx)) emit(transform( 1, dx));
		if(checkInput(dy)) emit(transform( 2, dy));
		break;
	case ACCEL:
		if(checkInput(dy)) emit(transform( 5, dy));
		break;
	case LIFT:
		if(checkInput(dy)) emit(transform( 4, dy));
		break;
	case STRAFE:
		if(checkInput(dx)) emit(transform( 3, dx));
		break;
	case ROT_XY:
		if(checkInput(dy)) emit(transform( 6, dy));
		if(checkInput(dx)) emit(transform( 7, dx));
		break;
	case ROT_Z:
		if(checkInput(dy)) emit(transform( 8, dy));
		break;
	case MOVE_XY:
		if(checkInput(dx)) emit(transform( 9, dx));
		if(checkInput(dy)) emit(transform(10, dy));
		break;
	case MOVE_Z:
		if(checkInput(dy)) emit(transform(10, dy));
		break;
	default:
		break;
	}

	old_x = e->x();
	old_y = e->y();

}

TouchPad::~TouchPad() {

}
