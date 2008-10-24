/*
 * EventHandler.cpp
 *
 *  Created on: 27.08.2008
 *      Author: Thomas Wiemann
 */

#include "EventHandler.h"
#include "EventHandlerMoc.cpp"

void EventHandler::camTurnLeft() {viewport->camTurnLeft();};
void EventHandler::camTurnRight() {viewport->camTurnRight();};
void EventHandler::camLookUp() {viewport->camLookUp();};
void EventHandler::camLookDown(){viewport->camLookDown();};
void EventHandler::camMoveForward() {viewport->camMoveForward();};
void EventHandler::camMoveBackward() {viewport->camMoveBackward();};
void EventHandler::camMoveUp(){ viewport->camMoveUp();};
void EventHandler::camMoveDown(){ viewport->camMoveDown();};
void EventHandler::camIncTransSpeed(){viewport->incTranslationSpeed(10);};
void EventHandler::camDecTransSpeed(){viewport->decTranslationSpeed(10);};

void EventHandler::printCurrentTransformation(){ objectHandler->printTransformation();};


EventHandler::EventHandler(ViewerWindow* v) {
	mainWindow = v;
	init();
}

void EventHandler::init(){

	objectHandler = new ObjectHandler(mainWindow->ListWidget());
	screen_width = 0;
	screen_height = 0;
	viewport = new Viewport(screen_width, screen_height);
	viewport->setTranslationSpeed(100);

	Matrix4 mirror;
	mirror.set(10, -1.0);
	Matrix4 rotation = Matrix4(Vertex(0.0, 1.0, 0.0), 3.14159);
	transform_to_gl = mirror * rotation;

}

void EventHandler::showStatusMessage(){

	Vertex pos = viewport->getPosition();
	Vertex angles = viewport->getOrientation();
	stringstream ss;
	ss << "Pos: ( " << pos.x << " , " << pos.y << " , " << pos.z << " ) ";
	ss << "Orientation: ( " << angles.x << " , " << angles.y << " , " << angles.z << ")";

	string str = ss.str();
	mainWindow->statusBar()->showMessage(QString(str.c_str()));
}

void EventHandler::keyUp(){

}

void EventHandler::keyScale(){
	double d = QInputDialog::getDouble(mainWindow,
			                           "Enter Scaling Factor:",
			                           "", 1.0, -100000, 100000, 4);
	objectHandler->transformSelectedObject(12, d);
}

void EventHandler::action_topView(){
	viewport->topView();
	viewport->applyTransformations();
}
void EventHandler::action_perspectiveView(){
	viewport->perspectiveView();
	viewport->applyTransformations();
}

void EventHandler::action_editObjects(QListWidgetItem* item){
	objectHandler->objectEdited(item);
	emit(updateGLWidget());
}

void EventHandler::action_objectSelected(QListWidgetItem* item){
	objectHandler->objectSelected();
}

void EventHandler::action_enterMatrix(){

	QDialog matrix_dialog(mainWindow);
	Ui::MatrixDialog matrix_dialog_ui;
	matrix_dialog_ui.setupUi(&matrix_dialog);

	int result = matrix_dialog.exec();

	if(result == QDialog::Accepted){
		Matrix4 m;
		m.set(0 , matrix_dialog_ui.doubleSpinBox00->value());
		m.set(1 , matrix_dialog_ui.doubleSpinBox01->value());
		m.set(2 , matrix_dialog_ui.doubleSpinBox02->value());
		m.set(3 , matrix_dialog_ui.doubleSpinBox03->value());
		m.set(4 , matrix_dialog_ui.doubleSpinBox04->value());
		m.set(5 , matrix_dialog_ui.doubleSpinBox05->value());
		m.set(6 , matrix_dialog_ui.doubleSpinBox06->value());
		m.set(7 , matrix_dialog_ui.doubleSpinBox07->value());
		m.set(8 , matrix_dialog_ui.doubleSpinBox08->value());
		m.set(9 , matrix_dialog_ui.doubleSpinBox09->value());
		m.set(10, matrix_dialog_ui.doubleSpinBox10->value());
		m.set(11, matrix_dialog_ui.doubleSpinBox11->value());
		m.set(12, matrix_dialog_ui.doubleSpinBox12->value());
		m.set(13, matrix_dialog_ui.doubleSpinBox13->value());
		m.set(14, matrix_dialog_ui.doubleSpinBox14->value());
		m.set(15, matrix_dialog_ui.doubleSpinBox15->value());
		objectHandler->transformSelectedObject(m);
		cout << m;
	}
}

void EventHandler::transform_from_file(){

	//Get file name
	QString filename = QFileDialog::getOpenFileName(NULL, tr("Load Matrix..."), "/home/twiemann/software/lufthansa", tr("Matrix Files (*.mat)"));

	//Load matrix
	ifstream in(filename.toStdString().c_str());

	double* matrix_data = new double[16];

	for(int i = 0; i < 16; i++){
		if(!in.good()){
			cout << "Warning: EventHandler::transform_from_file: File corrupted!" << endl;
			return;
		}
		in >> matrix_data[i];
	}

	Matrix4 trafo(matrix_data);
	objectHandler->transformSelectedObject(trafo);

	delete[] matrix_data;
}

void EventHandler::touchpad_transform(int mode, double d){
	objectHandler->transformSelectedObject(mode, d);
	emit(updateGLWidget());

}

void EventHandler::createDefaultObjects(){

	objectHandler->addObject(new GroundPlane(100, 30));
	objectHandler->addObject(new CoordinateAxes(50));

}

EventHandler::~EventHandler() {
	delete objectHandler;
}

void EventHandler::resize_event(int w, int h){
	viewport->resize(w, h);
	screen_width = w;
	screen_height = h;
}

void EventHandler::action_file_open(){

	QFileDialog file_dialog;
	QStringList file_names;
	QStringList file_types;

	file_types << "Point Clouds (*.pts)"
	           << "Points and Normals (*.nor)"
	           << "PLY Meshes (*.ply)"
	           << "Polygonal Meshes (*.bor)"
	           << "All Files (*.*)";

	//Set Title
	file_dialog.setWindowTitle("Open File");
	file_dialog.setFileMode(QFileDialog::ExistingFile);
	file_dialog.setFilters(file_types);

	if(file_dialog.exec()){
		file_names = file_dialog.selectedFiles();
	} else {
		return;
	}

	//Get filename from list
	string file_name = file_names.constBegin()->toStdString();

	loadObject(file_name);

}

void EventHandler::loadObject(string file_name){

	//Get extension
	string extension;
	size_t pos = file_name.find_last_of(".");
	extension = file_name.substr(pos+1);

	if(extension == "pts"){
		PointCloud* pc = new PointCloud(file_name);
		pc->setName(file_name);
		objectHandler->addObject(pc);
	} else if(extension == "3d"){

	} else if(extension == "nor"){
		NormalCloud* nc = new NormalCloud(file_name);
		nc->setName(file_name);
		objectHandler->addObject(nc);
	} else if(extension == "bor"){

	} else if(extension == "ply"){
		StaticMesh* mesh = new StaticMesh(file_name);
		mesh->setName(file_name);
		objectHandler->addObject(mesh);
	}

}
