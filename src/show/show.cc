#include "show.h"


MCShow* MCShow::master = 0;

MCShow::MCShow(char* _fileName){

  fileName = _fileName;

  renderMesh = false;
  renderPoints = false;

  mesh_display_list = -1;
  point_display_list = -1;
  
  init();
  initGlut();
  initGlui();

}

void MCShow::init(){
  
  MCShow::master = this;

  cam.setSpeed(35);
  cam.setTurnSpeed(0.03);
  
  old_mx = 0;
  old_my = 0;

}

void MCShow::initOpenGL(){

  
  glPolygonMode (GL_FRONT_AND_BACK, GL_FILL); 
  glMatrixMode(GL_MODELVIEW);
  
  mat_specular[0] = 1.0f; mat_ambient[0]  = 0.4f; mat_diffuse[0]  = 0.8f;
  mat_specular[1] = 1.0f; mat_ambient[1]  = 0.4f; mat_diffuse[1]  = 0.8f;
  mat_specular[2] = 1.0f; mat_ambient[2]  = 0.4f; mat_diffuse[2]  = 0.8f;
  mat_specular[3] = 1.0f; mat_ambient[3]  = 1.0f; mat_diffuse[3]  = 1.0f; 
  
  mat_shininess   = 50.0f;

  light0_position[0] =   1.0f; light0_ambient[0] = 0.3f; light0_diffuse[0] = 0.8f;
  light0_position[1] =   1.0f; light0_ambient[1] = 0.3f; light0_diffuse[1] = 0.8f;
  light0_position[2] =   0.0f; light0_ambient[2] = 0.3f; light0_diffuse[2] = 0.8f;
  light0_position[3] =   1.0f; light0_ambient[3] = 0.1f; light0_diffuse[3] = 1.0f;

  light1_position[0] =   0.0f; light1_ambient[0] = 0.1f; light1_diffuse[0] = 0.5f;
  light1_position[1] =  -1.0f; light1_ambient[1] = 0.1f; light1_diffuse[1] = 0.5f;
  light1_position[2] =   0.0f; light1_ambient[2] = 0.1f; light1_diffuse[2] = 0.5f;
  light1_position[3] =   1.0f; light1_ambient[3] = 1.0f; light1_diffuse[3] = 1.0f;
  
  //Lichtquelle 1
  glLightfv(GL_LIGHT0, GL_AMBIENT,  light0_ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE,  light0_diffuse);
  glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
  glEnable(GL_LIGHT0);

//   glLightfv(GL_LIGHT1, GL_AMBIENT,  light1_ambient);
//   glLightfv(GL_LIGHT1, GL_DIFFUSE,  light1_diffuse);
//   glLightfv(GL_LIGHT1, GL_POSITION, light1_position);
//   glEnable(GL_LIGHT1);

  GLfloat light_position[] = { 0.0, 0.0, -1.0, 0.0 }; //Licht
  GLfloat spot_direction[] = {0.0, 0.0, -1.0, 0.0};
  GLfloat light_ambient[] = { 0.4, 0.4, 0.4, 1.0 };
  GLfloat light_diffuse[] = { 0.7, 0.7, 0.7, 1.0 };
  GLfloat light_specular[] = { 0.2, 0.2, 0.2, 1.0 };
  glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, light_diffuse);
  glLightfv(GL_LIGHT1, GL_SPECULAR, light_specular);
  glLightfv(GL_LIGHT1, GL_POSITION, light_position);
  glEnable(GL_LIGHT1);

  
  glEnable(GL_LIGHTING);
  
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glShadeModel (GL_SMOOTH); 


}

void MCShow::callback_motion(int x, int y){
  MCShow::master->mouseMoved(x, y);
}

void MCShow::initGlut(){

  //Init glut
  int dummy_argc = 1;
  char *dummy_argv[1];
  dummy_argv[0] = new char[255];
  dummy_argv[0] = "Test";
  
  glutInit(&dummy_argc, dummy_argv);
  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowPosition(100,100);
  glutInitWindowSize(762, 576);
  glutCreateWindow("3D-Viewer");

//Register callback functions
  glutDisplayFunc(MCShow::callback_render);
  glutReshapeFunc(MCShow::callback_reshape);
  glutIdleFunc(MCShow::callback_render);
  glutMouseFunc(MCShow::callback_mouse);
  glutMotionFunc(MCShow::callback_motion);
  
  initOpenGL();

  //Read Data
  readFile(fileName);
 
  //Call main loop
  glutMainLoop();

}

void MCShow::initGlui(){

  cout << "initGlui()" << endl;

}

void MCShow::initDisplayLists(){

  //Delete already created display lists
  if(mesh_display_list != -1) glDeleteLists(mesh_display_list, 1);
  if(point_display_list != -1) glDeleteLists(point_display_list, 1);

 
  //Compile mesh list
  if(renderMesh){
    mesh_display_list = glGenLists(1);
    glNewList(mesh_display_list, GL_COMPILE);
    plyReader.mesh.render();
    glEndList();
  }
 
 
}



void MCShow::callback_mouse(int button, int state, int x, int y){
  MCShow::master->mousePressed(button, state, x, y);
}

void MCShow::mousePressed(int button, int state, int x, int y){

  mouseButton = button;
  mouseButtonState = state;
  
}

void MCShow::mouseMoved(int x, int y){

  int dx = x - old_mx;
  int dy = y - old_my;

  if(mouseButton == GLUT_LEFT_BUTTON) moveXY(dx, dy);
  if(mouseButton == GLUT_RIGHT_BUTTON) moveHead(dx, dy);
  if(mouseButton == GLUT_MIDDLE_BUTTON) moveZ(dx, dy);

  cam.apply();
  
  old_mx = x;
  old_my = y;
  
}

void MCShow::moveXY(int dx, int dy){

  if(abs(dx) > MOUSE_SENSITY){

    if(dx > 0)
	 cam.turn_right();
    else
	 cam.turn_left();
    
  }

  if(abs(dy) > MOUSE_SENSITY){

    if(dy > 0)
	 cam.move_backward();
    else
	 cam.move_forward();
    
  }
  
}

void MCShow::moveZ(int dx, int dy){

  if(abs(dy) > MOUSE_SENSITY){

    if(dy > 0)
	 cam.move_up();
    else
	 cam.move_down();
    
  }
  
}

void MCShow::moveHead(int dx, int dy){

  if(abs(dy) > MOUSE_SENSITY){

    if(dy > 0)
	 cam.turn_up();
    else
	 cam.turn_down();
    
  }

  if(abs(dx) > MOUSE_SENSITY){

    if(dx > 0)
	 cam.turn_right();
    else
	 cam.turn_left();
    
  }

}

void MCShow::callback_reshape(int w, int h){
  MCShow::master->reshape(w, h);
}

void MCShow::reshape(int w, int h){
  if(h == 0) h = 1;

  float ratio = 1.0* w / h;

  // Reset the coordinate system before modifying
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
	
  // Set the viewport to be the entire window
  glViewport(0, 0, w, h);

  // Set the correct perspective.
  gluPerspective(45,ratio,1,100000);
  glMatrixMode(GL_MODELVIEW);

  // Set 'LookAt'
  cam.apply();
  
}

void MCShow::callback_idle(){
  MCShow::master->idle();
}

void MCShow::idle(){
  //usleep(10000);
}

void MCShow::callback_render(){
  MCShow::master->render();
}

void MCShow::setColorMaterial(float r, float g, float b){

  //cout << "R: " << r << " G: " << g << " B: " << b << endl;

  mat_specular[0] = 0.7f; mat_ambient[0]  = 0.5f * r; mat_diffuse[0]  = r;
  mat_specular[1] = 0.7f; mat_ambient[1]  = 0.5f * g; mat_diffuse[1]  = g;
  mat_specular[2] = 0.7f; mat_ambient[2]  = 0.5f * b; mat_diffuse[2]  = b;
  mat_specular[3] = 1.0f; mat_ambient[3]  = 1.0f; mat_diffuse[3]  = 1.0f; 
  
  glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
  glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
  glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
  glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse); 
  
}


void MCShow::render(){

  glClearColor(1.0, 1.0, 1.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   

  if(renderMesh) glCallList(mesh_display_list);
  
  glFinish();
  glutSwapBuffers();

}

void MCShow::readPolygonFile(char* fileName){


}

void MCShow::readPointFile(char* fileName){

}

void MCShow::readFile(char* filename){

  char* ext = strchr(fileName, '.');

  if(strcmp(ext, ".ply") == 0){
    cout << "##### Reading PLY File: " << filename << endl;
    plyReader.read(filename);
    renderMesh = true;
    cout << "##### Finished reading. Current mesh has "
	    << plyReader.mesh.indices.size() / 3 << " faces with " 
	    << plyReader.mesh.vertices.size() << " vertices. " << endl;
    initDisplayLists();
  }

  if(strcmp(ext, ".bor") == 0){
 
    
  }

  if(strcmp(ext, ".pts") == 0){

    
  }  


}

MCShow::~MCShow(){

  for(size_t i = 0; i < points.size(); i++){
    delete[] points[i];
  }
  points.clear();
  
  for(size_t i = 0; i < polygons.size(); i++){
    for(size_t j = 0; j < polygons[i].size(); j++){
	 delete[] polygons[i][j];
    }
    polygons[i].clear();
  }
  polygons.clear();
  
}



