#ifndef __SHOW_H__
#define __SHOW_H__

#include <math.h>

#include <fstream>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::ios;
using std::ifstream;

#include <vector>
using std::vector;

#ifndef MAC_OSX
#include <GL/gl.h>	    /* Header File For The OpenGL32 Library */
#include <GL/glu.h>	    /* Header File For The GLu32 Library */
#include <GL/glext.h>   /* Header File For The OpenGL32 Library */
#include <GL/glx.h>     /* Header File For The glx libraries */
#include <GL/glxext.h>  /* Header File For The OpenGL32 Library */
#include <GL/glut.h>    /* Header File For The glu toolkit */
#include <GL/gle.h>     /* Header File For The gle toolkit */
#else
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#endif

#include "camera.h"
#include "../io/plyReader.h"

#define MOUSE_SENSITY 6

class MCShow{

public:
  MCShow(char* fileName);
  ~MCShow();

  static void callback_reshape(int w, int h);
  static void callback_render();
  static void callback_idle();
  static void callback_mouse(int button, int state, int x, int y);
  static void callback_motion(int x, int y);

  void readFile(char* fileName);

  static MCShow* master;
  
protected:
  void render();
  void reshape(int w, int h);
  void idle();
  void mouseMoved(int x, int y);
  void mousePressed(int button, int state, int x, int y);

  void setColorMaterial(float r, float g, float b);

  void readPolygonFile(char* filename);
  void readPointFile(char* filename);

  
private:

  void moveXY(int dx, int dy);
  void moveZ(int dx, int dy);
  void moveHead(int dx, int dy);

  void init();
  void initGlut();
  void initGlui();
  void initOpenGL();
  void initDisplayLists();

  int display_list_index;
  
  int old_mx;
  int old_my;

  int mouseButton;
  int mouseButtonState;

  float light0_position[4];
  float light0_ambient[4];
  float light0_diffuse[4];

  float light1_position[4];
  float light1_ambient[4];
  float light1_diffuse[4];

  float mat_specular[4];
  float mat_shininess;
  float mat_ambient[4];
  float mat_diffuse[4];
  
  char* fileName;

  PLYReader plyReader;
  
  GLCamera cam;

  vector<vector<double*> > polygons;
  vector<float*> points;
  vector<float*> normals;
  
  int renderMode;

  bool renderMesh;
  bool renderPoints;
  bool renderNormals;

  int mesh_display_list;
  int point_display_list;
  int normal_display_list;
  
};

#endif
