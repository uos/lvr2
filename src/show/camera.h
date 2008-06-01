#ifndef __GLCAMERA_H__
#define __GLCAMERA_H__

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
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#endif


#define PH  1.570796326

class GLCamera{

public:
  GLCamera();
  GLCamera(double x, double y, double z);
  ~GLCamera(){};

  void move_left();
  void move_right();
  void move_up();
  void move_down();
  void move_forward();
  void move_backward();

  void turn_left();
  void turn_right();
  void turn_up();
  void turn_down();

  void setRotMatrix(float[16]);
  
  void setSpeed(double _speed){speed = _speed;};
  void setTurnSpeed(double _speed) {turn_speed = _speed;};
  
  void apply();
  
private:

  float light_position[4];

  double speed;
  double turn_speed;

  double rotY;
  double rotX;
  
  double px;
  double py;
  double pz;

  double ix;
  double iy;
  double iz;

  double lx;
  double ly;
  double lz;
  
};

#endif
