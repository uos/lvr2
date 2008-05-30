#include "camera.h"

GLCamera::GLCamera(){
  px = 0;
  py = 0;
  pz = 0;

  //px = py = pz = 0.0;
  ix = 0.0;
  iy = 20.0;
  iz = 100.0;
  turn_speed = 0.2;
  speed = 0.2;
  rotX = rotY = 0.0;

  rotX = -0.42;
  rotY = -0.3;

  light_position[0] = 0.0;
  light_position[1] = 0.0;
  light_position[2] = 0.0;
  light_position[3] = -1.0;
  
}

GLCamera::GLCamera(double x, double y, double z){
  
  px = py = pz = 0.0;
  //px = -37.1692;
  //py = 108.0;
  //pz = -254.266;

   lx = ly = 0.0;
   lz = 1.0;
   speed = turn_speed = 0.2;
   ix = x;
   iy = y;
   iz = z;

   rotX = 0.0;
   rotY = 0.0;
  
}

void GLCamera::setRotMatrix(float m[16]){

  double lookAt[4];

  lookAt[0] = 0.0;
  lookAt[1] = 0.0;
  lookAt[2] = 1.0;
  lookAt[3] = 0.0;
  
  double newLookAt[4];


  //Rotationsmatrix anwenden und Drehwinkel
  //rückwärts ausrechnen
  
  for(int i = 0; i < 4; i++){

    newLookAt[i] =
	 m[4*i+0] * lookAt[0] +
	 m[4*i+1] * lookAt[1] +
	 m[4*i+2] * lookAt[2] +
	 m[4*i+3] * lookAt[3];
   
  }

  rotY = atan2(newLookAt[2], newLookAt[0]);
  rotX = atan2(newLookAt[1], newLookAt[2]);
  
}

void GLCamera::move_left(){

  px -= speed * sin(PH - rotY);
  pz -= speed * cos(PH - rotY);
  
}

void GLCamera::move_right(){

  px += speed * sin(PH + rotY);
  pz += speed * cos(PH + rotY);
  
}

void GLCamera::move_forward(){

  px += speed * sin(rotY);
  pz += speed * cos(rotY);

}
void GLCamera::move_backward(){

  px -= speed * sin(rotY);
  pz -= speed * cos(rotY);

}

void GLCamera::turn_up(){

  if(rotX < PH) rotX -= turn_speed;
  
}

void GLCamera::turn_down(){

  if(rotX > -PH) rotX += turn_speed;
  
}

void GLCamera::turn_left(){

  rotY -= turn_speed;
  
}

void GLCamera::turn_right(){
  
  rotY += turn_speed;
  
}

void GLCamera::move_up(){py += speed;}
void GLCamera::move_down(){py -= speed;}

void GLCamera::apply(){

  lx = ix + px + sin(rotY);
  lz = iz - pz - cos(rotY);
  ly = iy + py + sin(rotX);

  //cout << ix + px << " " << iz + pz << " " << iy + py << endl;
  
  glLoadIdentity();
   
  gluLookAt(ix + px, iy + py, iz - pz,
		  lx, ly, lz,
		  0.0, 1.0, 0.0);
  
}
