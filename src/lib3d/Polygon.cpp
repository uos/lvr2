/*
 * Polygon.cpp
 *
 *  Created on: 03.02.2009
 *      Author: twiemann
 */

#include "Polygon.h"

using Lib3D::Polygon;

Polygon::Polygon() {
  color_r = 0.0;
  color_g = 0.0;
  color_b = 1.0;
  color_alpha = 1.0;
}

Polygon::Polygon(const Polygon &other){
  
  for(size_t i = 0; i < other.vertices.size(); i++){
    vertices.push_back(other.vertices[i]);
  }

  color_r = other.color_r;
  color_g = other.color_g;
  color_b = other.color_b;
  color_alpha = other.color_alpha;
}

Polygon::~Polygon() {

}
