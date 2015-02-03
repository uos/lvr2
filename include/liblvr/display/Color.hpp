/*
 * Color.h
 *
 *  Created on: 31.08.2010
 *      Author: Thomas Wiemann
 */

#ifndef COLOR_H_
#define COLOR_H_

enum Color {RED, GREEN, BLUE, YELLOW, PINK, ORANGE, LIGHTBLUE, LIGHTGREY, BLACK, WHITE};

enum ColorTable {BASIC, LIGHT, HIGHLIGHT};

class Colors{
public:
	static void getColor(float* c, Color name, ColorTable table = BASIC);

	static float PrimeColorTable[][3];
	static float LightColorTable[][3];
	static float HighlightColorTable[][3];
};


#endif /* COLOR_H_ */
