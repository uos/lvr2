/*
 * Color.h
 *
 *  Created on: 31.08.2010
 *      Author: Thomas Wiemann
 */

#ifndef COLOR_H_
#define COLOR_H_

namespace lvr2
{

enum Color {RED, GREEN, BLUE, YELLOW, PINK, ORANGE, LIGHTBLUE, LIGHTGREY, BLACK, WHITE};

enum ColorTable {BASIC, LIGHT, HIGHLIGHT};

class Colors{
public:
	static void getColor(float* c, Color name, ColorTable table = BASIC);

	static float PrimeColorTable[][3];
	static float LightColorTable[][3];
	static float HighlightColorTable[][3];

	static unsigned long getRGBIndex(unsigned char r, unsigned char g, unsigned char b)
	{
		return ((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff);
	}
};

} // namespace lvr2

#include "Color.cpp"

#endif /* COLOR_H_ */
