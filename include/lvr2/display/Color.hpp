/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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

#endif /* COLOR_H_ */
