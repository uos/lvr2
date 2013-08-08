#include "display/Color.hpp"

float Colors::PrimeColorTable[][3] = {
	{ 0.80, 0.00, 0.00},	// Red 3
	{ 0.12, 0.80, 0.12},    // Lime Green
	{ 0.00, 0.00, 0.50},	// Navy Blue
	{ 0.93, 0.93, 0.00},	// Yellow 2
	{ 0.92, 0.23, 0.55}, 	// Violet Red 2
	{ 1.00, 0.73, 0.06},    // Dark Golden Rod 1
	{ 0.09, 0.45, 0.80},	// Dodger Blue 3
	{ 0.50, 0.50, 0.50},    // Grey
	{ 0.00, 0.00, 0.00},    // Black
	{ 1.00, 1.00, 1.00}     // White
};

float Colors::LightColorTable[][3] = {
	{ 1.00, 0.00, 0.00},    // Red
	{ 0.00, 1.00, 0.00},    // Green
	{ 0.69, 0.40, 1.00},    // Royal Blue
	{ 1.00, 1.00, 0.00},    // Yellow
	{ 0.93, 0.47, 0.62},    // Pale Violet Red 2
	{ 1.00, 0.65, 0.00},    // Orange
	{ 0.00, 0.75, 1.00},    // Deep Sky Blue 1
	{ 0.75, 0.75, 0.75},    // Grey
	{ 0.05, 0.05, 0.05},    // Black
	{ 0.90, 0.90, 0.90}     // White
};

float Colors::HighlightColorTable[][3] = {
    { 1.00, 0.42, 0.42},    // Indian Red 2
    { 0.33, 1.00, 0.62},    // Sea Green 1
    { 0.42, 0.65, 0.83},    // Sky Blue 3
    { 1.00, 0.96, 0.56},    // Khaki 1
    { 1.00, 0.71, 0.76},    // Light Pink
    { 1.00, 0.93, 0.55},    // Light Golden Rod 2
    { 0.75, 0.94, 1.00},    // Light Blue 1
    { 0.95, 0.95, 0.95},    // Grey
    { 0.10, 0.10, 0.10},    // Black
    { 0.95, 0.95, 0.95}     // White
};

void Colors::getColor(float* c, Color name, ColorTable table)
{
	switch(table)
	{
	default:
	case BASIC:
		c[0] = PrimeColorTable[name][0];
		c[1] = PrimeColorTable[name][1];
		c[2] = PrimeColorTable[name][2];
		break;
	case LIGHT:
		c[0] = LightColorTable[name][0];
		c[1] = LightColorTable[name][1];
		c[2] = LightColorTable[name][2];
		break;
	case HIGHLIGHT:
		c[0] = HighlightColorTable[name][0];
		c[1] = HighlightColorTable[name][1];
		c[2] = HighlightColorTable[name][2];
		break;
	}
}
