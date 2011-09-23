/*
 * AsciiIO.cpp
 *
 *  Created on: 09.03.2011
 *      Author: Thomas Wiemann
 */

#include <fstream>
#include <string.h>
using std::ifstream;

#include <boost/filesystem.hpp>

#include "AsciiIO.hpp"
#include "Progress.hpp"
#include "Timestamp.hpp"

namespace lssr
{

void AsciiIO::read(string filename) {

	// Check extension
	boost::filesystem::path selectedFile(filename);
	string extension(selectedFile.extension().c_str());

	if ( extension != ".pts" && extension != ".3d" && extension != ".xyz" && extension != ".txt" ) {
		cout << "»" << extension << "« is not a valid file extension." << endl;
		return;
	}
	// Count lines in file to estimate the number of present points
	int lines_in_file = countLines(filename);

	if ( lines_in_file < 2 ) {
		cout << timestamp << "AsciiIO: Too few lines in file (has to be > 2)." << endl;
		return;
	}
	// Open the given file. Skip the first line (as it may
	// contain meta data in some formats). Then try to guess
	// the additional data using some heuriscs that apply for
	// most data formats: If 4 values per point are, given
	// the 4th value usually is a reflectence information.
	// Six entries suggest RGB information, seven entries
	// intensity and RGB.

	// Open file
	ifstream in;
	in.open(filename.c_str());

	// Read first to lines, ignore the first one
	char buffer[2048];
	in.getline(buffer, 2048);
	in.getline(buffer, 2048);

	// Get number of entries in test line and analiuze
	int num_attributes  = AsciiIO::getEntriesInLine(filename) - 3;
	bool has_color      = (num_attributes == 3) || (num_attributes == 4) 
		|| (num_attributes == 5);
	bool has_intensity  = (num_attributes == 1) || (num_attributes == 4);
	bool has_accuracy   = num_attributes == 5;
	bool has_validcolor = num_attributes == 5;

	if ( has_color ) {
		cout << timestamp << "Reading color information." << endl;
	}

	if ( has_intensity ) {
		cout << timestamp << "Reading intensity information." << endl;
	}

	// Reopen file and read data
	in.close();
	in.open(filename.c_str());

	// Again skip first line
	in.getline(buffer, 2048);

	// Alloc memory for points
	m_num_points = lines_in_file - 1;
	m_points = new float[ m_num_points * 3 ];

	// Alloc buffer memory for additional attributes
	if ( has_color ) {
		m_point_colors = new uint8_t[ m_num_points * 3 ];
		m_num_point_colors = m_num_points;
	}

	if ( has_intensity ) {
		m_point_intensities = new float[ m_num_points ];
		m_num_point_intensities = m_num_points;
	}

	if ( has_accuracy ) {
		m_point_confidence = new float[ m_num_points ];
		m_num_point_confidence = m_num_points;
	}

	// Read data form file
	size_t c = 0;
	while (in.good() && c < m_num_points) {
		//cout << has_intensity << " " << has_color << endl;
		//cout << c << " " << m_colors << " " << m_numPoints << endl;
		float x, y, z, i, dummy, confidence;
		unsigned int r, g, b;

		// Read according to determined format
		if(has_intensity && has_color) {
			in >> x >> y >> z >> i >> r >> g >> b;
			m_point_intensities[c] = i;
			m_point_colors[ c * 3     ] = (uint8_t) r;
			m_point_colors[ c * 3 + 1 ] = (uint8_t) g;
			m_point_colors[ c * 3 + 2 ] = (uint8_t) b;

		} else if ( has_color && has_accuracy && has_validcolor ) {
			in >> x >> y >> z >> confidence >> dummy >> r >> g >> b;
			m_point_confidence[c] = confidence;
			m_point_colors[ c * 3     ] = (uint8_t) r;
			m_point_colors[ c * 3 + 1 ] = (uint8_t) g;
			m_point_colors[ c * 3 + 2 ] = (uint8_t) b;

		} else if (has_intensity) {
			in >> x >> y >> z >> i;
			m_point_intensities[c] = i;

		} else if(has_color) {
			in >> x >> y >> z >> r >> g >> b;
			m_point_colors[ c * 3     ] = (uint8_t) r;
			m_point_colors[ c * 3 + 1 ] = (uint8_t) g;
			m_point_colors[ c * 3 + 2 ] = (uint8_t) b;

		} else {
			in >> x >> y >> z;
			for(int n_dummys = 0; n_dummys < num_attributes; n_dummys++) in >> dummy;
		}
		m_points[ c * 3     ] = x;
		m_points[ c * 3 + 1 ] = y;
		m_points[ c * 3 + 2 ] = z;
		c++;
	}

}


size_t AsciiIO::countLines(string filename)
{
	// Open file for reading
	ifstream in(filename.c_str());

	// Count lines in file
	size_t c = 0;
	char line[2048];
	while(in.good())
	{
		in.getline(line, 1024);
		c++;
	}
	in.close();
	return c;
}


int AsciiIO::getEntriesInLine(string filename)
{

	ifstream in(filename.c_str());

	// Get first line from file and skip it (possibly metadata)
	char first_line[1024];
	in.getline(first_line, 1024);

	// Get second line -> hopefully point data
	char second_line[1024];
	in.getline(second_line, 1024);

	in.close();

	// Get number of blanks
	int c = 0;
	char* pch = strtok(second_line, " ");
	while(pch != NULL){
		c++;
		pch = strtok(NULL, " ");
	}

	in.close();

	return c;
}


} // namespace lssr
