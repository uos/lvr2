/*
 * PPMIO.cpp
 *
 *  Created on: 08.09.2011
 *      Author: pg2011
 */

namespace lssr
{

template<typename ColorType>
PPMIO<ColorType>::PPMIO()
{
    m_data    = 0;
    m_sizeX   = 0;
    m_sizeY   = 0;
}

template<typename ColorType>
void PPMIO<ColorType>::setDataArray(ColorType** array, size_t sizeX, size_t sizeY)
{
    m_data = array;
    m_sizeX = sizeX;
    m_sizeY = sizeY;
}


template<typename ColorType>
void PPMIO<ColorType>::write(string filename)
{
    ofstream out(filename.c_str());

    if(out.good())
    {
    	out<<"P6"<<" "<<m_sizeX<<" "<<m_sizeY<<" "<<"255"<<endl;
    	for(int y = 0; y<m_sizeY; y++)
    		for(int x = 0; x<m_sizeX; x++)
    			out<<m_data[y][x].r<<m_data[y][x].g<<m_data[y][x].b;
    }

    out.close();

}

}
