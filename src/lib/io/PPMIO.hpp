/*
 * PPMIO.h
 *
 *  Created on: 08.09.2011
 *      Author: pg2011
 */

#ifndef PPMIO_HPP_
#define PPMIO_HPP_

namespace lssr
{

/**
 * @brief An implementation of the PPM file format.
 */
template<typename ColorType>
class PPMIO
{
public:
    PPMIO();

    void write(string filename);
    void setDataArray(ColorType** array, size_t sizeX, size_t sizeY);

private:
    ColorType**              m_data;

    size_t                  m_sizeX;
    size_t                  m_sizeY;

};

}

#include "PPMIO.tcc"

#endif /* PPMIO_H_ */
