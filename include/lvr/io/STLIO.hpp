/*
 * STLIO.hpp
 *
 *  Created on: Dec 9, 2016
 *      Author: robot
 */

#ifndef INCLUDE_LVR_IO_STLIO_HPP_
#define INCLUDE_LVR_IO_STLIO_HPP_

#include "BaseIO.hpp"

namespace lvr {

/****
 * @brief 	Reader / Writer for STL file. Currently only binary STL files
 * 			are supported.
 */
class STLIO : public BaseIO
{
public:
	STLIO();
	virtual ~STLIO();

	virtual void save( string filename );
	virtual void save( ModelPtr model, string filename );
    /**
     * @brief Parse the given file and load supported elements.
     *
     * @param 	filename  The file to read.
     * @return	A new model. If the file could not be parsed, an empty model
     * 			is returned.
     */
    virtual ModelPtr read(string filename );

};

} /* namespace lvr */

#endif /* INCLUDE_LVR_IO_STLIO_HPP_ */
