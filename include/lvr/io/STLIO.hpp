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

class STLIO : public BaseIO
{
public:
	STLIO();
	virtual ~STLIO();

	virtual ModelPtr read(string filename);
	virtual void save( string filename );
	virtual void save( ModelPtr model, string filename );


};

} /* namespace lvr */

#endif /* INCLUDE_LVR_IO_STLIO_HPP_ */
