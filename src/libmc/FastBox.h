/*
 * FastBox.h
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

#ifndef FASTBOX_H_
#define FASTBOX_H_

#include "QueryPoint.h"

class FastBox {
public:
	FastBox();
	virtual ~FastBox();

private:
	int vertices     [8];
	int intersections[12];

};

#endif /* FASTBOX_H_ */
