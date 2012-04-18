/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

/*
 * SignalingMeshGenerator.hpp
 *
 *  Created on: 03.04.2012
 *      Author: Thomas Wiemann
 */

#ifndef SIGNALINGMESHGENERATOR_HPP_
#define SIGNALINGMESHGENERATOR_HPP_

#include <QtGui>
#include "io/PointBuffer.hpp"
#include "reconstruction/AdaptiveKSearchSurface.hpp"
#include "reconstruction/PCLKSurface.hpp"
#include "reconstruction/FastReconstruction.hpp"

using namespace lssr;

typedef AdaptiveKSearchSurface<cVertex, cNormal>        akSurface;
typedef PointsetSurface<cVertex>                        psSurface;

class SignalingMeshGenerator : public QThread
{
	Q_OBJECT
public:
	SignalingMeshGenerator();
	virtual ~SignalingMeshGenerator();

	virtual void run();

public Q_SLOTS:
	void newPointCloud(PointBufferPtr *buffer);

private:
	QMutex			m_mutex;
	PointBufferPtr	m_pointBuffer;

	bool			m_newData;
};

#endif /* SIGNALINGMESHGENERATOR_HPP_ */
