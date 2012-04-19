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
 * MeshCluster.hpp
 *
 *  Created on: 12.04.2012
 *      Author: Thomas Wiemann
 */

#ifndef MESHCLUSTER_HPP_
#define MESHCLUSTER_HPP_

#include <list>
using namespace std;

#include "Renderable.hpp"
#include "StaticMesh.hpp"

namespace lssr
{

class MeshCluster: public lssr::Renderable
{
public:
	MeshCluster() {};

	void addMesh(MeshBufferPtr buffer, string name);

	virtual ~MeshCluster() {};

	virtual inline void render()
	{
		list<StaticMesh*>::iterator it;
		for(it = m_meshes.begin(); it != m_meshes.end(); it++)
		{
			(*it)->render();
		}
	}

	list<StaticMesh*> getMeshes() { return m_meshes;}

private:
	list<StaticMesh*> m_meshes;
};

} // namespace lssr

#endif /* MESHCLUSTER_HPP_ */
