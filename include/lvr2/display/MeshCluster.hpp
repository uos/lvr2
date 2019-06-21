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
 * MeshCluster.hpp
 *
 *  Created on: 12.04.2012
 *      Author: Thomas Wiemann
 */

#ifndef MESHCLUSTER_HPP_
#define MESHCLUSTER_HPP_

#include <list>
using namespace std;

#include "lvr2/display/Renderable.hpp"
#include "lvr2/display/StaticMesh.hpp"

namespace lvr2
{

class MeshCluster : public Renderable
{
public:
	MeshCluster() {};

	void addMesh(MeshBufferPtr buffer, string name);

	virtual ~MeshCluster() { for (StaticMesh* sm : m_meshes) delete sm;};

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

} // namespace lvr2

#endif /* MESHCLUSTER_HPP_ */
