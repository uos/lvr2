/*
 * Software License Agreement (BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */
/*
 * FusionStage.cpp
 *
 *  @date 13.11.2015
 *  @author Tristan Igelbrink (Tristan@Igelbrink.com)
 */

#include <kfusion/FusionStage.hpp>

// default constructor
FusionStage::FusionStage(MeshPtr mesh, Options* options) : AbstractStage()
	, mesh_(mesh), mesh_count_(0), options_(options)
{

}

void FusionStage::firstStep() { /* skip */ };

void FusionStage::step()
{
	auto mesh_work = boost::any_cast<pair<pair<MeshPtr, bool>, vector<kfusion::ImgPose*> > >(getInQueue()->Take());
	bool last_shift = mesh_work.first.second;
	MeshPtr opti_mesh = mesh_work.first.first;
	string mesh_notice = ("#### D:                Mesh Fusion " +  to_string(mesh_count_) + "    ####");
	ScopeTime* fusion_time = new ScopeTime(mesh_notice.c_str());
	if(mesh_count_ == 0)
		mesh_ = opti_mesh;
	else
		mesh_->addMesh(opti_mesh, options_->textures());
	//mesh_->fillHoles(options_->getFillHoles());
	//optiMesh_->restorePlanes(options_->getMinPlaneSize());
	getOutQueue()->Add(mesh_);
	mesh_count_++;
	delete fusion_time;
	if(last_shift)
		done(true);
}
void FusionStage::lastStep()
{
	std::cout << "Global amount of vertices: " << mesh_->meshSize() << endl;
	std::cout << "Global amount of faces: " << mesh_->getFaces().size() << endl;
	mesh_->finalize();
	ModelPtr m( new Model( mesh_->meshBuffer() ) );
	if(!options_->textures())
		ModelFactory::saveModel( m, string(options_->getOutput() + ".ply"));
	else
		ModelFactory::saveModel( m, string(options_->getOutput() + ".obj"));
	std::cout << "Finished saving" << std::endl;
}
