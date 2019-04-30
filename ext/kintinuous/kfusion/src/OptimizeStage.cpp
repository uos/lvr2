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
 * OptimizeStage.cpp
 *
 *  @date 13.11.2015
 *  @author Tristan Igelbrink (Tristan@Igelbrink.com)
 */

#include <kfusion/OptimizeStage.hpp>

// default constructor
OptimizeStage::OptimizeStage(Options* options) : AbstractStage()
	,mesh_count_(0), options_(options), bounding_counter(0),texture_counter(0),pic_count_(0)
{
	 optiMesh_ = NULL;
	timestamp.setQuiet(!options_->verbose());
}

void OptimizeStage::firstStep() { }

void OptimizeStage::step()
{
	auto mesh_work = boost::any_cast<pair<pair<MeshPtr, bool>, vector<kfusion::ImgPose*> > >(getInQueue()->Take());
	bool last_shift = mesh_work.first.second;
	MeshPtr act_mesh = mesh_work.first.first;
	string mesh_notice = ("#### C:            Mesh Optimization " +  to_string(mesh_count_) + "    ####");
	ScopeTime* opti_time = new ScopeTime(mesh_notice.c_str());
	act_mesh->fillHoles(options_->getFillHoles());
	if(optiMesh_ == NULL)
		optiMesh_ = act_mesh;
	else
		optiMesh_->addMesh(act_mesh, options_->textures());
    std::vector<kfusion::ImgPose*> image_poses_buffer = mesh_work.second;
	//std::cout << "Loaded " << image_poses_buffer.size() << " Images. " << std::endl;
	// Set recursion depth for region growing
	if(options_->getDepth())
	{
		optiMesh_->setDepth(options_->getDepth());
	}
	optiMesh_->setClassifier(options_->getClassifier());
	optiMesh_->optimizePlanes(options_->getPlaneIterations(),
					options_->getNormalThreshold(),
					options_->getMinPlaneSize(),
					options_->getSmallRegionThreshold(), false);
	MeshPtr tmp_pointer = optiMesh_->retesselateInHalfEdge(options_->getLineFusionThreshold(), options_->textures(), texture_counter);
	if(tmp_pointer == NULL)
		return;
	optiMesh_->restorePlanes(options_->getMinPlaneSize());
	delete opti_time;
	mesh_count_++;

    ///texturing
	if(options_->textures())
	{
		int counter=0;
		int i;
		int progress = 0, j;


		for(i=0;i<image_poses_buffer.size();i++){
			counter = tmp_pointer->projectAndMapNewImage(*(image_poses_buffer[i]));
		}

		pic_count_+=i;

		texture_counter += tmp_pointer->textures.size();

		meshBufferPtr = tmp_pointer->meshBuffer();

	}
	image_poses_buffer.resize(0);
	getOutQueue()->Add(pair<pair<MeshPtr, bool>, vector<ImgPose*> >(
				pair<MeshPtr, bool>(tmp_pointer, last_shift), mesh_work.second));
	if(last_shift)
		done(true);
}
void OptimizeStage::lastStep()
{
	 delete optiMesh_;
}
