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
 * GridStage.cpp
 *
 *  @date 13.11.2015
 *  @author Tristan Igelbrink (Tristan@Igelbrink.com)
 */

#include <kfusion/GridStage.hpp>

// default constructor
GridStage::GridStage(double voxel_size, Options* options) : AbstractStage()
{
	grid_count_ = 0;
	voxel_size_  = voxel_size;
	bbox_ = BoundingBox<cVertex>(0.0, 0.0, 0.0, 300.0, 300.0, 300.0);
	bbox_.expand(300.0, 300.0, 300.0);
	timestamp.setQuiet(!options->verbose());
}

void GridStage::firstStep() { /* omit */ };

void GridStage::step()
{
	auto slice_work = boost::any_cast<pair<TSDFSlice, bool> >(getInQueue()->Take());
	cv::Mat& tsdf_values = slice_work.first.tsdf_values_;
	Vec3i offset = slice_work.first.offset_;
	Vec3i back_offset = slice_work.first.back_offset_;
	bool last_shift = slice_work.second;
	Point* tsdf_ptr = tsdf_values.ptr<Point>();
	TGrid* act_grid = NULL;
	string grid_notice = ("#### A:    Grid Creation " +  to_string(grid_count_) + "    ####");
	ScopeTime* grid_time = new ScopeTime(grid_notice.c_str());
	if(last_grid_queue_.size() == 0)
		act_grid = new TGrid(voxel_size_, bbox_, tsdf_ptr, tsdf_values.cols, offset[0], offset[1], offset[2], back_offset[0], back_offset[1], back_offset[2], NULL, true);
	else
		act_grid = new TGrid(voxel_size_, bbox_, tsdf_ptr, tsdf_values.cols, offset[0], offset[1], offset[2], back_offset[0], back_offset[1], back_offset[2], last_grid_queue_.front(), true);

	//grid_ptr->saveGrid("./slices/grid" + std::to_string(slice_count_) + ".grid");
	delete grid_time;
	if(last_grid_queue_.size() > 0)
	{
		delete last_grid_queue_.front();
		last_grid_queue_.pop();
	}
	last_grid_queue_.push(act_grid);
	getOutQueue()->Add(pair<pair<TGrid*, bool>, vector<ImgPose*> >(pair<TGrid*, bool>(act_grid, last_shift), slice_work.first.imgposes_));
	grid_count_++;
	if(last_shift)
		done(true);
}
void GridStage::lastStep()	{ /* omit */ };
