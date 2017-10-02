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
 * LVRPipeline.cpp
 *
 *  @date 13.11.2015
 *  @author Tristan Igelbrink (Tristan@Igelbrink.com)
 */

#include <kfusion/LVRPipeline.hpp>

using namespace lvr;

namespace kfusion
{
	LVRPipeline::LVRPipeline(KinFuParams params) : slice_count_(0)
	{
		meshPtr_ = new HMesh();
		meshPtr_->setQuiet(!params.cmd_options->verbose());
		omp_set_num_threads(omp_get_num_procs());

		// Adding the single processing stages to the pipeline
		pl_.AddStage(
			boost::shared_ptr<GridStage>(new GridStage((double)(params.volume_size[0] / params.volume_dims[0]), params.cmd_options))
			);
		pl_.AddStage(
			boost::shared_ptr<MeshStage>(new MeshStage(params.distance_camera_target, (double)(params.volume_size[0] / params.volume_dims[0]), params.cmd_options))
			);
		if(params.cmd_options->optimizePlanes())
		{
			pl_.AddStage(
				boost::shared_ptr<OptimizeStage>(new OptimizeStage(params.cmd_options))
				);
		}
		pl_.AddStage(
			boost::shared_ptr<FusionStage>(new FusionStage(meshPtr_, params.cmd_options))
			);

		pl_.Start();
	}

	LVRPipeline::~LVRPipeline()
	{
		pl_.join();
		delete meshPtr_;
	}

	void LVRPipeline::resetMesh()
	{
		//TODO implement mesh reset
	}

	void LVRPipeline::addTSDFSlice(TSDFSlice slice, const bool last_shift)
	{

		pair<TSDFSlice, bool> workload(slice, last_shift);
		pl_.AddWork(workload);
		slice_count_++;
	}

	double LVRPipeline::calcTimeStats()
	{
		return std::accumulate(timeStats_.begin(), timeStats_.end(), 0.0) / timeStats_.size();
	}

}
