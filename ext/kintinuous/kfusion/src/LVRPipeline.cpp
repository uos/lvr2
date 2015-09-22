#include <kfusion/LVRPipeline.hpp>

using namespace lvr;

namespace kfusion
{
	LVRPipeline::LVRPipeline(KinFuParams params) : slice_count_(0)
	{
		meshPtr_ = new HMesh();
		meshPtr_->setQuiet(!params.cmd_options->verbose());	
		omp_set_num_threads(omp_get_num_procs());
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
			boost::shared_ptr<FusionStage>(new FusionStage(meshPtr_, params.cmd_options->getOutput()))
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
