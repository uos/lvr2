#include <kfusion/LVRPipeline.hpp>

using namespace lvr;

namespace kfusion
{
	LVRPipeline::LVRPipeline(double camera_target_distance, double voxel_size) : slice_count_(0)
	{
		omp_set_num_threads(omp_get_num_procs());
		pl_.AddStage(
			boost::shared_ptr<GridStage>(new GridStage(voxel_size))
			);
		pl_.AddStage(
			boost::shared_ptr<MeshStage>(new MeshStage())
			);
		pl_.AddStage(
			boost::shared_ptr<OptimizeStage>(new OptimizeStage())
			);
		pl_.AddStage(
			boost::shared_ptr<FusionStage>(new FusionStage(new HMesh(), camera_target_distance_, voxel_size))
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
	//	std::vector<int> result = pl.GetResult();
	//	std::cout << result[0] << std::endl;

	double LVRPipeline::calcTimeStats()
	{			
		return std::accumulate(timeStats_.begin(), timeStats_.end(), 0.0) / timeStats_.size();
	}
        
}
