#include <kfusion/LVRPipeline.hpp>

using namespace lvr;

namespace kfusion
{
	LVRPipeline::LVRPipeline(double camera_target_distance, double voxel_size, bool optimize, string mesh_name) : slice_count_(0)
	{
		meshPtr_ = new HMesh();
		cout << "cam target dist " << camera_target_distance << endl;
		omp_set_num_threads(omp_get_num_procs());
		pl_.AddStage(
			boost::shared_ptr<GridStage>(new GridStage(voxel_size))
			);
		pl_.AddStage(
			boost::shared_ptr<MeshStage>(new MeshStage(camera_target_distance_, voxel_size))
			);
		if(optimize)
		{
			pl_.AddStage(
				boost::shared_ptr<OptimizeStage>(new OptimizeStage())
				);
		}
		pl_.AddStage(
			boost::shared_ptr<FusionStage>(new FusionStage(meshPtr_, mesh_name))
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
