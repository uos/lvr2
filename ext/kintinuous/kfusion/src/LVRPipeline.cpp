#include <kfusion/LVRPipeline.hpp>

using namespace lvr;

namespace kfusion
{
	LVRPipeline::LVRPipeline(double camera_target_distance, double voxel_size) : slice_count_(0), camera_target_distance_(camera_target_distance), voxel_size_(voxel_size)
	{
		omp_set_num_threads(omp_get_num_procs());
		pl_.AddStage(
			boost::shared_ptr<GridStage>(new GridStage())
			);
		pl_.AddStage(
			boost::shared_ptr<MeshStage>(new MeshStage())
			);
		pl_.AddStage(
			boost::shared_ptr<OptimizeStage>(new OptimizeStage())
			);
		pl_.AddStage(
			boost::shared_ptr<FusionStage>(new FusionStage())
			);
		pl_.Start();
	}

	void LVRPipeline::resetMesh()
	{
		//TODO implement mesh reset
	}

	void addTSDFSlice(cv::Mat& cloud_host,  Vec3i offset, const bool last_shift)
	{
		
		pair<pair<cv::Mat&, Vec3i>, bool> workload(pair<cv::Mat&, Vec3i>(cloud_host, offset), last_shift);
		pl.AddWork(workload);
	}

	// extract the results
	for(size_t i=0; i<10; ++i)
	{
		std::vector<int> result = pl.GetResult();
		std::cout << result[0] << std::endl;
	}

	// wait for all the pipeline to complete
	// currently, the pipeline will never complete because the stages
	// don't know when to finish. An API can be implemented easily to end
	// the pipeline no matter what.
	pl.Join();
	
	void LVRPipeline::transformMeshBack()
	{
		for(auto vert : meshPtr_->getVertices())
		{
			// calc in voxel
			vert->m_position.x 	*= voxel_size_;				
			vert->m_position.y 	*= voxel_size_;				
			vert->m_position.z 	*= voxel_size_;			
			//offset for cube coord to center coord
			vert->m_position.x 	-= 1.5;				
			vert->m_position.y 	-= 1.5;				
			vert->m_position.z 	-= 1.5 - camera_target_distance_;				
			
			//offset for cube coord to center coord
			vert->m_position.x 	-= 150;				
			vert->m_position.y 	-= 150;				
			vert->m_position.z 	-= 150;
		}
	}


	double LVRPipeline::calcTimeStats()
	{			
		return std::accumulate(timeStats_.begin(), timeStats_.end(), 0.0) / timeStats_.size();
	}
        
}
