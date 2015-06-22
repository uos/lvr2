#include <kfusion/marching_cubes.hpp>


using namespace lvr;



namespace kfusion
{
        MaCuWrapper::MaCuWrapper() : slice_count_(0) 
        {
				omp_set_num_threads(omp_get_num_procs());
				meshPtr_ = new HalfEdgeMesh<ColorVertex<float, unsigned char> , lvr::Normal<float> >();
				last_grid_ = NULL;
		}
		
		void MaCuWrapper::resetMesh()
		{
			if(last_grid_ != NULL)
			{
				delete last_grid_;
				last_grid_ = NULL;
			}
			delete meshPtr_;
			meshPtr_ = new HalfEdgeMesh<ColorVertex<float, unsigned char> , lvr::Normal<float> >();
        }
        
        void MaCuWrapper::createMeshSlice(cv::Mat& cloud_host,  Vec3i offset, const bool last_shift)
        {
			ScopeTime* cube_time = new ScopeTime("Marching cubes");
			timestamp.setQuiet(true);
			Point* tsdf_ptr = cloud_host.ptr<Point>();				
			BoundingBox<cVertex> bbox(0.0, 0.0, 0.0, 300.0, 300.0, 300.0);
			bbox.expand(300.0, 300.0, 300.0);
			float voxelsize = 3.0 / 512.0;
			TGrid* grid_ptr = new TGrid(voxelsize, bbox, tsdf_ptr, cloud_host.cols, offset[0], offset[1], offset[2],last_grid_, true);
			
			//grid_ptr->saveGrid("./slices/grid" + std::to_string(slice_count_) + ".grid");
			cFastReconstruction* fast_recon =  new cFastReconstruction(grid_ptr);
			// Create an empty mesh
		    fast_recon->getMesh(*meshPtr_);
			if(last_shift)
			{
				// plane_iterations, normal_threshold, min_plan_size, small_region_threshold
				//meshPtr_->optimizePlanes(3, 0.85, 7, 10, true);
				//meshPtr_->fillHoles(30);
				//meshPtr_->optimizePlaneIntersections();
				//min_plan_size
				//meshPtr_->restorePlanes(7);
				//meshPtr_->finalizeAndRetesselate(false, 0.01);
				meshPtr_->finalize();
				ModelPtr m( new Model( meshPtr_->meshBuffer() ) );
				ModelFactory::saveModel( m, "./slices/mesh_" + to_string(slice_count_) + ".ply");
			}
			//cout << "Global cell count: " << grid_ptr->m_global_cells.size() << endl;
			delete fast_recon;
			if(last_grid_ != NULL)
				delete last_grid_;
			last_grid_ = grid_ptr;
			double recon_factor = (cube_time->getTime()/cloud_host.cols) * 1000;
			delete cube_time;
			//std::cout << "Processed one tsdf value in " << recon_factor << "ns " << std::endl;
			std::cout << "####    Finished slice number: " << slice_count_ << "   ####" << std::endl;
			timeStats_.push_back(recon_factor);
			slice_count_++;
		}
		
		double MaCuWrapper::calcTimeStats()
		{			
			return std::accumulate(timeStats_.begin(), timeStats_.end(), 0.0) / timeStats_.size();
		}
        
}
