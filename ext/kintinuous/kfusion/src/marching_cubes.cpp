#include <kfusion/marching_cubes.hpp>


using namespace lvr;



namespace kfusion
{
	MaCuWrapper::MaCuWrapper(double camera_target_distance, double voxel_size) : slice_count_(0), camera_target_distance_(camera_target_distance), voxel_size_(voxel_size)
	{
			omp_set_num_threads(omp_get_num_procs());
			meshPtr_ = new HMesh();
			//last_grid_ = NULL;
			//grid_ptr_ = NULL;
			mcthread_ = NULL;
	}

	void MaCuWrapper::resetMesh()
	{
		/*	if(last_grid_ != NULL)
		{
			delete last_grid_;
			last_grid_ = NULL;
		}
		delete grid_ptr_;*/
		delete meshPtr_;
		meshPtr_ = new HalfEdgeMesh<ColorVertex<float, unsigned char> , lvr::Normal<float> >();
	}

	void MaCuWrapper::createGrid(cv::Mat& cloud_host,  Vec3i offset, const bool last_shift)
	{
		ScopeTime* grid_time = new ScopeTime("Grid Creation");
		timestamp.setQuiet(true);
		Point* tsdf_ptr = cloud_host.ptr<Point>();				
		BoundingBox<cVertex> bbox(0.0, 0.0, 0.0, 300.0, 300.0, 300.0);
		bbox.expand(300.0, 300.0, 300.0);
		float voxelsize = 3.0 / 512.0;
		TGrid* last_grid = NULL;
		TGrid* act_grid = NULL;
		if(last_grid_queue_.size() > 0)
		  last_grid = grid_queue_.front();
		act_grid = new TGrid(voxelsize, bbox, tsdf_ptr, cloud_host.cols, offset[0], offset[1], offset[2], last_grid, true);
		grid_queue_.push(act_grid);
		std::cout << "            ####     1 Finished grid number: " << slice_count_ << "   ####" << std::endl;
		//grid_ptr->saveGrid("./slices/grid" + std::to_string(slice_count_) + ".grid");
		double recon_factor = (grid_time->getTime()/cloud_host.cols) * 1000;
		timeStats_.push_back(recon_factor);
		if(mcthread_ != NULL)
		{
			mcthread_->join();
			delete mcthread_;
			mcthread_ = NULL;
		}
		delete grid_time;
		if(last_grid != NULL)
			delete last_grid;
		last_grid_queue_.pop();
		last_grid_queue_.push(act_grid);
		if(!last_shift)
			mcthread_ = new std::thread(&kfusion::MaCuWrapper::createMeshSlice, this , last_shift);
		else
			createMeshSlice(last_shift);
	}
	
	void MaCuWrapper::createMeshSlice(const bool last_shift)
	{
		TGrid* act_grid = grid_queue_.front();
		grid_queue_.pop();
		ScopeTime* cube_time = new ScopeTime("Marching Cubes");
		cFastReconstruction* fast_recon =  new cFastReconstruction(act_grid);
		// Create an empty mesh
		fast_recon->getMesh(*meshPtr_);
		// mark all fusion vertices in the mesh
		for(auto cellPair : act_grid->getFusionCells())
		{
			cFastBox* box = cellPair.second;
			for( int i = 0; i < 12; i++)
			{
				uint inter = box->m_intersections[i];
				if(inter != cFastBox::INVALID_INDEX)
					meshPtr_->setFusionVertex(inter);
			}
		}
		meshPtr_->optimizeIterativePlanes(3, 0.85, 7, 10);
		MeshPtr tmp_pointer = NULL;
		tmp_pointer = meshPtr_->retesselateInHalfEdge();
		// update neighborhood
		for(auto cellPair : act_grid->getFusionCells())
		{
			cFastBox* box = cellPair.second;
			for( int i = 0; i < 12; i++)
			{
				uint inter = box->m_intersections[i];
				if(inter != cFastBox::INVALID_INDEX)
				{
					int new_ind = -1;
					try{
						new_ind = meshPtr_->m_slice_verts.at(inter);
						box->m_intersections[i] = new_ind;
					}catch(...){cout << "failed to map vertice index from old slice " << endl;}
					tmp_pointer->setFusionVertex(new_ind);
				}
			}
		}
		delete meshPtr_;
		meshPtr_ = tmp_pointer;
		
		if(last_shift)
		{
			// plane_iterations, normal_threshold, min_plan_size, small_region_threshold
			//meshPtr_->optimizePlanes(3, 0.85, 7, 10, true);
			//meshPtr_->fillHoles(30);
			//meshPtr_->optimizePlaneIntersections();
			//min_plan_size
			//meshPtr_->restorePlanes(7);
			//meshPtr_->finalizeAndRetesselate(false, 0.01);
			transformMeshBack();
			meshPtr_->finalize();
			ModelPtr m( new Model( meshPtr_->meshBuffer() ) );
			ModelFactory::saveModel( m, "./slices/mesh_" + to_string(slice_count_) + ".ply");
			//ModelFactory::saveModel( m, "./test_mesh.ply");
		}
		//cout << "Global cell count: " << grid_ptr->m_global_cells.size() << endl;
		delete fast_recon;
		//last_grid_ = grid_ptr_;
		delete cube_time;
		//std::cout << "Processed one tsdf value in " << recon_factor << "ns " << std::endl;
		std::cout << "                        ####    2 Finished slice number: " << slice_count_ << "   ####" << std::endl;
		slice_count_++;
		//grid_ptr_ = NULL;
	}
	
	void MaCuWrapper::transformMeshBack()
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


	double MaCuWrapper::calcTimeStats()
	{			
		return std::accumulate(timeStats_.begin(), timeStats_.end(), 0.0) / timeStats_.size();
	}
        
}
