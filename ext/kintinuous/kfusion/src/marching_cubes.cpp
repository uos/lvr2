#include <kfusion/marching_cubes.hpp>


using namespace lvr;



namespace kfusion
{
	MaCuWrapper::MaCuWrapper(double camera_target_distance, double voxel_size) : slice_count_(0), camera_target_distance_(camera_target_distance), voxel_size_(voxel_size)
	{
			omp_set_num_threads(omp_get_num_procs());
			//meshPtr_ = new HMesh();
			//last_grid_ = NULL;
			//grid_ptr_ = NULL;
			mcthread_ = NULL;
			bbox_ = BoundingBox<cVertex>(0.0, 0.0, 0.0, 300.0, 300.0, 300.0);
			bbox_.expand(300.0, 300.0, 300.0);
			voxel_size_ = 3.0 / 512.0;
			float max_size = bbox_.getLongestSide();
			//Save needed grid parameters
			maxIndex_ = (int)ceil( (max_size + 5 * voxel_size_) / voxel_size_);
			maxIndexSquare_ = maxIndex_ * maxIndex_;
	}

	void MaCuWrapper::resetMesh()
	{
		/*	if(last_grid_ != NULL)
		{
			delete last_grid_;
			last_grid_ = NULL;
		}
		delete grid_ptr_;*/
		//delete meshPtr_;
		//meshPtr_ = new HalfEdgeMesh<ColorVertex<float, unsigned char> , lvr::Normal<float> >();
	}

	void MaCuWrapper::createGrid(cv::Mat& cloud_host,  Vec3i offset, const bool last_shift)
	{
		ScopeTime* grid_time = new ScopeTime("Grid Creation");
		timestamp.setQuiet(true);
		Point* tsdf_ptr = cloud_host.ptr<Point>();				
		TGrid* act_grid = NULL;
		if(last_grid_queue_.size() == 0)
			act_grid = new TGrid(voxel_size_, bbox_, tsdf_ptr, cloud_host.cols, offset[0], offset[1], offset[2], NULL, true);
		else
			act_grid = new TGrid(voxel_size_, bbox_, tsdf_ptr, cloud_host.cols, offset[0], offset[1], offset[2], last_grid_queue_.front(), true);
		grid_queue_.push(act_grid);
		std::cout << "    ####     1 Finished grid number: " << slice_count_ << "   ####" << std::endl;
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
		if(last_grid_queue_.size() > 0)
		{
			delete last_grid_queue_.front();
			last_grid_queue_.pop();
		}
		last_grid_queue_.push(act_grid);
		/*if(!last_shift)
			mcthread_ = new std::thread(&kfusion::MaCuWrapper::createMeshSlice, this , last_shift);
		else*/
			createMeshSlice(last_shift);
	}
	
	void MaCuWrapper::createMeshSlice(const bool last_shift)
	{
		unordered_map<size_t, size_t> verts_map;
		MeshPtr meshPtr = new HMesh();
		TGrid* act_grid = grid_queue_.front();
		MeshPtr oldMesh = last_mesh_queue_.front();
		ScopeTime* cube_time = new ScopeTime("Marching Cubes");
		cFastReconstruction* fast_recon =  new cFastReconstruction(act_grid);
		
		// Create an empty mesh
		fast_recon->getMesh(*meshPtr);
		meshPtr->m_fusionBoxes = act_grid->getFusionCells();
		meshPtr->m_oldfusionBoxes = act_grid->m_old_fusion_cells;
		meshPtr->m_fusionNeighborBoxes = act_grid->m_fusion_cells_neighbors;
		// mark all fusion vertices in the mesh
		for(auto cellPair : act_grid->getFusionCells())
		{
			cFastBox* box = cellPair.second;
			for( int i = 0; i < 12; i++)
			{
				uint inter = box->m_intersections[i];
				if(inter != cFastBox::INVALID_INDEX)
				{
					meshPtr->setFusionVertex(inter);
				}
			}
		}
		/*for(auto cellPair : meshPtr->m_fusionNeighborBoxes)
		{
			cFastBox* box = cellPair.second;
			for( int edge_index = 0; edge_index < 12; edge_index++)
			{
				uint inter = box->m_intersections[edge_index];
				if(inter != cFastBox::INVALID_INDEX)
				{
						meshPtr->setOldFusionVertex(inter);
				}
			}		
		}*/
		cout << "actual mesh size " << meshPtr->meshSize() << endl;
		for(auto cellPair : act_grid->m_old_fusion_cells)
		{
			cFastBox* box = cellPair.second;
			for( int edge_index = 0; edge_index < 12; edge_index++)
			{
				uint inter = box->m_intersections[edge_index];
				uint inter2  = -1;
				if(inter != cFastBox::INVALID_INDEX)
				{
					for(int i = 0; i < 3; i++)
					{
						auto current_neighbor = box->m_neighbors[neighbor_table[edge_index][i]];
						if(current_neighbor != 0)
						{
							uint in2 = current_neighbor->m_intersections[neighbor_vertex_table[edge_index][i]];
							auto vert_it = verts_map.find(in2);
							if(vert_it == verts_map.end() && in2 != cFastBox::INVALID_INDEX && in2 != 0 && in2 != inter && current_neighbor->m_fusionNeighborBox)
							{
								inter2 = in2;
								
								verts_map.insert(pair<size_t, size_t>(inter, inter2));
								//cout << "inter1 " << inter << " inter2 " << inter2 << " verts from inter2 " << endl;
								//cout << "vert " << meshPtr->getVertices()[inter2]->m_position << endl; 
								//cout << "vert " << oldMesh->getVertices()[inter]->m_position << endl; 
								break;
							}
						}
					}
				}
				
			}
		}
		meshPtr->m_fusion_verts = verts_map;
		if(slice_count_ == 0)
			meshPtr_ = meshPtr;
		else
		{
			mesh_queue_.push(meshPtr);
		}
		grid_queue_.pop();
		//delete act_grid;
		if(last_mesh_queue_.size() > 0)
		{
			//delete last_mesh_queue_.front();
			last_mesh_queue_.pop();
		}
		last_mesh_queue_.push(meshPtr);
		delete cube_time;
		delete fast_recon;
		//std::cout << "Processed one tsdf value in " << recon_factor << "ns " << std::endl;
		std::cout << "        ####     2 Finished reconstruction number: " << slice_count_ << "   ####" << std::endl;
		optimizeMeshSlice(last_shift);
		//grid_ptr_ = NULL;
	}
	
	void MaCuWrapper::optimizeMeshSlice(const bool last_shift)
	{
		MeshPtr act_mesh = NULL;
		if(slice_count_ == 0)
			act_mesh = meshPtr_;
		else
			act_mesh = mesh_queue_.front();
		//cout << "size " << mesh_queue_.size() << endl;
		//act_mesh->optimizeIterativePlanes(3, 0.85, 7, 10);
		MeshPtr tmp_pointer = NULL;
		//tmp_pointer = act_mesh->retesselateInHalfEdge();
		
		if(slice_count_ > 0)
		{
			//opti_mesh_queue_.push(tmp_pointer);
			opti_mesh_queue_.push(act_mesh);
			mesh_queue_.pop();
			//delete act_mesh;
		}
		// update neighborhood
		/*for(auto cellPair : tmp_pointer->m_fusionBoxes())
		{
			cFastBox* box = cellPair.second;
			for( int i = 0; i < 12; i++)
			{
				uint inter = box->m_intersections[i];
				if(inter != cFastBox::INVALID_INDEX)
				{
					int new_ind = -1;
					try{
						new_ind = act_mesh_->m_slice_verts.at(inter);
						box->m_intersections[i] = new_ind;
					}catch(...){cout << "failed to map vertice index from old slice " << endl;}
					tmp_pointer->setFusionVertex(new_ind);
				}
			}
		}*/
		
		std::cout << "            ####     3 Finished optimisation number: " << slice_count_ << "   ####" << std::endl;
		if(slice_count_ > 0)
			fuseMeshSlice(last_shift);
		else
			slice_count_++;
		
	}
	
	void MaCuWrapper::fuseMeshSlice(const bool last_shift)
	{
		
		MeshPtr opti_mesh = opti_mesh_queue_.front();
		/*for(auto vert_it = opti_mesh->m_fusion_verts.begin(); it != opti_mesh->m_fusion_verts.end(); vert_it++)
		{
			vert_it->first += meshPtr_->m_old_size;
		}*/
		/*for(auto cellPair : opti_mesh->m_oldfusionBoxes)
		{
			cFastBox* box = cellPair.second;
			size_t index_x = box->m_center.x;
			size_t index_y = box->m_center.y;
			size_t index_z = box->m_center.z;
			for(int a = -1; a < 2; a++)
			{
				for(int b = -1; b < 2; b++)
				{
					for(int c = -1; c < 2; c++)
					{
						
						//Calculate hash value for current neighbor cell
						size_t neighbor_hash = this->hashValue(index_x + a,
								index_y + b,
								index_z + c);
						
						//Try to find this cell in the grid
						auto neighbor_it = opti_mesh->m_fusionNeighborBoxes.find(neighbor_hash);

						//If it exists, save pointer in box
						if(neighbor_it != opti_mesh->m_fusionNeighborBoxes.end())
						{
							
							cout << "neighbor found " << endl;
						}
					}
				}
			}
		}*/	
		meshPtr_->addMesh(opti_mesh, opti_mesh->m_slice_verts);
		opti_mesh_queue_.pop();
		//delete opti_mesh;
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
			ModelFactory::saveModel( m, "./mesh_" + to_string(slice_count_) + ".ply");
			//ModelFactory::saveModel( m, "./test_mesh.ply");
		}
		std::cout << "                        ####    4 Finished slice number: " << slice_count_ << "   ####" << std::endl;
		slice_count_++;
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
