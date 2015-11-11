#include <kfusion/MeshStage.hpp>
#include <registration/ICPPointAlign.hpp>
#include <io/DataStruct.hpp>

// default constructor
MeshStage::MeshStage(double camera_target_distance, double voxel_size, Options* options) : AbstractStage(),
					camera_target_distance_(camera_target_distance), voxel_size_(voxel_size), options_(options), fusion_count_(0)
{
	mesh_count_ = 0;
	timestamp.setQuiet(!options->verbose());
}

void MeshStage::firstStep() { /* skip */ };

void MeshStage::step()
{
	auto grid_work = boost::any_cast<pair<pair<TGrid*, bool>, vector<ImgPose*> > >(getInQueue()->Take());
	TGrid* act_grid = grid_work.first.first;
	bool last_shift = grid_work.first.second;
	MeshPtr meshPtr = new HMesh();
	string mesh_notice = ("#### B:        Mesh Creation " +  to_string(mesh_count_) + "    ####");
	ScopeTime* cube_time = new ScopeTime(mesh_notice.c_str());

	cFastReconstruction* fast_recon =  new cFastReconstruction(act_grid);
	timestamp.setQuiet(!options_->verbose());
	// Create an empty mesh
	fast_recon->getMesh(*meshPtr);
	transformMeshBack(meshPtr);
	if(meshPtr->meshSize() == 0)
		return;
	unordered_map<HMesh::VertexPtr, HMesh::VertexPtr> verts_map;
	size_t misscount = 0;
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
						HMesh::VertexPtr old_vert = last_mesh_queue_.front()->getVertices()[inter];
						auto vert_it = verts_map.find(old_vert);
						if(vert_it == verts_map.end() && in2 != cFastBox::INVALID_INDEX && in2 != 0 && in2 != inter && current_neighbor->m_fusionNeighborBox)
						{
							inter2 = in2;
							HMesh::VertexPtr act_vert = meshPtr->getVertices()[inter2];
							verts_map.insert(pair<HMesh::VertexPtr, HMesh::VertexPtr>(old_vert, act_vert));
							meshPtr->setOldFusionVertex(inter2);
							if(act_vert->m_position[0] != old_vert->m_position[0] ||  act_vert->m_position[1] != old_vert->m_position[1]
								 || act_vert->m_position[2] != old_vert->m_position[2])
							{
								//cout << "missalignment" << endl;
								/*float dist_x = pos1.x - pos2.x;
								float dist_y = pos1.y - pos2.y;
								float dist_z = pos1.z - pos2.z;
								cout << "pos1 " << pos1 << " pos 2 " << pos2 <<endl;
								cout << "dist_x " << dist_x << endl;
								cout << "dist_y " << dist_y << endl;
								cout << "dist_z " << dist_z << endl;*/
								misscount++;
							}
							break;
						}
					}
				}
			}
		}
	}
	if(last_mesh_queue_.size() > 0)
	{
		auto m = last_mesh_queue_.front();
		cout << "founded fusion verts from old slice " << m->m_fusionVertices.size() << endl;
		cout << "founded fusion verts from new slice " << meshPtr->m_oldFusionVertices.size() << endl;
		if(verts_map.size() > 0)
		{
			cout << "merged fusion verts " << (double)verts_map.size()/m->m_fusionVertices.size() << endl;
			cout << "misscount quote " << (double)(misscount/verts_map.size()) << endl;
			if(((double)verts_map.size()/m->m_fusionVertices.size() < 0.5) || ((double)misscount/verts_map.size() > 0.9) )
			{
				/*cout << "SLICE CORRECTION " << endl;
				float euler[6];
				PointBufferPtr buffer(new PointBuffer());
				PointBufferPtr dataBuffer(new PointBuffer());
				floatArr vertexBuffer( new float[3 * m->m_fusionVertices.size()] );
				floatArr dataVertexBuffer( new float[3 * meshPtr->m_oldFusionVertices.size()] );
				for(size_t i = 0; i < m->m_fusionVertices.size()*3; i+=3)
				{
					vertexBuffer[i] = -m->m_fusionVertices[i/3]->m_position.x * 100;
					vertexBuffer[i + 1] = -m->m_fusionVertices[i/3]->m_position.y * 100;
					vertexBuffer[i + 2] = m->m_fusionVertices[i/3]->m_position.z * 100;
				}
				for(size_t i = 0; i < meshPtr->m_oldFusionVertices.size()*3; i+=3)
				{
					dataVertexBuffer[i] = -meshPtr->m_oldFusionVertices[i/3]->m_position.x * 100;
					dataVertexBuffer[i + 1] = -meshPtr->m_oldFusionVertices[i/3]->m_position.y * 100;
					dataVertexBuffer[i + 2] = meshPtr->m_oldFusionVertices[i/3]->m_position.z * 100;
				}
				buffer->setPointArray(vertexBuffer, m->m_fusionVertices.size());
				dataBuffer->setPointArray(dataVertexBuffer, meshPtr->m_oldFusionVertices.size());
				Vertexf position(0, 0, 0);
				Vertexf angle(0, 0, 0);
				Matrix4f transformation(position, angle);

				ICPPointAlign align(buffer, dataBuffer, transformation);
				align.setMaxIterations(20);
				align.setMaxMatchDistance(0.8);
				Matrix4f correction = align.match();
				correction.set(12, correction[12]/100.0);
				correction.set(13, correction[13]/100.0);
				correction.set(14, -correction[14]/100.0);
				correction.toPostionAngle(euler);
				double correction_value = sqrt(pow(correction[12],2) + pow(correction[13],2) + pow(correction[14],2));
				cout << "correction_value " << correction_value << endl;
				if(correction_value < 0.08)
				{
					cout << "Applieng ICP Pose " << endl;
					cout << "Pose: " << correction[12] << " " << correction[13] << " " << correction[14] << " " << euler[3] << " " << euler[4] << " " << euler[5] << endl;
					for(auto vert : meshPtr->getVertices())
					{
						vert->m_position.transform(correction);
					}
					PointPairVector pairs;
					double sum;
					Vertexf centroid_m;
					Vertexf centroid_d;
					align.getPointPairs(pairs, centroid_m, centroid_d, sum);
					//verts_map.clear();

					for(std::pair<Vertexf, Vertexf> vpair : pairs)
					{
						cout << "second entry " << vpair.second << endl;
						map<double, int> oldVertMap;
						for(int i = 0; i < meshPtr->m_oldFusionVertices.size(); i++)
						{
							double diff_sum =    (pow(vpair.second[0],2) - (-pow(meshPtr->m_oldFusionVertices[i]->m_position.x * 100,2)))
							 			+ (pow(vpair.second[1],2) - (-pow(meshPtr->m_oldFusionVertices[i]->m_position.y * 100,2)))
							 			+ (pow(vpair.second[2],2) - pow(meshPtr->m_oldFusionVertices[i]->m_position.z * 100,2));
							oldVertMap.insert(pair<double, int>(diff_sum, i));
						}
						cout << "best match old verts " << oldVertMap.begin()->first << " " <<  meshPtr->m_oldFusionVertices[oldVertMap.begin()->second]->m_position << endl;
					}

					global_correction_ *= correction;
				}*/

			}
		}



		//cout << "merged quote " << verts_map.size() << endl;
		//cout << "missaligned verts " << misscount << endl;
		meshPtr->m_fusion_verts = verts_map;
	}
	if(last_mesh_queue_.size() > 0)
	{
		//delete last_grid_queue_.front();
		last_mesh_queue_.pop();
	}
	last_mesh_queue_.push(meshPtr);

	mesh_count_++;

	delete cube_time;
	delete fast_recon;
	getOutQueue()->Add(pair<pair<MeshPtr, bool>, vector<ImgPose*> >(
				pair<MeshPtr, bool>(meshPtr, last_shift), grid_work.second));
	if(last_shift)
		done(true);
}

void MeshStage::lastStep()	{ /* skip */ }


void MeshStage::transformMeshBack(MeshPtr mesh)
{
	cout << "global correction " << global_correction_ << endl;
	for(auto vert : mesh->getVertices())
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
		vert->m_position.transform(global_correction_);
	}
}
