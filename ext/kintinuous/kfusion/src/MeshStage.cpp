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
							current_neighbor->m_fusionNeighborBox = false;
							//break;
						}
					}
				}
			}
		}
	}
	if(last_mesh_queue_.size() > 0)
	{
		auto m = last_mesh_queue_.front();
		//cout << "founded fusion verts from old slice " << m->m_fusionVertices.size() << endl;
		//cout << "founded fusion verts from new slice " << meshPtr->m_oldFusionVertices.size() << endl;
		if(verts_map.size() > 0)
		{
			//cout << "merged fusion verts " << (double)verts_map.size()/m->m_fusionVertices.size() << endl;
			//cout << "misscount quote " << (double)(misscount/verts_map.size()) << endl;
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
				Matrix4f trans;
				trans.set(12, -correction[12]/100.0);
				trans.set(13, -correction[13]/100.0);
				trans.set(14, correction[14]/100.0);
				cout << " X: " << trans[12] << " Y: " << trans[13] << " Z: " << trans[14] << endl;
				trans.toPostionAngle(euler);
				double correction_value = sqrt(pow(trans[12],2) + pow(trans[13],2) + pow(trans[14],2));
				cout << "correction_value " << correction_value << endl;
				if(correction_value < 0.08)
				{
					cout << "Applieng ICP Pose " << endl;
					cout << "Pose: " << correction[12] << " " << correction[13] << " " << correction[14] << " " << euler[3] << " " << euler[4] << " " << euler[5] << endl;
					for(auto vert : meshPtr->getVertices())
					{
						vert->m_position.transform(trans);
					}
					cout << " size " << m->m_fusionVertices.size() << endl;
					map<size_t, HMesh::VertexPtr> kdFusionVertsMap;
					cv::Mat data;
					data.create(cvSize(3,m->m_fusionVertices.size()), CV_32F); // The set A
					for(size_t i = 0; i < m->m_fusionVertices.size();i++)
					{
						data.at<float>(i,0) =  m->m_fusionVertices[i]->m_position.x;
						data.at<float>(i,1) =  m->m_fusionVertices[i]->m_position.y;
						data.at<float>(i,2) =  m->m_fusionVertices[i]->m_position.z;
						kdFusionVertsMap.insert(pair<size_t, HMesh::VertexPtr>(i,m->m_fusionVertices[i]));
					}
					cout << " size " << m->m_fusionVertices.size() << endl;
					map<size_t, HMesh::VertexPtr> kdOldFusionVertsMap;
					cv::Mat query;
					query.create(cvSize(3,meshPtr->m_oldFusionVertices.size()), CV_32F); // The set A
					for(size_t i = 0; i < meshPtr->m_oldFusionVertices.size();i++)
					{
						query.at<float>(i,0) =  meshPtr->m_oldFusionVertices[i]->m_position.x;
						query.at<float>(i,1) =  meshPtr->m_oldFusionVertices[i]->m_position.y;
						query.at<float>(i,2) =  meshPtr->m_oldFusionVertices[i]->m_position.z;
						kdOldFusionVertsMap.insert(pair<size_t, HMesh::VertexPtr>(i, meshPtr->m_oldFusionVertices[i]));
					}
					cout << " size " << m->m_fusionVertices.size() << endl;
					cv::Mat matches; //This mat will contain the index of nearest neighbour as returned by Kd-tree
					cv::Mat distances; //In this mat Kd-Tree return the distances for each nearest neighbour
					 //This set B
					const cvflann::SearchParams params(32); //How many leaves to search in a tree
					cv::flann::GenericIndex< cvflann::L2<float> > *kdtrees; // The flann searching tree

					// Create matrices
					matches.create(cvSize(1,meshPtr->m_oldFusionVertices.size()), CV_32SC1);
					distances.create(cvSize(1,meshPtr->m_oldFusionVertices.size()), CV_32FC1);
					kdtrees =  new cv::flann::GenericIndex< cvflann::L2<float> >(data, cvflann::KDTreeIndexParams(4)); // a 4 k-d tree
					cout << " size " << m->m_fusionVertices.size() << endl;
					// Search KdTree
					kdtrees->knnSearch(query, matches, distances, 1,  cvflann::SearchParams(8));
					int NN_index;
					float dist;
					verts_map.clear();
					//for(int i = 0; i < meshPtr->m_oldFusionVertices.size(); i++) {
					for(int i = 0; i < 10; i++) {

					    NN_index = matches.at<int>(i,0);
					    dist = distances.at<float>(i, 0);
						cout << " index " << NN_index << endl;
						cout << " dist "  << dist << endl;
						cout << "gift " << kdFusionVertsMap[NN_index] << endl;
						cout << "old gift " << kdOldFusionVertsMap[i] << endl;
						cout << " fusing  " << kdFusionVertsMap[NN_index]->m_position << endl;
						cout << " with  " << kdOldFusionVertsMap[i]->m_position << endl;

						verts_map.insert(pair<HMesh::VertexPtr, HMesh::VertexPtr>(kdFusionVertsMap[NN_index], kdOldFusionVertsMap[i]));
					}
					delete kdtrees;
					global_correction_ *= trans;
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
