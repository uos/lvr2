/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
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


#include <kfusion/cyclical_buffer.h>
#include "precomp.hpp"

bool
kfusion::cuda::CyclicalBuffer::checkForShift (cv::Ptr<cuda::TsdfVolume> volume, const Affine3f &cam_pose, const double distance_camera_target, const bool perform_shift, const bool last_shift, const bool record_mode)
{
    bool result = false;
	
 	cv::Vec3f targetPoint(0,0, distance_camera_target);
 	targetPoint = cam_pose * targetPoint;
    targetPoint[1] = cam_pose.translation()[1];
	cv::Vec3f center_cube;
	center_cube[0] = buffer_.origin_metric.x + buffer_.volume_size.x/2.0f;
	center_cube[1] = buffer_.origin_metric.y + buffer_.volume_size.y/2.0f;
	center_cube[2] = buffer_.origin_metric.z + buffer_.volume_size.z/2.0f;

	double dist = norm(targetPoint, center_cube, cv::NORM_L2);
	//printf("   dist: %f   \n", dist);
	if (dist > distance_threshold_)
		result = true;

	// perform shifting operations
	if (result || last_shift || perform_shift)
	{
		// sync old marching cubes thread
		if(marching_thread_ != NULL)
		{
			cout << "####    Next shift incoming!    ####" << endl;
			marching_thread_->join();
			delete marching_thread_;
			marching_thread_ = NULL;
		}
		performShift (volume, targetPoint, cam_pose, last_shift, record_mode);
		return true;
	}

	return (result);
}


void
kfusion::cuda::CyclicalBuffer::performShift (cv::Ptr<cuda::TsdfVolume> volume, const cv::Vec3f& target_point, const Affine3f &cam_pose, const bool last_shift, const bool record_mode)
{
	std::cout << "####    Performing slice number: " << slice_count_ << "   ####" << std::endl;	
	//ScopeTime* time = new ScopeTime("Whole Cube shift");
	// compute new origin and offsets
	Vec3i offset;
	Vec3i minBounds;
	Vec3i maxBounds;

	computeAndSetNewCubeMetricOrigin (volume, target_point, offset);
	//ScopeTime* slice_time = new ScopeTime("Slice download");
	// extract current slice from the TSDF volume (coordinates are in indices! (see fetchSliceAsCloud() )
	if(!record_mode)
	{
		DeviceArray<Point> cloud; 
		
		// calculate mininum and maximum slice bounds
		if(last_shift)
		{
			minBounds[0] = minBounds[1] = minBounds[2] = 0;
			maxBounds[0] = maxBounds[1] = maxBounds[2] = 511;
		}
		else
		{
			calcBounds(offset, minBounds, maxBounds);
		}
		
		cloud = volume->fetchSliceAsCloud(cloud_buffer_device_, &buffer_, minBounds, maxBounds, global_shift_ );
		
		cloud_slice_ = cv::Mat(1, (int)cloud.size(), CV_32FC4);
		cloud.download(cloud_slice_.ptr<Point>());
		
		//delete slice_time;
		
		cout << "TSDF Values: " << cloud.size() << endl;
		Point* tsdf_ptr = cloud_slice_.ptr<Point>();
		if(cloud.size() > 0)
		{
			/*int min_x = 1000;
			int max_x = 0;
			int min_y = 1000;
			int max_y = 0;
			int min_z = 1000;
			int max_z = 0;
			for(int i = 1; i < cloud.size() - 1; i++)
			{
				if(tsdf_ptr[i].x > max_x)
					max_x = tsdf_ptr[i].x;
				if(tsdf_ptr[i].y > max_y)
					max_y = tsdf_ptr[i].y;
				if(tsdf_ptr[i].z > max_z)
					max_z = tsdf_ptr[i].z;
				if(tsdf_ptr[i].x < min_x)
					min_x = tsdf_ptr[i].x;
				if(tsdf_ptr[i].y < min_y)
					min_y = tsdf_ptr[i].y;
				if((tsdf_ptr[i].x < minBounds[0] || tsdf_ptr[i].x > maxBounds[0]) && (tsdf_ptr[i].y < minBounds[1] || tsdf_ptr[i].y > maxBounds[1]))
					if(tsdf_ptr[i].z < min_z)
						min_z = tsdf_ptr[i].z;
			}
			//cout << "Max x y z: " << max_x - global_shift_[0] << " " << max_y - global_shift_[1] << " " << max_z - global_shift_[2] << endl;
			//cout << "Min x y z: " << min_x - global_shift_[0] << " " << min_y - global_shift_[1] << " " << min_z - global_shift_[2] << endl;
			cout << "Global Max x y z: " << max_x + 26500 << " " << max_y + 26500 << " " << max_z + 26500 << endl;
			cout << "Global Min x y z: " << min_x + 26500 << " " << min_y + 26500 << " " << min_z + 26500 << endl;*/
			Vec3i fusionShift = global_shift_;
			for(int i = 0; i < 3; i++)
			{
				if(minBounds[i] == 0 && maxBounds[i] == 0 || last_shift)
					fusionShift[i] == -1000000;
				else if(minBounds[i] == 1)
					fusionShift[i] += maxBounds[i];
				else
					fusionShift[i] += minBounds[i];
			}
			//cout << "Fusion shift: " << fusionShift << endl;
			/*cout << "MinBounds Global: " << minBounds[0] + 26500 + global_shift_[0] << " " << minBounds[1] + 26500 + global_shift_[1] << " " << minBounds[2] + 26500 + global_shift_[2] << endl;
			cout << "MaxBounds Global: " << maxBounds[0] + 26500 + global_shift_[0] << " " << maxBounds[1] + 26500 + global_shift_[1] << " " << maxBounds[2] + 26500 + global_shift_[2] << endl;
			cout << "MinBounds Local : " << minBounds[0] << " " << minBounds[1] << " " << minBounds[2] << endl;
			cout << "MaxBounds Local : " << maxBounds[0] << " " << maxBounds[1] << " " << maxBounds[2] << endl;
			*/
			//cout << "Fusion shift Global : " << fusionShift[0] + 26500 << " " << fusionShift[1] + 26500 << " " << fusionShift[2] + 26500 << endl;
				
			if(!last_shift)
			{
				marching_thread_ = new std::thread(&kfusion::MaCuWrapper::createMeshSlice, &mcwrap_ , std::ref(cloud_slice_), fusionShift, last_shift);
			}
			else
			{
				mcwrap_.createMeshSlice(cloud_slice_, fusionShift, last_shift);
				return;
			}
		}
		else
		{
			mcwrap_.slice_count_++;
		}
	}
	// clear buffer slice and update the world model
	volume->clearSlice(&buffer_, offset);

	// shift buffer addresses
	shiftOrigin (volume, offset);
	
	slice_count_++;
}

void
kfusion::cuda::CyclicalBuffer::computeAndSetNewCubeMetricOrigin (cv::Ptr<cuda::TsdfVolume> volume, const cv::Vec3f& target_point, Vec3i& offset)
{
	// compute new origin for the cube, based on the target point
	float3 new_cube_origin_meters;
	new_cube_origin_meters.x = target_point[0] - buffer_.volume_size.x/2.0f;
	new_cube_origin_meters.y = target_point[1] - buffer_.volume_size.y/2.0f;
	new_cube_origin_meters.z = target_point[2] - buffer_.volume_size.z/2.0f;
	//printf("The old cube's metric origin was    (%f, %f, %f).\n", buffer_.origin_metric.x, buffer_.origin_metric.y, buffer_.origin_metric.z);
	//printf("The new cube's metric origin is now (%f, %f, %f).\n", new_cube_origin_meters.x, new_cube_origin_meters.y, new_cube_origin_meters.z);

	// deduce each shift in indices
	offset[0] = calcIndex((new_cube_origin_meters.x - buffer_.origin_metric.x) * ( buffer_.voxels_size.x / (float) (buffer_.volume_size.x) ));
	offset[1] = calcIndex((new_cube_origin_meters.y - buffer_.origin_metric.y) * ( buffer_.voxels_size.y / (float) (buffer_.volume_size.y) ));
	offset[2] = calcIndex((new_cube_origin_meters.z - buffer_.origin_metric.z) * ( buffer_.voxels_size.z / (float) (buffer_.volume_size.z) ));
	
	printf("The shift indices are (X:%d, Y:%d, Z:%d).\n", offset[0], offset[1], offset[2]);
	// update the cube's metric origin
	buffer_.origin_metric = new_cube_origin_meters;
	volume->setPose(Affine3f().translate(Vec3f(new_cube_origin_meters.x, new_cube_origin_meters.y,  new_cube_origin_meters.z)));
}

void kfusion::cuda::CyclicalBuffer::calcBounds(Vec3i& offset, Vec3i& minBounds, Vec3i& maxBounds)
{
	
	//Compute slice bounds
	int newX = buffer_.origin_GRID.x + offset[0];
	int newY = buffer_.origin_GRID.y + offset[1];
	int newZ = buffer_.origin_GRID.z + offset[2];

	//X
	if (newX >= 0)
	{
		minBounds[0] = buffer_.origin_GRID.x;
		maxBounds[0] = newX;    
	}
	else
	{
		minBounds[0] = newX + buffer_.voxels_size.x;
		maxBounds[0] = buffer_.origin_GRID.x + buffer_.voxels_size.x;
	}

	if (minBounds[0] > maxBounds[0])
	  std::swap (minBounds[0], maxBounds[0]);

	//Y
	if (newY >= 0)
	{
		minBounds[1] = buffer_.origin_GRID.y;
		maxBounds[1] = newY;
	}
	else
	{
		minBounds[1] = newY + buffer_.voxels_size.y;
		maxBounds[1] = buffer_.origin_GRID.y + buffer_.voxels_size.y;
	}

	if(minBounds[1] > maxBounds[1])
	  std::swap (minBounds[1], maxBounds[1]);

	//Z
	if (newZ >= 0)
	{
		minBounds[2] = buffer_.origin_GRID.z;
		maxBounds[2] = newZ;
	}
	else
	{
	  minBounds[2] = newZ + buffer_.voxels_size.z;
	  maxBounds[2] = buffer_.origin_GRID.z + buffer_.voxels_size.z;
	}

	if (minBounds[2] > maxBounds[2])
	  std::swap(minBounds[2], maxBounds[2]);

	minBounds[0] -= buffer_.origin_GRID.x;
	maxBounds[0] -= buffer_.origin_GRID.x;

	minBounds[1] -= buffer_.origin_GRID.y;
	maxBounds[1] -= buffer_.origin_GRID.y;

	minBounds[2] -= buffer_.origin_GRID.z;
	maxBounds[2] -= buffer_.origin_GRID.z;

	if (minBounds[0] < 0) // We are shifting Left
	{
	  minBounds[0] += buffer_.voxels_size.x;
	  maxBounds[0] += buffer_.voxels_size.x;
	}
		

	if (minBounds[1] < 0) // We are shifting up
	{
	  minBounds[1] += buffer_.voxels_size.y;
	  maxBounds[1] += buffer_.voxels_size.y;
	}

	if (minBounds[2] < 0) // We are shifting forward
	{
	  minBounds[2] += buffer_.voxels_size.z;
	  maxBounds[2] += buffer_.voxels_size.z;
	}
	for(int i = 0; i < 3; i++)
	{
		if(maxBounds[i] > 0)
		{
			if(minBounds[i] == 0)
			{
				maxBounds[i] += 1;
				minBounds[i] += 1;
			}
			if(maxBounds[i] == 512)
			{
				minBounds[i] -= 1;
				maxBounds[i] -= 1;
				offset[i] -=1;
			}
		}
	}
	
	
	//cout << "minBounds: " << minBounds[0] <<  ", " << minBounds[1] <<  ", " << minBounds[2] << endl;
	//cout << "maxBounds: " << maxBounds[0] <<  ", " << maxBounds[1] <<  ", " << maxBounds[2] << endl;
}
