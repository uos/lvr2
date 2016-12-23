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
 /*
  * cyclical_buffer.h
  *
  *  @date 13.11.2015
  *  @author Tristan Igelbrink (Tristan@Igelbrink.com)
  */


#ifndef CYCLICAL_BUFFER_IMPL_H_
#define CYCLICAL_BUFFER_IMPL_H_

#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/tsdf_buffer.h>
#include <kfusion/LVRPipeline.hpp>
#include <Eigen/Core>
#include "types.hpp"
#include <cuda_runtime.h>
#include <thread>

namespace kfusion
{
    namespace cuda
    {
		/** \brief CyclicalBuffer implements a cyclical TSDF buffer.
		*  The class offers a simple interface, by handling shifts and maintaining the world autonomously.
		* \author Raphael Favier, Francisco Heredia
		*/
		class KF_EXPORTS CyclicalBuffer
		{

		  public:
			/** \brief Constructor for a cubic CyclicalBuffer.
			* \param[in] distance_threshold distance between cube center and target point at which we decide to shift.
			* \param[in] cube_size physical size (in meters) of the volume (here, a cube) represented by the TSDF buffer.
			* \param[in] nb_voxels_per_axis number of voxels per axis of the volume represented by the TSDF buffer.
			*/
			CyclicalBuffer (KinFuParams params): pl_(params),
			                optimize_(params.cmd_options->optimizePlanes()), no_reconstruct_(params.cmd_options->noReconstruction())
			{
				distance_threshold_ = params.shifting_distance;
				buffer_.volume_size.x = params.volume_size[0];
				buffer_.volume_size.y = params.volume_size[1];
				buffer_.volume_size.z = params.volume_size[2];
				buffer_.voxels_size.x = params.volume_dims[0];
				buffer_.voxels_size.y = params.volume_dims[1];
				buffer_.voxels_size.z = params.volume_dims[2];
				global_shift_[0] = 0;
				global_shift_[1] = 0;
				global_shift_[2] = 0;
			}


			/** \brief Constructor for a non-cubic CyclicalBuffer.
			* \param[in] distance_threshold distance between cube center and target point at which we decide to shift.
			* \param[in] volume_size_x physical size (in meters) of the volume, X axis.
			* \param[in] volume_size_y physical size (in meters) of the volume, Y axis.
			* \param[in] volume_size_z physical size (in meters) of the volume, Z axis.
			* \param[in] nb_voxels_x number of voxels for X axis of the volume represented by the TSDF buffer.
			* \param[in] nb_voxels_y number of voxels for Y axis of the volume represented by the TSDF buffer.
			* \param[in] nb_voxels_z number of voxels for Z axis of the volume represented by the TSDF buffer.
			*/
			/*CyclicalBuffer (const double distance_threshold,
			                const double volume_size_x, const double volume_size_y,
			                const double volume_size_z, const int nb_voxels_x, const int nb_voxels_y,
			                const int nb_voxels_z)
			{
				distance_threshold_ = distance_threshold;
				buffer_.volume_size.x = volume_size_x;
				buffer_.volume_size.y = volume_size_y;
				buffer_.volume_size.z = volume_size_z;
				buffer_.voxels_size.x = nb_voxels_x;
				buffer_.voxels_size.y = nb_voxels_y;
				buffer_.voxels_size.z = nb_voxels_z;
			}*/

			~CyclicalBuffer()
			{
				//double averageMCTime = mcwrap_.calcTimeStats();
				//cout << "----- Average time for processing one tsdf value " << averageMCTime << "ns -----" << endl;
			}

		    /** \brief Check if shifting needs to be performed, returns true if so.
			  Shifting is considered needed if the target point is farther than distance_treshold_.
			  The target point is located at distance_camera_point on the local Z axis of the camera.
			* \param[in] volume pointer to the TSDFVolume living in GPU
			* \param[in] cam_pose global pose of the camera in the world
			* \param[in] distance_camera_target distance from the camera's origin to the target point
			* \param[in] perform_shift if set to false, shifting is not performed. The function will return true if shifting is needed.
			* \param[in] last_shift if set to true, the whole cube will be shifted. This is used to push the whole cube to the world model.
			* \param[in] force_shift if set to true, shifting is forced.
			* \return true is the cube needs to be or has been shifted.
			*/
			bool checkForShift (cv::Ptr<cuda::TsdfVolume> volume,
			                    const Affine3f &cam_pose, const double distance_camera_target,
			                    const bool perform_shift = true, const bool last_shift = false,
			                    const bool record_mode = false);

		    /** \brief Perform shifting operations:
			  Compute offsets.
			  Extract current slice from TSDF buffer.
			  Extract existing data from world.
			  Clear shifted slice in TSDF buffer.
			  Push existing data into TSDF buffer.
			  Update rolling buffer
			  Update world model.
			* \param[in] volume pointer to the TSDFVolume living in GPU
			* \param[in] target_point target point around which the new cube will be centered
			* \param[in] last_shift if set to true, the whole cube will be shifted. This is used to push the whole cube to the world model.
			*/
			void performShift (cv::Ptr<cuda::TsdfVolume> volume, const cv::Vec3f& target_point,  const Affine3f &cam_pose, const bool last_shift = false, const bool record_mode = false);

		   /** \brief Sets the distance threshold between cube's center and target point that triggers a shift.
			* \param[in] threshold the distance in meters at which to trigger shift.
			*/
		    void setDistanceThreshold (const double threshold)
		    {
			  distance_threshold_ = threshold;
		    }

		    /** \brief Returns the distance threshold between cube's center and target point that triggers a shift. */
		    float getDistanceThreshold () { return (distance_threshold_); }

		    /** \brief get a pointer to the tsdf_buffer structure.
			* \return a pointer to the tsdf_buffer used by cyclical buffer object.
			*/
		    tsdf_buffer& getBuffer () { return (buffer_); }

		    /** \brief Set the physical size represented by the default TSDF volume.
		    * \param[in] size_x size of the volume on X axis, in meters.
		    * \param[in] size_y size of the volume on Y axis, in meters.
		    * \param[in] size_z size of the volume on Z axis, in meters.
		    */
			void setVolumeSize (const Vec3f size)
			{
				buffer_.volume_size.x = size[0];
				buffer_.volume_size.y = size[1];
				buffer_.volume_size.z = size[2];
			}

			void setVoxelSize (const Vec3f vsize)
			{
				buffer_.voxels_size.x = vsize[0];
				buffer_.voxels_size.y = vsize[1];
				buffer_.voxels_size.z = vsize[2];
			}

			Vec3f getOrigin()
			{
				return Vec3f(buffer_.origin_metric.x, buffer_.origin_metric.y, buffer_.origin_metric.z);
			}

		  /** \brief Set the physical size represented by the default TSDF volume.
		  * \param[in] size size of the volume on all axis, in meters.
		  */
		  void setVolumeSize (const double size)
		  {
			buffer_.volume_size.x = size;
			buffer_.volume_size.y = size;
			buffer_.volume_size.z = size;
		  }

		  /** \brief Computes and set the origin of the new cube (relative to the world), centered around a the target point.
			* \param[in] target_point the target point around which the new cube will be centered.
			* \param[out] shiftX shift on X axis (in indices).
			* \param[out] shiftY shift on Y axis (in indices).
			* \param[out] shiftZ shift on Z axis (in indices).
			*/
		  void computeAndSetNewCubeMetricOrigin (cv::Ptr<cuda::TsdfVolume> volume, const cv::Vec3f& target_point, Vec3i& offset);

		  /** \brief Initializes memory pointers of the  cyclical buffer (start, end, current origin)
			* \param[in] tsdf_volume pointer to the TSDF volume managed by this cyclical buffer
			*/
		  void initBuffer (cv::Ptr<cuda::TsdfVolume> tsdf_volume)
		  {
			buffer_.tsdf_memory_start = tsdf_volume->getCoord(0, 0, 0, 0, 0);
			buffer_.tsdf_memory_end = tsdf_volume->getCoord(buffer_.voxels_size.x - 1, buffer_.voxels_size.y - 1, buffer_.voxels_size.z - 1, buffer_.voxels_size.x, buffer_.voxels_size.y);

			buffer_.tsdf_rolling_buff_origin = buffer_.tsdf_memory_start;
		  }

		  /** \brief Reset buffer structure
			* \param[in] tsdf_volume pointer to the TSDF volume managed by this cyclical buffer
			*/
		  void resetBuffer (cv::Ptr<cuda::TsdfVolume> tsdf_volume)
		  {
			buffer_.origin_GRID.x = 0; buffer_.origin_GRID.y = 0; buffer_.origin_GRID.z = 0;
			buffer_.origin_GRID_global.x = 0.f; buffer_.origin_GRID_global.y = 0.f; buffer_.origin_GRID_global.z = 0.f;
			Vec3f position = tsdf_volume->getPose().translation();
			buffer_.origin_metric.x = position[0];
			buffer_.origin_metric.y = position[1];
			buffer_.origin_metric.z = position[2];
			initBuffer (tsdf_volume);
		  }

		  void resetMesh(){/*mcwrap_.resetMesh();*/}
		  void addImgPose(ImgPose* imgPose){ imgPoses_.push_back(imgPose);}

		  MeshPtr getMesh() {return pl_.getMesh();}

		  int getSliceCount(){return slice_count_;}


		private:

		  /** \brief buffer used to extract XYZ values from GPU */
		  DeviceArray<Point> cloud_buffer_device_;

		  /** \brief distance threshold (cube's center to target point) to trigger shift */
		  double distance_threshold_;
		  Vec3i global_shift_;
		  cv::Mat cloud_slice_;
		  std::vector<ImgPose*> imgPoses_;
		  int slice_count_ = 0;
		  Affine3f last_camPose_;
		  bool optimize_, no_reconstruct_;

		  /** \brief structure that contains all TSDF buffer's addresses */
		  tsdf_buffer buffer_;

		  //MaCuWrapper mcwrap_;
		  LVRPipeline pl_;
			inline int calcIndex(float f) const
			{
				return f < 0 ? f-.5:f+.5;
			}

		  float euclideanDistance(const Point& p1, const Point& p2);

		  void getEulerYPR(float& yaw, float& pitch, float& roll, cv::Affine3<float>::Mat3& mat,  unsigned int solution_number = 1);

		  void calcBounds(Vec3i& offset, Vec3i& minBounds, Vec3i& maxBounds);

		  /** \brief updates cyclical buffer origins given offsets on X, Y and Z
			* \param[in] tsdf_volume pointer to the TSDF volume managed by this cyclical buffer
			* \param[in] offset_x offset in indices on axis X
			* \param[in] offset_y offset in indices on axis Y
			* \param[in] offset_z offset in indices on axis Z
			*/
		  void shiftOrigin (cv::Ptr<cuda::TsdfVolume> tsdf_volume, Vec3i offset)
		  {
			// shift rolling origin (making sure they keep in [0 - NbVoxels[ )
			buffer_.origin_GRID.x += offset[0];
			if(buffer_.origin_GRID.x >= buffer_.voxels_size.x)
			  buffer_.origin_GRID.x -= buffer_.voxels_size.x;
			else if(buffer_.origin_GRID.x < 0)
			{
			  buffer_.origin_GRID.x += buffer_.voxels_size.x ;
			}

			buffer_.origin_GRID.y += offset[1];
			if(buffer_.origin_GRID.y >= buffer_.voxels_size.y)
			  buffer_.origin_GRID.y -= buffer_.voxels_size.y;
			else if(buffer_.origin_GRID.y < 0)
			{
			  buffer_.origin_GRID.y += buffer_.voxels_size.y;
			}

			buffer_.origin_GRID.z += offset[2];
			if(buffer_.origin_GRID.z >= buffer_.voxels_size.z)
			  buffer_.origin_GRID.z -= buffer_.voxels_size.z;
			else if(buffer_.origin_GRID.z < 0)
			{
			  buffer_.origin_GRID.z += buffer_.voxels_size.z;
			}
			// update memory pointers
			CudaData localVolume = tsdf_volume->data();
			buffer_.tsdf_memory_start = tsdf_volume->getCoord(0, 0, 0, 0, 0);
			buffer_.tsdf_memory_end = tsdf_volume->getCoord(buffer_.voxels_size.x - 1, buffer_.voxels_size.y - 1, buffer_.voxels_size.z - 1, buffer_.voxels_size.x, buffer_.voxels_size.y);
			buffer_.tsdf_rolling_buff_origin = tsdf_volume->getCoord(buffer_.origin_GRID.x, buffer_.origin_GRID.y, buffer_.origin_GRID.z, buffer_.voxels_size.x, buffer_.voxels_size.y);

			// update global origin
			global_shift_[0] = buffer_.origin_GRID_global.x += offset[0];
			global_shift_[1] = buffer_.origin_GRID_global.y += offset[1];
			global_shift_[2] = buffer_.origin_GRID_global.z += offset[2];
		  }

	  };
	}
}

#endif // CYCLICAL_BUFFER_IMPL_H_
