#pragma once

#include <kfusion/types.hpp>
#include <kfusion/tsdf_buffer.h>
#include <cuda_runtime.h>

namespace kfusion
{
    namespace cuda
    {
        class KF_EXPORTS TsdfVolume
        {
        public:
            TsdfVolume(const cv::Vec3i& dims);
            virtual ~TsdfVolume();

            void create(const Vec3i& dims);

            Vec3i getDims() const;
            Vec3f getVoxelSize() const;

            const CudaData data() const;
            CudaData data();

            Vec3f getSize() const;
            void setSize(const Vec3f& size);

            float getTruncDist() const;
            void setTruncDist(float distance);

            int getMaxWeight() const;
            void setMaxWeight(int weight);

            Affine3f getPose() const;
            void setPose(const Affine3f& pose);

            float getRaycastStepFactor() const;
            void setRaycastStepFactor(float factor);
			
			ushort2* getCoord(int x, int y, int z, int dim_x, int dim_y);
			
			
            float getGradientDeltaFactor() const;
            void setGradientDeltaFactor(float factor);

            Vec3i getGridOrigin() const;
            void setGridOrigin(const Vec3i& origin);
            
            void clearSlice(const kfusion::tsdf_buffer* buffer, const Vec3i offset) const;

            
            virtual void clear();
            virtual void applyAffine(const Affine3f& affine);
            virtual void integrate(const Dists& dists, tsdf_buffer& buffer, const Affine3f& camera_pose, const Intr& intr);
            virtual void raycast(const Affine3f& camera_pose, tsdf_buffer& buffer, const Intr& intr, Depth& depth, Normals& normals);
            virtual void raycast(const Affine3f& camera_pose, tsdf_buffer& buffer, const Intr& intr, Cloud& points, Normals& normals);

			/** \brief Generates cloud using GPU in connected6 mode only
			  * \param[out] cloud_buffer_xyz buffer to store point cloud
			  * \param cloud_buffer_intensity
			  * \param[in] buffer Pointer to the buffer struct that contains information about memory addresses of the tsdf volume memory block, which are used for the cyclic buffer.
			  * \param[in] shiftX Offset in indices.
			  * \param[in] shiftY Offset in indices.
			  * \param[in] shiftZ Offset in indices.
			  * \return DeviceArray with disabled reference counting that points to filled part of cloud_buffer.
			  */
			DeviceArray<Point>
			fetchSliceAsCloud (DeviceArray<Point>& cloud_buffer, const kfusion::tsdf_buffer* buffer, const Vec3i minBounds, const Vec3i maxBounds, const Vec3i globalShift ) const;

            void swap(CudaData& data);

            DeviceArray<Point> fetchCloud(DeviceArray<Point>& cloud_buffer, const tsdf_buffer& buffer) const;
            void fetchNormals(const DeviceArray<Point>& cloud, const tsdf_buffer& buffer, DeviceArray<Normal>& normals) const;

            struct Entry
            {
                typedef unsigned short half;

                half tsdf;
                unsigned short weight;

                static float half2float(half value);
                static half float2half(float value);
            };
        private:
            CudaData data_;

            float trunc_dist_;
            int max_weight_;
            Vec3i dims_;
            Vec3f size_;
            Affine3f pose_;

            float gradient_delta_factor_;
            float raycast_step_factor_;
        };
    }
}
