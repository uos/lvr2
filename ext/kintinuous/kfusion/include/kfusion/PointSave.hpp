#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <lvr/geometry/ColorVertex.hpp>


#include <kfusion/types.hpp>

class PointSave {
	public:
		/**
		 * saves more contours	 
		 * @param size : size of all elements
		 * @param filename : output filename
		 */
		bool savePoints(const char* filename,std::vector<std::vector<lvr::ColorVertex<float, unsigned char> > > rcb,int size);
		
		bool savePointsWithCameras(const char* filename,std::vector<std::vector<lvr::ColorVertex<float, unsigned char> > > rcb,int size,std::vector<kfusion::ImgPose> image_pose_buffer);

		bool saveBoundingRectangles(const char* filename, std::vector< std::vector<cv::Point3f> > b_rects, int size);
	
	private:
	
};
