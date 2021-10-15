namespace lvr2
{

template<typename BaseVecT>
TextureHandle SpectralTexturizer<BaseVecT>::generateTexture(
    int index,
    const PointsetSurface<BaseVecT>& surface,
    const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect
)
{
    // Calculate texture size
    unsigned short int sizeX = ceil((boundingRect.m_maxDistA - boundingRect.m_minDistA) / this->m_texelSize);
    unsigned short int sizeY = ceil((boundingRect.m_maxDistB - boundingRect.m_minDistB) / this->m_texelSize);

    // Create texture
    Texture texture(index, sizeX, sizeY, 3, 1, this->m_texelSize);

    // cout << "images: " << images.size() << endl;

    // load image if not already done
    if(!image_data_initialized)
    {
        this->init_image_data(0, 0);
    }

    if(!image_data_initialized)
    {
        for (int y = 0; y < sizeY; y++) {
            for (int x = 0; x < sizeX; x++) {
                texture.m_data[(sizeX * y + x) * 3 + 0] = 0;
                texture.m_data[(sizeX * y + x) * 3 + 1] = 0;
                texture.m_data[(sizeX * y + x) * 3 + 1] = 0;
            }
        }
        return this->m_textures.push(texture);
    }
    int totalCallCount = 0;
    int correctAngleCount = 0;
    int correctSizeCount = 0;
    int visibleCount = 0;

    // #pragma omp parallel for collapse(2)
    for (int y = 0; y < sizeY; y++) {
        for (int x = 0; x < sizeX; x++) {
            BaseVecT currentPos =
                boundingRect.m_supportVector
                + boundingRect.m_vec1 * (x * this->m_texelSize + boundingRect.m_minDistA - this->m_texelSize / 2.0)
                + boundingRect.m_vec2 * (y * this->m_texelSize + boundingRect.m_minDistB - this->m_texelSize / 2.0);

            int c = 0;


            // init pixel with color red
            texture.m_data[(sizeX * y + x) * 3 + 0] = 255;
            texture.m_data[(sizeX * y + x) * 3 + 1] = 0;
            texture.m_data[(sizeX * y + x) * 3 + 2] = 0;

            double ac_angle = (36.0 + 72.0 + c);
            ac_angle *= (M_PI/180.0);

            Eigen::Vector3d direction(1.0, 0.0, 0.0);
            Eigen::Matrix3d angle_rot;
            angle_rot = Eigen::AngleAxisd(ac_angle, Eigen::Vector3d::UnitZ());

            Eigen::Vector3d p(currentPos[0], currentPos[1], currentPos[2]);

            Intrinsicsd intrinsics;
            intrinsics  <<
                2395.4336550315002 , 0 , 3027.8728609530291 ,         // UOS
                0 , 2393.3126174899603 , 2031.02743729632 ,
                0 , 0 , 1;

            Transformd tmp;
            tmp << 
                -0.16570779, -0.00014603, 0.98617489, -0.19101685,
                -0.02101020, -0.99977249, -0.00367840, -0.00125086,
                0.98595106, -0.02132927,  0.16566702, -0.05212102,
                0, 0, 0, 1;

            p = angle_rot * p;

            double angle = p.dot(direction);

            totalCallCount++;
            // TODO: --RED BUG--: remove angle and try again?
            if(angle > 0) // is in new view 
            {
                correctAngleCount++;
                p = tmp.block<3, 3>(0, 0) * p; // rotation
                p = p + tmp.block<3,1>(0,3); // translation

                Eigen::Vector3d proj = intrinsics * p;
                proj /= proj[2];

                std::cout << proj[0] << " " << spectralPanorama.cols << " " << proj[1] << " " << spectralPanorama.rows << std::endl;


                if(std::floor(proj[0] < spectralPanorama.cols) && std::floor(proj[1]) < spectralPanorama.rows)
                {
                    correctSizeCount++;
                    // check if visible
                    if(proj[0] >= 0 &&proj[1] >= 0)
                    {
                        visibleCount++;
                        cv::Vec3b p = spectralPanorama.template at<cv::Vec3b>(std::floor(proj[1]), std::floor(proj[0]));
                        texture.m_data[(sizeX * y + x) * 3 + 0] = p[2];
                        texture.m_data[(sizeX * y + x) * 3 + 1] = p[1];
                        texture.m_data[(sizeX * y + x) * 3 + 2] = p[0];
                        break;
                    }
                        
                }
            }
            c++;
        }
    }

    std::cout << "START COUNT" << std::endl;
    std::cout << "TOTAL COUNT: " << totalCallCount << std::endl;
    std::cout << "ANGLE COUNT: " << correctAngleCount  << std::endl;
    std::cout << "SIZE COUNT: " << correctSizeCount << std::endl;
    std::cout << "VISIBLE COUNT: " << visibleCount << std::endl;
    std::cout << "END COUNT" << std::endl;


    return this->m_textures.push(texture);
}


template<typename BaseVecT>
void SpectralTexturizer<BaseVecT>::init_image_data(int scanPositionIndex, int spectralIndex)
{
    ScanPositionPtr scanPos = project->positions.at(scanPositionIndex);
    HyperspectralCameraPtr hyperCam = scanPos->hyperspectral_cameras.at(0);
    // TODO: check this
    // CameraPtr camPtr = scanPos->cameras.at(0);

    HyperspectralPanoramaPtr panorama = hyperCam->panoramas.at(0);
    HyperspectralPanoramaChannelPtr panoChannel = panorama->channels.at(0);
    spectralPanorama = panoChannel->channel;

    this->image_data_initialized = true;
}

template<typename BaseVecT>
template<typename ValueType>
void SpectralTexturizer<BaseVecT>::undistorted_to_distorted_uv(
    ValueType &u,
    ValueType &v,
    const cv::Mat &img)
{
    ValueType x, y, ud, vd, r_2, r_4, r_6, r_8, fx, fy, Cx, Cy, k1, k2, k3, k4, p1, p2;




}

} // namespace lvr2