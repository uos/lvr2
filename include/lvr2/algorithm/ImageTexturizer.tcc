/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

namespace lvr2
{

template<typename BaseVecT>
bool ImageTexturizer<BaseVecT>::exclude_image(BaseVecT pos, const ScanImage &image_data)
{
    if (point_behind_camera(pos, image_data)) {
        return true;
    }

    // @TODO raytracing for objects between point and camera...

    return false;
}

template<typename BaseVecT>
bool ImageTexturizer<BaseVecT>::point_behind_camera(BaseVecT pos, const ScanImage &image_data)
{
    BaseVecT diff = image_data.pos - pos;
    
    Normal<double> norm(diff.x, diff.y, diff.z);
    norm.normalize();
    if (norm.dot(image_data.dir) >= 0.0f) {
        return true;
    }

    return false;
}

template<typename BaseVecT>
TextureHandle ImageTexturizer<BaseVecT>::generateTexture(
    int index,
    const PointsetSurface<BaseVecT>& surface,
    const BoundingRectangle<typename BaseVecT::CoordType>& boundingRect
)
{
    // Calculate the texture size
    unsigned short int sizeX = ceil((boundingRect.m_maxDistA - boundingRect.m_minDistA) / this->m_texelSize);
    unsigned short int sizeY = ceil((boundingRect.m_maxDistB - boundingRect.m_minDistB) / this->m_texelSize);

    // Create texture
    Texture texture(index, sizeX, sizeY, 3, 1, this->m_texelSize);

    cout << "images: " << images.size() << endl;

    // load images if not already done
    if (!image_data_initialized)
    {
        this->init_image_data();
    }


    if (image_data_initialized)
    {
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < sizeY; y++)
        {
            for (int x = 0; x < sizeX; x++)
            {

                BaseVecT currentPos =
                    boundingRect.m_supportVector
                    + boundingRect.m_vec1 * (x * this->m_texelSize + boundingRect.m_minDistA - this->m_texelSize / 2.0)
                    + boundingRect.m_vec2 * (y * this->m_texelSize + boundingRect.m_minDistB - this->m_texelSize / 2.0);

                int c = 0;

                // Init pixel with red color
                texture.m_data[(sizeX * y + x) * 3 + 0] = 255;
                texture.m_data[(sizeX * y + x) * 3 + 1] = 0;
                texture.m_data[(sizeX * y + x) * 3 + 2] = 0;

                for (const ImageData<BaseVecT> &img_data : images)
                {
                    
                    double ac_angle = (36.0 + 72.0 * c);
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
                    tmp << -0.16570779, -0.00014603, 0.98617489, -0.19101685,
                        -0.02101020, -0.99977249, -0.00367840, -0.00125086,
                        0.98595106, -0.02132927,  0.16566702, -0.05212102,
                        0, 0, 0, 1;

//                    img_data.extrinsics = tmp;

                    //cout << "Projection: " << intrinsics << endl;

                    p = angle_rot * p;

                    //cout << "P:" << p << endl;

                    double angle = p.dot(direction);

                    if (angle > 0) // is in view direction
                    {
                        // TODO I think this should maybe be done in homogeneous coords
                        p = tmp.block<3, 3>(0, 0) * p; // rotation
                        p = p + tmp.block<3, 1>(0, 3); // translation

                        Eigen::Vector3d proj = intrinsics * p; // [s * u, s * v, s * 1] = s * [u, v, 1]
                        proj /= proj[2];                       //  (s * [u, v, 1] ) / s

                        //cout << "Undist: " << proj[0] << " " << proj[1] << endl;

                        undistorted_to_distorted_uv(proj[0], proj[1], img_data);

                        //cout << "Dist:   " << proj[0] << " " << proj[1] << endl;

                        // TODO fix negated logic...
                        // if (proj[0] < 0 || std::floor(proj[0]) >= img_data.data.cols ||
                        //     proj[1] < 0 || std::floor(proj[1]) >= img_data.data.rows)
                        
                        // Check if projected point is within current image
                        if(std::floor(proj[0]) < img_data.data.cols && std::floor(proj[1]) < img_data.data.rows)
                        {
                            // Visibility check
                            if(proj[0] >= 0  && proj[1] >= 0 )
                            {
                                cv::Vec3b p = img_data.data.template at<cv::Vec3b>(std::floor(proj[1]), std::floor(proj[0]));
                                texture.m_data[(sizeX * y + x) * 3 + 0] = p[2];
                                texture.m_data[(sizeX * y + x) * 3 + 1] = p[1];
                                texture.m_data[(sizeX * y + x) * 3 + 2] = p[0];

                                // We found a valid value for this point so we can stop
                                // for know. For later optimization a best found point
                                // should be used
                                break;
                            }
                        }
                    }
                    c++;
                }
            }
        }
    }
    else
    {
        for (int y = 0; y < sizeY; y++)
        {
            for (int x = 0; x < sizeX; x++)
            {
                texture.m_data[(sizeX * y + x) * 3 + 0] = 0;
                texture.m_data[(sizeX * y + x) * 3 + 1] = 0;
                texture.m_data[(sizeX * y + x) * 3 + 2] = 0;
            }
        }

    }

    return this->m_textures.push(texture);
}

template<typename BaseVecT>
void ImageTexturizer<BaseVecT>::init_image_data()
{
    // orthogonal projection matrix is the same for every image
    for (const ScanPosition &pos : project.scans)
    {
        for (const ImageFile &img : pos.images)
        {
            ImageData<BaseVecT> image_data;

            cout << "INIT ADD: " << img.image_file.string() << endl;

            //load image
            image_data.data = cv::imread(img.image_file.string(), CV_LOAD_IMAGE_COLOR);

            cv::rotate(image_data.data, image_data.data, cv::ROTATE_90_CLOCKWISE);

            // skip image if we weren't able to load it
            if (image_data.data.empty())
            {
                continue;
            }

            for (int i = 0; i < 6; i++)
            {
                image_data.distortion_params[i] = img.distortion_params[i];
            }

            for (int i = 0; i < 4; i++)
            {
                image_data.intrinsic_params[i] = img.intrinsic_params[i];
            }

            //calculate transformation matrix
            Intrinsicsd pro;
            double* projection = pro.data();
            projection[0] = img.intrinsic_params[0];
            projection[5] = img.intrinsic_params[1];
            projection[2] = img.intrinsic_params[2];
            projection[6] = img.intrinsic_params[3];

            //Transformd transform;
            image_data.orientation = img.orientation_transform.transpose();
            image_data.extrinsics = img.extrinsic_transform.transpose();
 
            //Transformd orientation = img.orientation_transform;

            // because matrix multipl. is CM and our matrices are RM we have to do it this way
            // transform = lvr2::slam6dToLvr(pos.transform).inverse();
            // transform = transform * orientation.inverse();
            // transform = transform * img.extrinsic_transform;
            // Transformd transform_inverse = transform.inverse();
            // transform_inverse.transposeInPlace();

            //caluclate cam direction and cam pos for image in project space
            BaseVecT cam_pos(0.0f, 0.0f, 0.0f);
            Normal<double> cam_dir(0.0f, 0.0f, 1.0f);
            // cam_pos = transform_inverse * cam_pos;
            // cam_dir = transform_inverse * cam_dir;

            image_data.pos = cam_pos;
            image_data.dir = cam_dir;

            // transform from project space to image space incl orthogonal projection
            // image_data.project_to_image_transform = transform * pro;
            // image_data.project_to_image_transform.transposeInPlace();

            images.push_back(image_data);
        }
    }

    // only if we have images we should try to texturize with them...
    if (!images.empty())
    {
        image_data_initialized = true;
    }
}

template<typename BaseVecT>
template<typename ValueType>
void ImageTexturizer<BaseVecT>::undistorted_to_distorted_uv(
    ValueType &u,
    ValueType &v,
    const ImageData<BaseVecT> &img)
{
    ValueType x, y, ud, vd, r_2, r_4, r_6, r_8, fx, fy, Cx, Cy, k1, k2, k3, k4, p1, p2;

    fx = img.intrinsic_params[0];
    fy = img.intrinsic_params[1];
    Cx = img.intrinsic_params[2];
    Cy = img.intrinsic_params[3];

    k1 = img.distortion_params[0];
    k2 = img.distortion_params[1];
    k3 = img.distortion_params[2];
    k4 = img.distortion_params[3];
    p1 = img.distortion_params[4];
    p2 = img.distortion_params[5];

    x = (u - Cx)/fx;
    y = (v - Cy)/fy;

    //r_2 = std::pow(x, 2) + std::pow(y, 2);
    //r_2 = std::atan(std::sqrt(std::pow(x, 2) + std::pow(y, 2)));
    r_2 = std::pow(std::atan(std::sqrt(std::pow(x, 2) + std::pow(y, 2))), 2);
    r_4 = std::pow(r_2, 2);
    r_6 = std::pow(r_2, 3);
    r_8 = std::pow(r_2, 4);

    ud = u + x*fx*(k1*r_2 + k2*r_4 + k3*r_6 + k4*r_8) + 2*fx*x*y*p1 + p2*fx*(r_2 + 2*std::pow(x, 2));
    vd = v + y*fy*(k1*r_2 + k2*r_4 + k3*r_6 + k4*r_8) + 2*fy*x*y*p2 + p1*fy*(r_2 + 2*std::pow(y, 2));

    u = ud;
    v = vd;
}

} // namespace lvr2
