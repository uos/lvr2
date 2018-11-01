namespace lvr2
{

template<typename BaseVecT>
bool ImageTexturizer<BaseVecT>::exclude_image(Vector<BaseVecT> pos, const ImageData<BaseVecT> &image_data)
{
    if (point_behind_camera(pos, image_data)) {
        return true;
    }

    // @TODO raytracing for objects between point and camera...

    return false;
}

template<typename BaseVecT>
bool ImageTexturizer<BaseVecT>::point_behind_camera(Vector<BaseVecT> pos, const ImageData<BaseVecT> &image_data)
{
    Vector<BaseVecT> norm = image_data.pos - pos;
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
    const BoundingRectangle<BaseVecT>& boundingRect
)
{
    // Calculate the texture size
    unsigned short int sizeX = ceil((boundingRect.m_maxDistA - boundingRect.m_minDistA) / this->m_texelSize);
    unsigned short int sizeY = ceil((boundingRect.m_maxDistB - boundingRect.m_minDistB) / this->m_texelSize);

    // Create texture
    Texture texture(index, sizeX, sizeY, 3, 1, this->m_texelSize);

    // load images if not already done
    if (!image_data_initialized)
    {
        this->init_image_data();
    }

    if (image_data_initialized)
    {
        for (int y = 0; y < sizeY; y++)
        {
            for (int x = 0; x < sizeX; x++)
            {

                Vector<BaseVecT> currentPos =
                    boundingRect.m_supportVector
                    + boundingRect.m_vec1 * (x * this->m_texelSize + boundingRect.m_minDistA - this->m_texelSize / 2.0)
                    + boundingRect.m_vec2 * (y * this->m_texelSize + boundingRect.m_minDistB - this->m_texelSize / 2.0);

                for (const ImageData<BaseVecT> &img_data : images)
                {
                    // transforming from slam6D coords to riegl coords
                    Vector<BaseVecT> pos(currentPos.z/100.0, -currentPos.x/100.0, currentPos.y/100.0);

                    if (exclude_image(pos, img_data))
                        continue;

                    pos = img_data.project_to_image_transform * pos;

                    float u = (float) img_data.data.rows - pos[0]/pos[2];
                    float v = pos[1]/pos[2];

                    undistorted_to_distorted_uv(u, v, img_data);

                    // @TODO option to do bilinear filtering aswell for pixel selection...
                    int ud = (int) (u + 0.5);
                    int vd = (int) (v + 0.5);

                    if (ud >= 0 && ud < img_data.data.rows && vd >= 0 && vd < img_data.data.cols)
                    {
                        // using template keyword because elsewise < would be interpreted as less
                        // than operator
                        const cv::Vec3b color = img_data.data.template at<cv::Vec3b>(ud, vd);

                        // this works as well...
                        //const cv::Vec3b color = ((cv::Mat) (img_data.data)).at<cv::Vec3b>(ud, vd);

                        // and this as well...
                        //const cv::Mat &bla = img_data.data;
                        //const cv::Vec3b color = bla.at<cv::Vec3b>(ud, vd);


                        // OpenCV saves colors in BGR order
                        uint8_t r = color[2], g = color[1], b = color[0];

                        texture.m_data[(sizeX * y + x) * 3 + 0] = r;
                        texture.m_data[(sizeX * y + x) * 3 + 1] = g;
                        texture.m_data[(sizeX * y + x) * 3 + 2] = b;
                        break;
                    }
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

            //load image
            image_data.data = cv::imread(img.image_file.string(), CV_LOAD_IMAGE_COLOR);

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
            Matrix4<BaseVecT> projection;
            projection[0] = img.intrinsic_params[0];
            projection[5] = img.intrinsic_params[1];
            projection[2] = img.intrinsic_params[2];
            projection[6] = img.intrinsic_params[3];

            bool dummy;
            Matrix4<BaseVecT> transform;
            Matrix4<BaseVecT> orientation = img.orientation_transform;

            // because matrix multipl. is CM and our matrices are RM we have to do it this way
            transform = Util::slam6d_to_riegl_transform(pos.transform).inv(dummy);
            transform = transform * orientation.inv(dummy);
            transform = transform * img.extrinsic_transform;
            Matrix4<BaseVecT> transform_inverse = transform.inv(dummy);
            transform_inverse.transpose();

            //caluclate cam direction and cam pos for image in project space
            Vector<BaseVecT> cam_pos = {0.0f, 0.0f, 0.0f};
            Normal<BaseVecT> cam_dir = {0.0f, 0.0f, 1.0f};
            cam_pos = transform_inverse * cam_pos;
            cam_dir = transform_inverse * cam_dir;

            image_data.pos = cam_pos;
            image_data.dir = cam_dir;

            // transform from project space to image space incl orthogonal projection
            image_data.project_to_image_transform = transform * projection;
            image_data.project_to_image_transform.transpose();

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
