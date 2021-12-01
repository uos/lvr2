#include <math.h>

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
        this->init_image_data(0);
    }

    if(!image_data_initialized)
    {
        for (int y = 0; y < sizeY; y++) {
            for (int x = 0; x < sizeX; x++) {
                texture.m_data[(sizeX * y + x) * 3 + 0] = 0;
                texture.m_data[(sizeX * y + x) * 3 + 1] = 0;
                texture.m_data[(sizeX * y + x) * 3 + 2] = 0;
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

            Vector3d point = Vector3d(currentPos[0],currentPos[1],currentPos[2]);
            // TODO: This needs to be replaced with h5 Data (needs to be implemented!!!)
            Vector2d principal_point = Vector2d(-0.01985554, 0.0);
            Vector2d focal_length = Vector2d(1,1);
            float distortions[3] = {-0.15504703, -0.14184141, 0.0};

            // get uv_coord and floor it
            Vector2d uv_coord = point_to_panorama_coord(point, principal_point, focal_length, distortions);
            uv_coord[0] = std::floor(uv_coord[0]);
            uv_coord[1] = std::floor(uv_coord[1]);

            if(std::floor(uv_coord[0] < spectralPanorama.rows) && std::floor(uv_coord[1]) < spectralPanorama.cols)
            {
                if(uv_coord[0] >= 0 && uv_coord[1] >= 0)
                {
                    cv::Vec<unsigned char, 150 > pixel = spectralPanorama.at<cv::Vec<unsigned char, 150>>(uv_coord[0], uv_coord[1]);
                    texture.m_data[(sizeX * y + x) * 3 + 0] = pixel[0];
                    texture.m_data[(sizeX * y + x) * 3 + 1] = pixel[0];
                    texture.m_data[(sizeX * y + x) * 3 + 2] = pixel[0];
                }
            }
            c++;
        }
    }
    return this->m_textures.push(texture);
}


template<typename BaseVecT>
void SpectralTexturizer<BaseVecT>::init_image_data(int spectralIndex)
{
    HyperspectralCameraPtr hyperCam = scanPosition->hyperspectral_cameras.at(0);

    HyperspectralPanoramaPtr panorama = hyperCam->panoramas.at(0);
    CylindricalModel cameraModel = panorama->model;

    HyperspectralPanoramaChannelPtr panoChannel = panorama->channels.at(0);
    spectralPanorama = panoChannel->channel;

    this->image_data_initialized = true;
}

template<typename BaseVecT>
Vector2d SpectralTexturizer<BaseVecT>::point_to_panorama_coord(Vector3d point, Vector2d principal_point, Vector2d focal_length, float distortion[])
{
    Vector2d out_coord = Vector2d();
    float x = point[0];
    float y = point[1];
    float z = point[2];

    // calculate the angle of the horizontal axis
    float theta = -atan2(y, x);
    out_coord[1] = (focal_length[1] * theta) + principal_point[1];

    // calculate the angle of the vertical axis
    float phi = -z / sqrt(pow(x, 2) + pow(y, 2));

    float ndistortion = sizeof(distortion) / sizeof(distortion[0]) + 1;

    float phi_distorted = phi;
    float r = 1.0;

    // if there is distortion, calculate the distorted angle
    if(ndistortion >= 1) 
    {
        phi_distorted += distortion[0] * phi;
    }
    if(ndistortion >= 2)
    {
        phi_distorted += distortion[1] * phi * (pow(phi,2) - pow(r, 2));
    }
    if(ndistortion >= 3)
    {
        phi_distorted += distortion[2] * phi * (pow(phi,4) - pow(r, 4));
    }
    phi_distorted *= focal_length[0];
    phi_distorted += principal_point[0];

    out_coord[0] = phi_distorted;

    // adjust out_coord to image size
    out_coord[1] += M_PI;
    out_coord[1] = out_coord[1] * (spectralPanorama.cols / (2*M_PI));
    out_coord[0] += M_PI / 2;
    out_coord[0] = out_coord[0] * (spectralPanorama.rows / M_PI);
    // TODO: check if the values are adjusted correctly!!!

    return out_coord;

}
} // namespace lvr2