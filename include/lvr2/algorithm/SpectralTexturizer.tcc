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

            // get uv_coord and floor it
            Vector2d uv_coord = point_to_panorama_coord(point, principal_point, focal_length, distortions);
            uv_coord[0] = std::floor(uv_coord[0]);
            uv_coord[1] = std::floor(uv_coord[1]);

            if(std::floor(uv_coord[0] < spectralPanorama.rows) && std::floor(uv_coord[1]) < spectralPanorama.cols)
            {
                if(uv_coord[0] >= 0 && uv_coord[1] >= 0)
                {
                    cv::Vec<unsigned char, 1 > pixel = spectralPanorama.at<cv::Vec<unsigned char, 1>>(uv_coord[0], uv_coord[1]);
                    texture.m_data[(sizeX * y + x) * 3 + 0] = pixel[0];
                    texture.m_data[(sizeX * y + x) * 3 + 1] = pixel[0];
                    texture.m_data[(sizeX * y + x) * 3 + 2] = pixel[0];
                }
            }
            c++;
        }
    }
    texture.m_layerName = "hyperspectral_grayscale_" + std::to_string(channelIndex);
    return this->m_textures.push(texture);
}


template<typename BaseVecT>
void SpectralTexturizer<BaseVecT>::init_image_data(HyperspectralPanoramaPtr pano, int channelIndex)
{

    // TODO: load data from h5 file (after h5 rework is implemented!!)
    principal_point = Vector2d(-0.01985554, 0.0);
    focal_length = Vector2d(0,0);
    camera_fov = Vector2d(0.82903139,6.28318531);

    distortions.push_back(-0.15504703);
    distortions.push_back(-0.14184141);
    distortions.push_back(0.0);

    this->channelIndex = channelIndex;
    HyperspectralPanoramaChannelPtr panoChannel = pano->channels[channelIndex];
    spectralPanorama = panoChannel->channel;

    prepare_camera_data();
    this->image_data_initialized = true;
}

template<typename BaseVecT>
void SpectralTexturizer<BaseVecT>::prepare_camera_data() {
    // resolution(y,x)
    Vector2d resolution = Vector2d(spectralPanorama.rows, spectralPanorama.cols);
    focal_length[0] = resolution[0] / camera_fov[0] + focal_length[0] * resolution[0];
    focal_length[1] = resolution[1] / camera_fov[1] + focal_length[1] * resolution[1];
    principal_point[0] = resolution[0] / 2.0 + principal_point[0] * resolution[0];
    principal_point[1] = resolution[1] / 2.0 + principal_point[1] * resolution[1];
}


template<typename BaseVecT>
Vector2d SpectralTexturizer<BaseVecT>::point_to_panorama_coord(Vector3d point, Vector2d principal_point, Vector2d focal_length, std::vector<float> distortions)
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

    float phi_distorted = phi;
    float r = 1.0;

    // if there is distortion, calculate the distorted angle
    if(distortions.size() >= 1) 
    {
        phi_distorted += distortions[0] * phi;
    }
    if(distortions.size() >= 2)
    {
        phi_distorted += distortions[1] * phi * (pow(phi,2) - pow(r, 2));
    }
    if(distortions.size() >= 3)
    {
        phi_distorted += distortions[2] * phi * (pow(phi,4) - pow(r, 4));
    }
    phi_distorted *= focal_length[0];
    phi_distorted += principal_point[0];

    out_coord[0] = phi_distorted;

    return out_coord;

}
} // namespace lvr2