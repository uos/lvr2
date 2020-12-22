#include "lvr2/util/Synthetic.hpp"
#include <opencv2/imgproc.hpp>

namespace lvr2 {

namespace synthetic {


MeshBufferPtr genSphere(
    int num_long,
    int num_lat)
{
    MeshBufferPtr dst_mesh = MeshBufferPtr(new MeshBuffer);

    // x = cx + r * sin(alpha) * cos(beta)
    // y = cy + r * sin(alpha) * sin(beta)
    // z = cz + r * cos(alpha)

    // alpha [-pi, pi)
    // beta [-pi, pi)

    std::vector<float> vertices;
    std::vector<unsigned int> face_indices;


    float long_inc = M_PI / static_cast<float>(num_long + 1);
    float lat_inc = (2 * M_PI) / static_cast<float>(num_lat);

    // add first and last point


    // add first vertex manually
    vertices.push_back(sin(0.0) * cos(0.0));
    vertices.push_back(sin(0.0) * sin(0.0));
    vertices.push_back(cos(0.0));

    // add first faces manually
    for(int i=0; i<num_lat; i++)
    {
        int id_bl = 0;
        int id_tl = i;
        int id_tr = i + 1;

        if(i == num_lat - 1)
        {
            id_tr -= num_lat;
        }

        face_indices.push_back(id_bl);
        face_indices.push_back(id_tl + 1);
        face_indices.push_back(id_tr + 1);
    }

    for(int i=0; i<num_long; i++)
    {
        float alpha = long_inc * (i+1);

        for(int j=0; j<num_lat; j++)
        {
            float beta = lat_inc * (j+1);

            vertices.push_back(sin(alpha) * cos(beta));
            vertices.push_back(sin(alpha) * sin(beta));
            vertices.push_back(cos(alpha));

            if(i > 0)
            {

                int id_bl = num_lat * (i-1) + j;
                int id_br = num_lat * (i-1) + j + 1;
                int id_tl = num_lat * (i)   + j;
                int id_tr = num_lat * (i)   + j + 1;

                if(j == num_lat - 1)
                {
                    id_br -= num_lat;
                    id_tr -= num_lat;
                }

                // clockwise
                
                // first face
                face_indices.push_back(id_br + 1);
                face_indices.push_back(id_bl + 1);
                face_indices.push_back(id_tl + 1);

                // second face
                face_indices.push_back(id_tl + 1);
                face_indices.push_back(id_tr + 1);
                face_indices.push_back(id_br + 1);
            }
        }
    }

    // add last vertex
    vertices.push_back(sin(M_PI) * cos(2.0 * M_PI));
    vertices.push_back(sin(M_PI) * sin(2.0 * M_PI));
    vertices.push_back(cos(M_PI));

    int num_vertices = vertices.size() / 3;
    for(int i=num_vertices-1-num_lat; i<num_vertices-1; i++)
    {
        int id_bl = i;
        int id_br = i+1;
        int id_tl = num_vertices-1;

        if(id_br == id_tl)
        {
            id_br -= num_lat;
        }

        face_indices.push_back(id_br);
        face_indices.push_back(id_bl);
        face_indices.push_back(id_tl);
    }


    // COPY
    floatArr vertex_arr(new float[vertices.size()]);
    indexArray index_arr(new unsigned int[face_indices.size()]);

    for(int i=0; i<vertices.size(); i++)
    {
        vertex_arr[i] = vertices[i];
    }

    for(int i=0; i<face_indices.size(); i++)
    {
        index_arr[i] = face_indices[i];
    }


    dst_mesh->setVertices(vertex_arr, vertices.size() / 3);
    dst_mesh->setFaceIndices(index_arr, face_indices.size() / 3);

    return dst_mesh;
}

PointBufferPtr genSpherePoints(
    int num_long,
    int num_lat)
{
    PointBufferPtr ret(new PointBuffer);

    auto mesh = genSphere(num_long, num_lat);
    (*ret)["points"] = (*mesh)["vertices"]; 

    return ret;
}

ScanImagePtr genLVRImage()
{
    ScanImagePtr imgPtr(new ScanImage);

    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(239,234,224));

    // outter lines
    cv::line(img, cv::Point(250, 50), cv::Point(450, 450), cv::Scalar(0, 0, 0), 3, CV_AA );
    cv::line(img, cv::Point(250, 50), cv::Point(50, 450), cv::Scalar(0, 0, 0), 3, CV_AA );
    cv::line(img, cv::Point(50, 450), cv::Point(450, 450), cv::Scalar(0, 0, 0), 3, CV_AA );

    // inner lines
    cv::line(img, cv::Point(150, 250), cv::Point(350, 250), cv::Scalar(0, 0, 0), 3, CV_AA );
    cv::line(img, cv::Point(150, 250), cv::Point(250, 450), cv::Scalar(0, 0, 0), 3, CV_AA );
    cv::line(img, cv::Point(350, 250), cv::Point(250, 450), cv::Scalar(0, 0, 0), 3, CV_AA );

    imgPtr->image = img;

    return imgPtr;
}

} // namespace synthetic

} // namespace lvr2