namespace lvr2
{

template<typename T>
void transformAndReducePointCloud(ModelPtr& model, int modulo, const CoordinateTransform<T>& c)
{
    transformAndReducePointCloud(model, modulo, c.sx, c.sy, c.sz, c.x, c.y, c.z);
}

template<typename T>
void transformAndReducePointCloud(
    ModelPtr model, int modulo, 
    const T& sx, const T& sy, const T& sz,
    const unsigned char& xPos, const unsigned char& yPos, const unsigned char& zPos)
{
    size_t n_ip, n_colors;
    size_t cntr = 0;
    unsigned w_colors;

    n_ip = model->m_pointCloud->numPoints();
    floatArr arr = model->m_pointCloud->getPointArray();
    ucharArr colors = model->m_pointCloud->getUCharArray("colors", n_colors, w_colors);

    // Plus one because it might differ because of the 0-index
    // better waste memory for one float than having not enough space.
    // TO-DO think about exact calculation.
    size_t targetSize = (3 * ((n_ip)/modulo)) + modulo;
    size_t targetSizeColors = (w_colors * ((n_ip)/modulo)) + modulo;
    floatArr points(new float[targetSize ]);
    ucharArr newColorsArr;

    if(n_colors)
    {
        newColorsArr = ucharArr(new unsigned char[targetSizeColors]);
    }

    for(int i = 0; i < n_ip; i++)
    {
        if(i % modulo == 0)
        {
            if(sx != 1)
            {
                arr[i * 3]         *= sx;
            }

            if(sy != 1)
            {
                arr[i * 3 + 1]     *= sy;
            }

            if(sz != 1)
            {
                arr[i * 3 + 2]     *= sz;
            }

            if((cntr * 3) < targetSize)
            {
                points[cntr * 3]     = arr[i * 3 + xPos];
                points[cntr * 3 + 1] = arr[i * 3 + yPos];
                points[cntr * 3 + 2] = arr[i * 3 + zPos];
            }
            else
            {
                std::cout << "The following is for debugging purpose" << std::endl;
                std::cout << "Cntr: " << (cntr * 3) << " targetSize: " << targetSize << std::endl;
                std::cout << "nip : " << n_ip << " modulo " << modulo << std::endl;
                break;
            }

            if(n_colors)
            {
                for (unsigned j = 0; j < w_colors; j++)
                {
                    newColorsArr[cntr * w_colors + j] = colors[i * w_colors + j];
                }
            }

            cntr++;
        }
    }

    // Pass counter because it is the actual number of points used after reduction
    // it might be 1 less than the size
    model->m_pointCloud->setPointArray(points, cntr);

    if(n_colors)
    {
        model->m_pointCloud->setColorArray(newColorsArr, cntr, w_colors);
    }
}

template<typename T>
Eigen::Matrix4d transformFrame(Eigen::Matrix4d frame, const CoordinateTransform<T>& ct)
{
    Eigen::Matrix3d basisTrans;
    Eigen::Matrix3d reflection;
    Eigen::Vector3d tmp;
    std::vector<Eigen::Vector3d> xyz;
    xyz.push_back(Eigen::Vector3d(1,0,0));
    xyz.push_back(Eigen::Vector3d(0,1,0));
    xyz.push_back(Eigen::Vector3d(0,0,1));

    reflection.setIdentity();

    if(ct.sx < 0)
    {
        reflection.block<3,1>(0,0) = (-1) * xyz[0];
    }

    if(ct.sy < 0)
    {
        reflection.block<3,1>(0,1) = (-1) * xyz[1];
    }

    if(ct.sz < 0)
    {
        reflection.block<3,1>(0,2) = (-1) * xyz[2];
    }

    // axis reflection
    frame.block<3,3>(0,0) *= reflection;

    // We are always transforming from the canonical base => T = (B')^(-1)
    basisTrans.col(0) = xyz[ct.x];
    basisTrans.col(1) = xyz[ct.y];
    basisTrans.col(2) = xyz[ct.z];

    // Transform the rotation matrix
    frame.block<3,3>(0,0) = basisTrans.inverse() * frame.block<3,3>(0,0) * basisTrans;

    // Setting translation vector
    tmp = frame.block<3,1>(0,3);
    tmp = basisTrans.inverse() * tmp;

    (frame.rightCols<1>())(0) = tmp(0);
    (frame.rightCols<1>())(1) = tmp(1);
    (frame.rightCols<1>())(2) = tmp(2);
    (frame.rightCols<1>())(3) = 1.0;

    return frame;
}

} // namespace lvr2