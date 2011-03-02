/*
 * StannPointCloudManager.cpp
 *
 *  Created on: 07.02.2011
 *      Author: Thomas Wiemann
 */

#include <boost/filesystem.hpp>
#include "BoundingBox.hpp"

namespace lssr{

template<typename T>
StannPointCloudManager<T>::StannPointCloudManager(T **points,
        T **normals,
        size_t n,
        const size_t &kn,
        const size_t &ki)
        : m_kn(kn), m_ki(ki), m_numPoints(n)
{

    // Save data
    this->m_points = points;
    this->m_normals = normals;

    // Be sure that point information was given
    assert(this->m_points);

    // Calculate bounding box
    cout << timestamp << " Calculating bounding box." << endl;
    for(size_t i = 0; i < n; i++)
    {
        this->m_boundingBox.expand(points[i][0], points[i][1], points[i][2]);
    }

    // Create kd tree
    cout << timestamp << " Creating STANN Kd-Tree..." << endl;
    m_pointTree = sfcnn< T*, 3, T>(this->m_points, n, 4);

    // Estimate surface normals if necessary
    if(!this->m_normals)
    {
        estimateSurfaceNormals();
        interpolateSurfaceNormals();
    }
    else
    {
        cout << timestamp << " Using the given normals." << endl;
    }

}

template<typename T>
void StannPointCloudManager<T>::estimateSurfaceNormals()
{
    int k_0 = m_kn;

    cout << timestamp << "Initializing normal array..." << endl;

    //Initialize normal array
    this->m_normals = new float*[m_numPoints];

    float mean_distance;
    // Create a progress counter
    string comment = timestamp.getElapsedTime() + "Estimating normals ";
    ProgressBar progress(m_numPoints, comment);

    #pragma omp parallel for
    for(int i = 0; i < m_numPoints; i++){

        Vertexf query_point;
        Normalf normal;

        // We have to fit these vector to have the
        // correct return values when performing the
        // search on the stann kd tree. So we don't use
        // the template parameter T for di
        vector<unsigned long> id;
        vector<double> di;

        int n = 0;
        size_t k = k_0;

        while(n < 5){

            n++;
            /**
             *  @todo Maybe this should be done at the end of the loop
             *        after the bounding box check
             */
            k = k * 2;

            //T* point = this->m_points[i];
            m_pointTree.ksearch(this->m_points[i], k, id, di, 0);

            float min_x = 1e15;
            float min_y = 1e15;
            float min_z = 1e15;
            float max_x = - min_x;
            float max_y = - min_y;
            float max_z = - min_z;

            float dx, dy, dz;
            dx = dy = dz = 0;

            // Calculate the bounding box of found point set
            /**
             * @todo Use the bounding box object from the old model3d
             *       library for bounding box calculation...
             */
            for(int j = 0; j < k; j++){
                min_x = min(min_x, this->m_points[id[j]][0]);
                min_y = min(min_y, this->m_points[id[j]][1]);
                min_z = min(min_z, this->m_points[id[j]][2]);

                max_x = max(max_x, this->m_points[id[j]][0]);
                max_y = max(max_y, this->m_points[id[j]][1]);
                max_z = max(max_z, this->m_points[id[j]][2]);

                dx = max_x - min_x;
                dy = max_y - min_y;
                dz = max_z - min_z;
            }

            if(boundingBoxOK(dx, dy, dz)) break;
            //break;

        }

        // Create a query point for the current point
        query_point = Vertex<T>(this->m_points[i][0],
                this->m_points[i][1],
                this->m_points[i][2]);

        // Interpolate a plane based on the k-neighborhood
        Plane<T> p = calcPlane(query_point, k, id);

        // Get the mean distance to the tangent plane
        mean_distance = meanDistance(p, id, k);

        // Flip normals towards the center of the scene
        normal =  p.n;
        if(normal * (query_point - m_centroid) < 0) normal = normal * -1;

        // Save result in normal array
        this->m_normals[i] = new T[3];
        this->m_normals[i][0] = normal[0];
        this->m_normals[i][1] = normal[1];
        this->m_normals[i][2] = normal[2];

        ++progress;
    }
    cout << endl;;
}


template<typename T>
void StannPointCloudManager<T>::interpolateSurfaceNormals()
{
    // Create a temporal normal array for the
    vector<Normal<T> > tmp(m_numPoints, Normal<T>());

    // Create progress output
    string comment = timestamp.getElapsedTime() + "Interpolating normals ";
    ProgressBar progress(m_numPoints, comment);

    // Interpolate normals
    #pragma omp parallel for
    for(int i = 0; i < m_numPoints; i++){

        vector<unsigned long> id;
        vector<double> di;

        m_pointTree.ksearch(this->m_points[i], m_ki, id, di, 0);

        Vertex<T> mean;
        Normal<T> mean_normal;

        for(int j = 0; j < m_ki; j++){
            mean += Vertex<T>(this->m_normals[id[j]][0],
                              this->m_normals[id[j]][1],
                              this->m_normals[id[j]][2]);
        }
        mean_normal = Normal<T>(mean);

        tmp[i] = mean;

        /**
         * @todo Try to remove this code. Should improve the results at all.
         */
        for(int j = 0; j < m_ki; j++){
            Normal<T> n(this->m_normals[id[j]][0],
                        this->m_normals[id[j]][1],
                        this->m_normals[id[j]][2]);


            // Only override existing normals if the interpolated
            // normals is significantly different from the initial
            // estimation. This helps to avoid a to smooth normal
            // field
            if(fabs(n * mean_normal) > 0.2 ){
                this->m_normals[id[j]][0] = mean_normal[0];
                this->m_normals[id[j]][1] = mean_normal[1];
                this->m_normals[id[j]][2] = mean_normal[2];
            }
        }
        ++progress;
    }
    cout << endl;
    cout << timestamp << "Copying normals..." << endl;

    for(int i = 0; i < m_numPoints; i++){
        this->m_normals[i][0] = tmp[i][0];
        this->m_normals[i][1] = tmp[i][1];
        this->m_normals[i][2] = tmp[i][2];
    }

}

template<typename T>
bool StannPointCloudManager<T>::boundingBoxOK(const T &dx, const T &dy, const T &dz)
{
    /**
     * @todo Replace magic number here.
     */
    float e = 0.05;
    if(dx < e * dy) return false;
    else if(dx < e * dz) return false;
    else if(dy < e * dx) return false;
    else if(dy < e * dz) return false;
    else if(dz < e * dx) return false;
    else if(dy < e * dy) return false;
    return true;
}

template<typename T>
T StannPointCloudManager<T>::meanDistance(const Plane<T> &p,
        const vector<unsigned long> &id, const int &k)
{
    T sum = 0;
    for(int i = 0; i < k; i++){
        sum += distance(fromID(id[i]), p);
    }
    sum = sum / k;
    return sum;
}

template<typename T>
T StannPointCloudManager<T>::distance(Vertex<T> v, Plane<T> p)
{
    return fabs((v - p.p) * p.n);
}

template<typename T>
T StannPointCloudManager<T>::distance(Vertex<T> v)
{
    int k = 1;

    vector<unsigned long> id;
    vector<double> di;

    //Allocate ANN point
    float * p;
    p = new float[3];
    p[0] = v[0]; p[1] = v[1]; p[2] = v[2];

    //Find nearest tangent plane
    m_pointTree.ksearch(p, k, id, di, 0);

    Vertex<T> nearest;
    Normal<T> normal;

    for(int i = 0; i < k; i++){
        //Get nearest tangent plane
        Vertex<T> vq (this->m_points[id[i]][0], this->m_points[id[i]][1], this->m_points[id[i]][2]);

        //Get normal
        Normal<T> n(this->m_normals[id[i]][0], this->m_normals[id[i]][1], this->m_normals[id[i]][2]);

        nearest += vq;
        normal += n;

    }

    normal /= k;
    nearest /= k;


    //Calculate distance
    float distance = (v - nearest) * normal;

    delete[] p;

    return distance;
}

template<typename T>
Vertex<T> StannPointCloudManager<T>::fromID(int i){
    return Vertex<T>(
            this->m_points[i][0],
            this->m_points[i][1],
            this->m_points[i][2]);
}

template<typename T>
Plane<T> StannPointCloudManager<T>::calcPlane(const Vertex<T> &queryPoint,
        const int &k,
        const vector<unsigned long> &id)
{
    /**
     * @todo Think of a better way to code this magic number.
     */
    float epsilon = 100.0;

    Vertex<T> diff1, diff2;
    Normal<T> normal;

    T z1 = 0;
    T z2 = 0;

    // Calculate a least sqaures fit to the given points
    Vector3f C;
    VectorXf F(k);
    MatrixXf B(k,3);

    for(int j = 0; j < k; j++){
        F(j)    =  this->m_points[id[j]][1];
        B(j, 0) = 1.0f;
        B(j, 1) = this->m_points[id[j]][0];
        B(j, 2) = this->m_points[id[j]][2];
    }

    MatrixXf Bt = B.transpose();
    MatrixXf BtB = Bt * B;
    MatrixXf BtBinv = BtB.inverse();

    MatrixXf M = BtBinv * Bt;
    C = M * F;

    // Calculate to vectors in the fitted plane
    z1 = C(0) + C(1) * (queryPoint[0] + epsilon) + C(2) * queryPoint[2];
    z2 = C(0) + C(1) * queryPoint[0] + C(2) * (queryPoint[2] + epsilon);

    // Calculcate the plane's normal via the cross product
    diff1 = Vertex<T>(queryPoint[0] + epsilon, z1, queryPoint[2]) - queryPoint;
    diff2 = Vertex<T>(queryPoint[0], z2, queryPoint[2] + epsilon) - queryPoint;

    normal = diff1.cross(diff2);

    // Create a plane representation and return the result
    Plane<T> p;
    p.a = C(0);
    p.b = C(1);
    p.c = C(2);
    p.n = normal;
    p.p = queryPoint;

    return p;
}

template<typename T>
void StannPointCloudManager<T>::save(string filename)
{
    // Get file extension
    boost::filesystem::path selectedFile(filename);
    string extension = selectedFile.extension();

    // Try to load file by extension
    if(extension == ".ply")
    {
        savePLY(filename);
    }
    else if (extension == ".nor")
    {
        savePointsAndNormals(filename);
    }
    else if (extension == ".pts" || extension == ".3d" || extension == ".xyz")
    {
        savePoints(filename);
    }

}

template<typename T>
void StannPointCloudManager<T>::savePointsAndNormals(string filename)
{
    ofstream out(filename.c_str());

    if(!out.good())
    {
        cout << timestamp
             << " StannPointCloudManager::SavePointsAndNormals(): Could not open file "
             << filename << "." << endl;

        return;
    }

    string prefix = timestamp.getElapsedTime() + "Saving points and normals to '" + filename + "'.";
    ProgressCounter p(m_numPoints, prefix);

    for(size_t i = 0; i < m_numPoints; i++)
    {
        out << this->m_points[i][0]  << " " << this->m_points[i][1]  << " " << this->m_points[i][2] << " "
            << this->m_normals[i][0] << " " << this->m_normals[i][1] << " " << this->m_normals[i][2] << endl;
        ++p;
    }
    cout << endl;
}

template<typename T>
void StannPointCloudManager<T>::savePoints(string filename)
{
    ofstream out(filename.c_str());

    if(!out.good())
    {
        cout << timestamp
                << " StannPointCloudManager::SavePointsAndNormals(): Could not open file "
                << filename << "." << endl;

        return;
    }

    string prefix = timestamp.getElapsedTime() + "Saving points to '" + filename + "'.";
    ProgressCounter p(m_numPoints, prefix);

    for(size_t i = 0; i < m_numPoints; i++)
    {
        out << this->m_points[i][0] << " " << this->m_points[i][1] << " " << this->m_points[i][2] << endl;
        ++p;
    }

    cout << endl;
}

template<typename T>
void StannPointCloudManager<T>::savePLY(string filename)
{
    PLYIO ply_writer;

    // Create vertex element
    if(this->m_points)
    {
        PLYElement* vertex_element = new PLYElement("vertex", m_numPoints);
        vertex_element->addProperty("x", "float");
        vertex_element->addProperty("y", "float");
        vertex_element->addProperty("z", "float");
        ply_writer.addElement(vertex_element);
        ply_writer.setIndexedVertexArray(this->m_points, m_numPoints);
    }

    // Create normal element
    if(this->m_normals)
      {
          PLYElement* normal_element = new PLYElement("normal", m_numPoints);
          normal_element->addProperty("x", "float");
          normal_element->addProperty("y", "float");
          normal_element->addProperty("z", "float");
          ply_writer.addElement(normal_element);
          ply_writer.setIndexedVertexArray(this->m_normals, m_numPoints);
      }

    ply_writer.save(filename);
}

} // namespace lssr


