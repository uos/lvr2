//
// Created by Nikolas on 13.09.22.
//

#include "lvr2/io/modelio/RdbxIO.hpp"


#include <riegl/rdb.hpp>
#include <riegl/rdb/default.hpp>

#include <array>
#include <vector>
#include <cstdint>
#include <iostream>
#include <exception>


namespace lvr2
{
    RdbxIO::RdbxIO()
    {

    }
    RdbxIO::~RdbxIO()
    {

    }

    /**
     * @brief Creates a new rdb database and
     * write its content into @filename, currently only
     * x,y,z and reflectance
     *
     * @param filename
     */
    void RdbxIO::save(string filename)
    {
        // checks for validity
        if (!m_model)
        {
            std::cerr << "No model set for export!" << std::endl;
            return;
        }

        if ( !this->m_model->m_pointCloud ) {
            std::cerr << "No point buffer available for output." << std::endl;
            return;
        }

        floatArr m_points;
        floatArr m_pointReflectance;

        // Get buffers
        if ( m_model->m_pointCloud )
        {
            PointBufferPtr pointBuffer = m_model->m_pointCloud;
        }

        //RDB code to read the data
        try
        {
            // New RDB library context
            riegl::rdb::Context context;

            // New database instance
            riegl::rdb::Pointcloud rdb(context);

            // Step 1: Create new point cloud database
            {
                // This object contains all settings that are required
                // to create a new RDB point cloud database.
                riegl::rdb::pointcloud::CreateSettings settings;

                // Define primary point attribute, usually the point coordinates
                // details see class riegl::rdb::pointcloud::PointAttribute
                settings.primaryAttribute.name         = "riegl.xyz";
                settings.primaryAttribute.title        = "XYZ";
                settings.primaryAttribute.description  = "Cartesian point coordinates";
                settings.primaryAttribute.unitSymbol   = "m";
                settings.primaryAttribute.length       =  3;
                settings.primaryAttribute.resolution   =  0.00025;
                settings.primaryAttribute.minimumValue = -535000.0; // minimum,
                settings.primaryAttribute.maximumValue = +535000.0; //   maximum and
                settings.primaryAttribute.defaultValue =       0.0; //     default in m
                settings.primaryAttribute.storageClass = riegl::rdb::pointcloud::PointAttribute::VARIABLE;

                // Define database settings
                settings.chunkMode = riegl::rdb::pointcloud::CreateSettings::POINT_COUNT;
                settings.chunkSize = 50000; // maximum number of points per chunk
                settings.compressionLevel = 10; // 10% compression rate

                // Finally create new database
                rdb.create(filename + ".rdbx", settings);
            }

            // Step 2: Define some additional point attributes
            //         Please note that there is also a shortcut for built-in
            //         RIEGL default point attributes which we use to define
            //         the "riegl.class" attribute at the end of this block.
            {
                // Before we can modify the database, we must start a transaction
                riegl::rdb::pointcloud::TransactionScope transaction(rdb,
                       "Initialization",      // transaction title
                       "Save rdbx" // software name
                );

                // Target surface reflectance
                {
                    riegl::rdb::pointcloud::PointAttribute attribute;
                    //
                    attribute.name         = "riegl.reflectance";
                    attribute.title        = "Reflectance";
                    attribute.description  = "Target surface reflectance";
                    attribute.unitSymbol   = "dB";
                    attribute.length       =  1;
                    attribute.resolution   =    0.010;
                    attribute.minimumValue = -100.000; // minimum,
                    attribute.maximumValue = +100.000; //   maximum and
                    attribute.defaultValue =    0.000; //     default in dB
                    attribute.storageClass = riegl::rdb::pointcloud::PointAttribute::CONSTANT;
                    //
                    rdb.pointAttribute().add(attribute);
                }
                /** // Point color
                {
                    riegl::rdb::pointcloud::PointAttribute attribute;
                    //
                    attribute.name         = "riegl.rgba";
                    attribute.title        = "True Color";
                    attribute.description  = "Point color acquired by camera";
                    attribute.unitSymbol   = ""; // has no unit
                    attribute.length       =   4;
                    attribute.resolution   =   1.000;
                    attribute.minimumValue =   0.000;
                    attribute.maximumValue = 255.000;
                    attribute.defaultValue = 255.000;
                    attribute.storageClass = riegl::rdb::pointcloud::PointAttribute::VARIABLE;
                    //
                    rdb.pointAttribute().add(attribute);
                }
                // Point classification - by using a shortcut for built-in RIEGL attributes:
                {
                    rdb.pointAttribute().add("riegl.class");
                }
                // Echo amplitude - by using the constant from "riegl/rdb/default.hpp"
                {
                    rdb.pointAttribute().add(riegl::rdb::pointcloud::RDB_RIEGL_AMPLITUDE);
                }**/

                // Finally commit transaction
                transaction.commit();
            }
        }
        catch(const riegl::rdb::Error &error)
        {
            std::cerr << error.what() << " (" << error.details() << ")" << std::endl;
        }
        catch(const std::exception &error)
        {
            std::cerr << error.what() << std::endl;
        }
    }


    void RdbxIO::save(ModelPtr model, string filename)
    {
        m_model = model;
        save(filename);
    }


    // ID -> x,y,z -> amplitude -> reflectance -> r,g,b -> pointclass?
    ModelPtr RdbxIO::read(string filename)
    {
        ModelPtr model (new Model);
        PointBufferPtr pointBuffer(new PointBuffer);

        try {
            // New RDB library context
            riegl::rdb::Context context;

            // Access existing database
            riegl::rdb::Pointcloud rdb(context);
            riegl::rdb::pointcloud::OpenSettings settings;
            rdb.open(filename, settings);

            // Prepare point attribute buffers
            static const uint32_t BUFFER_SIZE = 10000;
            std::vector<std::array<double, 3> > bufferCoordinates(BUFFER_SIZE);
            std::vector<float> bufferReflectance(BUFFER_SIZE);

            // Select query with empty filter to get all Points
            riegl::rdb::pointcloud::QuerySelect select = rdb.select("");

            // variable to count points
            uint32_t numPoints = 0;

            // Counts the number of points
            // could or should be improved
            while (const uint32_t count = select.next(BUFFER_SIZE)) {
                // iterating over all points to count them
                for (uint32_t i = 0; i < count; i++) {
                    ++numPoints;
                }
            }

            // arrays to store point coordinates and their reflectance
            float *pointArray = new float[3 * numPoints];
            float *reflectanceArray = new float[numPoints];

            // variable to keep track of current point,
            // since i in following for loop is reset after BUFFER_SIZE=10000 steps
            uint32_t currPoint = 0;

            while (const uint32_t count = select.next(BUFFER_SIZE)) {
                // Print points to output stream
                for (uint32_t i = 0; i < count; i++) {
                    // filling arrays with data from rdbx file
                    pointArray[3 * currPoint] = bufferCoordinates[i][0];
                    pointArray[3 * currPoint + 1] = bufferCoordinates[i][1];
                    pointArray[3 * currPoint + 2] = bufferCoordinates[i][2];
                    reflectanceArray[currPoint] = bufferReflectance[i];
                    ++currPoint;
                }
            }

            // parsing float Arrays to floatArr
            floatArr parr(pointArray);
            floatArr rarr(reflectanceArray);

            // filling PointBuffer with data
            pointBuffer->setPointArray(parr, numPoints);
            pointBuffer->addFloatChannel(rarr, "reflectance");

            // adding pointBuffer to model
            model->m_pointCloud = pointBuffer;
            return model;
        }

        catch(const riegl::rdb::Error &error)
        {
            std::cerr << error.what() << " (" << error.details() << ")" << std::endl;
            return 1; // error
        }
        catch(const std::exception &error)
        {
            std::cerr << error.what() << std::endl;
            return 1; // error
        }

    }

} // lvr2