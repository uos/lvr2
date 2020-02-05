#include "ascii_viewer/AsciiRenderer.hpp"
#include <iostream>
#include <locale>
#include <chrono>

namespace lvr2 {

void AsciiRendererErrorFunc(void* userPtr, enum RTCError error, const char* str)
{
    printf("error %d: %s\n", error, str);
}

AsciiRenderer::AsciiRenderer(
    MeshBufferPtr mesh, 
    const unsigned int& num_grays)
:m_ascii_width(0)
,m_ascii_height(0)
,m_num_grays(num_grays)
,m_chars(nullptr)
,m_colors(nullptr)
{
    std::locale::global(std::locale("en_US.utf8"));
    std::wcout.imbue(std::locale());
    init_curses(num_grays);

    m_camera.setIdentity();

    initBuffers();
    initEmbree(mesh);
    clear();

}

AsciiRenderer::~AsciiRenderer()
{
    destroyBuffers();
    destroyEmbree();
    destroy_curses();
}

void AsciiRenderer::initBuffers()
{
    unsigned int W = curses_width();
    unsigned int H = curses_height();

    if(W != m_ascii_width || H != m_ascii_height)
    {
        destroyBuffers();
        
        m_ascii_width = W;
        m_ascii_height = H;

        m_chars = static_cast<BrailleChar**>( malloc( m_ascii_width * sizeof(BrailleChar*) ) );
    
        for(int i=0; i<m_ascii_width; i++)
        {
            m_chars[i] = static_cast<BrailleChar*>(malloc( m_ascii_height * sizeof(BrailleChar) ) );
        }

        m_colors = static_cast<float**>( malloc( m_ascii_width * sizeof(float*) ) );
    
        for(int i=0; i<m_ascii_width; i++)
        {
            m_colors[i] = static_cast<float*>(malloc( m_ascii_height * sizeof(float) ) );
        }
    }
}

void AsciiRenderer::destroyBuffers()
{
    if(m_chars)
    {
        for(int i=0; i<m_ascii_width; i++)
        {
            free(m_chars[i]);
        }
        free(m_chars);
    }

    if(m_colors)
    {
        for(int i=0; i<m_ascii_width; i++)
        {
            free(m_colors[i]);
        }
        free(m_colors);
    }
}

void AsciiRenderer::initEmbree(MeshBufferPtr mesh)
{
    m_device = initializeDevice();
    m_scene = initializeScene(m_device, mesh);
    rtcInitIntersectContext(&m_context);
}

RTCDevice AsciiRenderer::initializeDevice()
{
    RTCDevice device = rtcNewDevice(NULL);

    if (!device)
    {
        std::cerr << "error " << rtcGetDeviceError(NULL) << ": cannot create device" << std::endl;
    }

    rtcSetDeviceErrorFunction(device, AsciiRendererErrorFunc, NULL);
    return device;
}

RTCScene AsciiRenderer::initializeScene(
    RTCDevice device,
    const MeshBufferPtr mesh)
{
    RTCScene scene = rtcNewScene(device);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    auto lvr_vertices = mesh->getVertices();
    int num_vertices = mesh->numVertices();

    auto lvr_indices = mesh->getFaceIndices();
    int num_faces = mesh->numFaces();

    float* embree_vertices = (float*) rtcSetNewGeometryBuffer(geom,
                                                        RTC_BUFFER_TYPE_VERTEX,
                                                        0,
                                                        RTC_FORMAT_FLOAT3,
                                                        3*sizeof(float),
                                                        num_vertices);

    unsigned* embree_indices = (unsigned*) rtcSetNewGeometryBuffer(geom,
                                                            RTC_BUFFER_TYPE_INDEX,
                                                            0,
                                                            RTC_FORMAT_UINT3,
                                                            3*sizeof(unsigned),
                                                            num_faces);

    if (embree_vertices && embree_indices)
    {
        // copy mesh to embree buffers
        const float* lvr_vertices_begin = lvr_vertices.get();
        const float* lvr_vertices_end = lvr_vertices_begin + num_vertices * 3;
        std::copy(lvr_vertices_begin, lvr_vertices_end, embree_vertices);

        const unsigned int* lvr_indices_begin = lvr_indices.get();
        const unsigned int* lvr_indices_end = lvr_indices_begin + num_faces * 3;
        std::copy(lvr_indices_begin, lvr_indices_end, embree_indices);
    }

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    return scene;
}

void AsciiRenderer::destroyEmbree()
{
    rtcReleaseScene(m_scene);
    rtcReleaseDevice(m_device);
}

void AsciiRenderer::set(const unsigned int& x, const unsigned int& y)
{
    unsigned int x_outer = x / 2;
    unsigned int x_inner = x % 2;

    unsigned int y_outer = y / 4;
    unsigned int y_inner = y % 4;

    m_chars[x_outer][y_outer].set(x_inner, y_inner);
}

void AsciiRenderer::clear(const unsigned int& x, const unsigned int& y)
{
    unsigned int x_outer = x / 2;
    unsigned int x_inner = x % 2;

    unsigned int y_outer = y / 4;
    unsigned int y_inner = y % 4;

    m_chars[x_outer][y_outer].clear(x_inner, y_inner);
}

void AsciiRenderer::clear()
{
    for(int i=0; i<m_ascii_width; i++)
    {
        for(int j=0; j<m_ascii_height; j++)
        {
            m_chars[i][j].data = 0;
            m_colors[i][j] = 0.0;
        }
    }
}

void AsciiRenderer::raycast()
{
    unsigned int image_width = m_ascii_width * 2;
    unsigned int image_height = m_ascii_height * 4;

    double c_u = image_width / 2;
    double c_v = image_height / 2;

    float focal_length = 700.0;

    double fx = focal_length;
    double fy = focal_length;


    Vector3d origin;
    origin = m_camera.translation();

    #pragma omp parallel for
    for(int u = 0; u < image_width; u++)
    {
        for(int v = 0; v < image_height; v++)
        {
            RTCRayHit rayhit;
            rayhit.ray.org_x = origin.coeff(0);
            rayhit.ray.org_y = origin.coeff(1);
            rayhit.ray.org_z = origin.coeff(2);
            rayhit.ray.tnear = 0;
            rayhit.ray.tfar = INFINITY;
            rayhit.ray.mask = 0;
            rayhit.ray.flags = 0;
            rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

            Vector3d dir(1.0, -(static_cast<double>(u) - c_u ) / fx, -(static_cast<double>(v) - c_v) / fy);

            dir.normalize();

            dir = m_camera.linear() * dir;

            rayhit.ray.dir_x = dir.coeff(0);
            rayhit.ray.dir_y = dir.coeff(1);
            rayhit.ray.dir_z = dir.coeff(2);


            rtcIntersect1(m_scene, &m_context, &rayhit);
            
            if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
            {
                float norm_normal = sqrt( pow(rayhit.hit.Ng_x, 2.0) 
                            + pow(rayhit.hit.Ng_y,2.0) 
                            + pow(rayhit.hit.Ng_z,2.0));

                rayhit.hit.Ng_x /= norm_normal;
                rayhit.hit.Ng_y /= norm_normal;
                rayhit.hit.Ng_z /= norm_normal;

                float scalar = rayhit.ray.dir_x * rayhit.hit.Ng_x 
                            + rayhit.ray.dir_y * rayhit.hit.Ng_y 
                            + rayhit.ray.dir_z * rayhit.hit.Ng_z;

                unsigned int ascii_x = u / 2;
                unsigned int ascii_y = v / 4;
                m_colors[ascii_x][ascii_y] += std::max(0.1f, fabs(scalar)) / 8.0;
                set(u, v);   
            } else {
                clear(u, v);
            }
        }
    }
}

void AsciiRenderer::processKey(int key)
{

    if(key == 'a') {
        // trans left
        m_camera.translate(Eigen::Vector3d(0.0, m_vrot * m_sidereduce, 0.0));
    } else if( key == 'A') {
        m_camera.translate(Eigen::Vector3d(0.0, m_vtrans * m_boost * m_sidereduce, 0.0));
    } else if( key == 'w') {
        // trans front
        m_camera.translate(Eigen::Vector3d(m_vtrans, 0.0, 0.0));
    } else if( key == 'W') {
        // trans front
        m_camera.translate(Eigen::Vector3d(m_vtrans * m_boost, 0.0, 0.0));
    } else if( key == 's') {
        // trans back
        m_camera.translate(Eigen::Vector3d(-m_vtrans, 0.0, 0.0));
    } else if( key == 'S') {
        // trans back
        m_camera.translate(Eigen::Vector3d(-m_vtrans * m_boost, 0.0, 0.0));
    } else if( key == 'd') {
        // trans right
        m_camera.translate(Eigen::Vector3d(0.0, -m_vtrans * m_sidereduce, 0.0));
    } else if( key == 'D') {
        m_camera.translate(Eigen::Vector3d(0.0, -m_vtrans * m_boost * m_sidereduce, 0.0));
    } else if( key == 'q' ) {
        float roll = -m_vrot;
        m_camera.rotate(Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
    } else if( key == 'Q' ) {
        float roll = -m_vrot;
        m_camera.rotate(Eigen::AngleAxisd(roll * m_boost, Eigen::Vector3d::UnitX()));
    } else if( key == 'e' ) {
        float roll = m_vrot;
        m_camera.rotate(Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
    } else if( key == 'E' ) {
        float roll = m_vrot;
        m_camera.rotate(Eigen::AngleAxisd(roll * m_boost, Eigen::Vector3d::UnitX()));
    } else if( key == KEY_LEFT ) {
        // yaw + 
        float yaw = m_vrot;
        m_camera.rotate(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    } else if( key == KEY_RIGHT ) {
        // yaw -
        float yaw = -m_vrot;
        m_camera.rotate(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    } else if( key == KEY_UP ) {
        float pitch = -m_vrot;
        m_camera.rotate(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));
    } else if( key == KEY_DOWN ) {
        float pitch = m_vrot;
        m_camera.rotate(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));
    } else if( key == ' ') {
        m_track_mouse = !m_track_mouse;
    } else if( key == KEY_MOUSE ) {
        // rotate
        MEVENT event;
        
        if(getmouse(&event) == OK)
        {

            if(event.bstate & BUTTON1_PRESSED) // LEFT KLICK
            {
                // record start point
                // from now on we need to track the rotation
                m_track_mouse = true;
            } else if(event.bstate & MOUSE_SCROLL_UP) {
                // SCROLL
                m_camera.translate(Eigen::Vector3d(m_vtrans, 0.0, 0.0));
            } else if( (event.bstate & MOUSE_MOVE) && m_track_mouse) {
                double mouse_x = static_cast<double>(event.x);
                double mouse_y = static_cast<double>(event.y);

                double delta_x = m_mouse_x - mouse_x;
                double delta_y = m_mouse_y - mouse_y;

                // sometimes SIGINTS here i guess. maybe use quaternions instead?
                m_camera.rotate(
                    Eigen::AngleAxisd(-delta_y * m_vrot, Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(delta_x * m_vrot, Eigen::Vector3d::UnitZ())
                );

            } else {
                m_track_mouse = false;
            }

            m_mouse_x = static_cast<double>(event.x);
            m_mouse_y = static_cast<double>(event.y);
        }
    } else if(key == KEY_RESIZE) {
        initBuffers();
    } else if(key == 'i') {
        m_vtrans *= 5.0;
    } else if(key == 'k') {
        m_vtrans /= 5.0;
    } else if(key == 'h') {
        m_show_controls = !m_show_controls;
    } 

    move(0, 0);
}

void AsciiRenderer::render()
{
    auto start = std::chrono::steady_clock::now();
    initBuffers();
    clear();
    raycast();

    unsigned int W = m_ascii_width;
    unsigned int H = m_ascii_height;
    
    for(unsigned int y=0; y< H; y++)
    {
        for(unsigned int x=0; x< W; x++)
        {
            console_print_grayscale(y,x, m_chars[x][y].getChar(), m_colors[x][y]);
        }
    }

    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    
    // max fps 1000
    double fps = 1000.0 / 0.001;
    if(ms > 0.001)
    {
        fps = 1000.0 / ms;
    }

    // update avg fps
    m_avg_fps = m_fps_alpha * m_avg_fps + (1.0 - m_fps_alpha) * fps;

    mvprintw(0, 0, "%4d fps", int(m_avg_fps) );
    
    printControls();
    
    refresh();
}

void AsciiRenderer::consolePrintMessage(std::string message)
{
    m_messages.push_back(message);
}

void AsciiRenderer::consoleReleaseMessages()
{
    unsigned int H = curses_height();
    move(H-1, 0);

    for(auto message : m_messages)
    {
        addstr(message.c_str());
        addch(' ');
    }

    m_messages.clear();
}

void AsciiRenderer::printControls()
{
    unsigned int H = curses_height();

    if(m_show_controls)
    {
        mvprintw(H-8, 0, "CONTROLLS (H)");
        mvprintw(H-7, 0, "Translate: W A S D Scroll");
        mvprintw(H-6, 0, "Rotate: ← ↑ → ↓ q e");
        mvprintw(H-5, 0, "Mouse control:");
        mvprintw(H-4, 0, "  - ON: Space, Left Click");
        mvprintw(H-3, 0, "  - OFF: Space, Other Click");
        mvprintw(H-2, 0, "Speed inc/dec: i k");
        mvprintw(H-1, 0, "Exit: Esc");
    } else {
        mvprintw(H-1, 0, "CONTROLLS (H)");
    }
}

unsigned int AsciiRenderer::width()
{
    return m_ascii_width * 2;
}

unsigned int AsciiRenderer::height()
{
    return m_ascii_height * 4;
}

} // namespace lvr2
