#ifndef LVR2_ASCII_RENDERER_HPP
#define LVR2_ASCII_RENDERER_HPP

#include "lvr2/types/MatrixTypes.hpp"
#include "Braille.hpp"
#include "BitField.hpp"
#include <ncursesw/ncurses.h>
#include <unordered_map>
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/types/MatrixTypes.hpp"

#include "CursesHelper.hpp"

#include <embree3/rtcore.h>
#include <memory>

namespace lvr2 {


void AsciiRendererErrorFunc(void* userPtr, enum RTCError error, const char* str);

class AsciiRenderer {
public:
    AsciiRenderer(
        MeshBufferPtr mesh, 
        const unsigned int& num_grays = 200
    );
    ~AsciiRenderer();

    void set(const unsigned int& i, const unsigned int& j);
    void clear(const unsigned int& i, const unsigned int& j);
    void clear();


    
    void render();
    void processKey(int key);

    unsigned int width();
    unsigned int height();

protected:

    void raycast();

private:
    
    
    void initBuffers();
    void destroyBuffers();
    void initEmbree(MeshBufferPtr mesh);
    void destroyEmbree();
    RTCScene initializeScene(
        RTCDevice device,
        const MeshBufferPtr mesh);
    RTCDevice initializeDevice();

    void consolePrintMessage(std::string message);
    void consoleReleaseMessages();

    void printControls();

    std::vector<std::string> m_messages;

    unsigned int m_ascii_width;
    unsigned int m_ascii_height; 
    unsigned int m_num_grays;

    WINDOW* m_window;

    BrailleChar** m_chars;

    float** m_colors;

    RTCDevice m_device;
    RTCScene m_scene;
    RTCIntersectContext m_context;

    // Camera
    Eigen::Affine3d m_camera;
    
    // Dynamic cam intrinsics
    // Button and Keyboard

    bool m_track_mouse = false;
    double m_mouse_x;
    double m_mouse_y;

    // show controls
    bool m_show_controls = false;

    // controls params
    double m_vrot = 0.005;
    double m_vtrans = 0.5;
    double m_sidereduce = 0.8;
    double m_boost = 3.0;

    double m_avg_fps = 20;
    double m_fps_alpha = 0.8;

};

using AsciiRendererPtr = std::shared_ptr<AsciiRenderer>;

} // namespace lvr2



#endif // LVR2_ASCII_RENDERER_HPP
