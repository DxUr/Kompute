cmake_minimum_required(VERSION 3.5.0)
project(compute VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libavutil libswscale libavfilter libavdevice)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


file(GLOB_RECURSE SOURCES
"src/*.cpp"
)

add_executable(compute ${SOURCES})

target_include_directories(compute PUBLIC include)
target_compile_options(compute PUBLIC -fsanitize=address -g)
target_link_options(compute PUBLIC -fsanitize=address -g)
target_link_libraries(compute
    ${FFMPEG_LIBRARIES}    
    m
    gbm
    drm
    EGL
    GL
    GLEW
    opencv_core
    opencv_dnn
    opencv_imgproc
    opencv_highgui
)
