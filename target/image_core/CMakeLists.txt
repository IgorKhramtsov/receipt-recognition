cmake_minimum_required(VERSION 3.12)
project(OCR_service)

set(CMAKE_CXX_STANDARD 14)
#set(OpenCV_FOUND 1)

include_directories (/usr/local/include)
include_directories (../../3dparty)
link_directories (/usr/local/lib)


FIND_PACKAGE( OpenCV REQUIRED)
#find_package( tesseract REQUIRED )

add_executable(OCR_service main.cpp
        ../../src/image_utils.cpp
        ../../src/image_utils.h
        main.cpp
        ../../src/Process.cpp
        ../../src/Process.h
        ../../3dparty/nlohmann/json.hpp
        )
target_link_libraries(OCR_service ${OpenCV_LIBS})
target_link_libraries(OCR_service tesseract)
