cmake_minimum_required(VERSION 3.12)
project(OCR_service_server)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ldl -lssl -lcrypto")
set(CMAKE_EXE_LINKER_FLAGS "-lboost_system -lpthread ")
find_package(OpenSSL REQUIRED)
find_package( Boost COMPONENTS thread system filesystem REQUIRED ) #whatever libs you need
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories (../../3dparty)
find_package( Threads )
FIND_PACKAGE( OpenCV REQUIRED)
#find_package( tesseract REQUIRED )

set(OPENSSL_LIBRARIES "${OPENSSL_LIBRARIES};${CMAKE_DL_LIBS};${CMAKE_THREAD_LIBS_INIT}")
set( LIBS_TO_LINK
        ${Boost_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${OPENSSL_LIBRARIES}
        ${OpenCV_LIBS}
        tesseract
        cpprest
        )

include_directories (/usr/local/include)
link_directories (/usr/local/lib)
#find_package( tesseract REQUIRED )

add_executable(OCR_service_server main.cpp ../../src/image_utils.cpp ../../src/image_utils.h ../../src/Process.cpp ../../src/Process.h OcrServiceController.h OcrServiceController.cpp)
target_link_libraries( OCR_service_server
        ${LIBS_TO_LINK}
        )
