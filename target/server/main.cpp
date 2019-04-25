#include <iostream>
#include <dirent.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "tesseract/baseapi.h"
#include <leptonica/allheaders.h>
#include <opencv/cv.hpp>
#include "../../src/Process.h"
#include <cpprest/http_client.h>
#include <cpprest/http_listener.h>
#include "OcrServiceController.h"
#include <iostream>

//using namespace cv;
using namespace std;

tesseract::TessBaseAPI *ocr;

OcrServiceController *controller;

void on_initialize(const char* address) {
    uri_builder uri(address);
    utility::string_t url = uri.to_uri().to_string().c_str();


    controller = new OcrServiceController(url);
    controller->open().wait();

    std::cout << "Listening for requests at " << url << ": " << std::endl;
}

void on_shutdown() {
    controller->close().wait();
}

int main( int argc, char ** argv ) {
    const char *address = "http://127.0.0.1:6502";

    on_initialize(address);
    std::cout << "Press any to exit." << std::endl;

    getchar();
    on_shutdown();

    return 0;
}