#include <iostream>
#include <dirent.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "../../src/image_utils.h"
#include "tesseract/baseapi.h"
#include <leptonica/allheaders.h>
#include <opencv/cv.hpp>
#include "../../src/Process.h"

using namespace cv;
using namespace std;

tesseract::TessBaseAPI *ocr;

int main( int argc, char ** argv ) {
    const char *dirname = argc >= 2 ? argv[1] : "../../sample_images/";

    // namedWindow(windowName_cutted, WINDOW_NORMAL);
    // moveWindow(windowName_cutted, 50, 50);
    // resizeWindow(windowName_cutted, 500, 800);

    namedWindow("Corrected", WINDOW_NORMAL);
    resizeWindow("Corrected", 500, 800);
    moveWindow("Corrected", 50, 80);

    // namedWindow("Recognized", WINDOW_NORMAL);
    // resizeWindow("Recognized", 500, 800);
    // moveWindow("Recognized", 50, 80);

    createButton("Recognize", recognize);

    setlocale(LC_ALL, "C");
    ocr = new tesseract::TessBaseAPI();
    ocr->Init(NULL, "rus", tesseract::OEM_LSTM_ONLY);
    ocr->SetPageSegMode(tesseract::PSM_AUTO_ONLY);

    DIR *dir;
    class dirent *ent;
    dir = opendir(dirname);
    while ((ent = readdir(dir)) != NULL) {
        String name = ent->d_name;
        if( !(name.length() > 0 && name.find('.') > 0) )
            continue;

        std::cout << name << std::endl;
        processFile((dirname + name).c_str());

        waitKey(0);
    }

    return 0;
}
