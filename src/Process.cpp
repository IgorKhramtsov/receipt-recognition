#include <iostream>
#include <dirent.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "image_utils.h"
#include "tesseract/baseapi.h"
#include <leptonica/allheaders.h>
#include <opencv/cv.hpp>
#include "Process.h"

using namespace cv;
using namespace std;

Mat image;
Mat image_contours;
Mat image_cutted;
Mat image_corrected;
Mat image_morphed;
Mat image_sharpened;
Mat image_deskew;

nlohmann::json recognized;

void recognize( int, void* ) {
    recognizeImage(image_deskew, ocr, recognized);
}

int processFile(const char *path) {
    Mat image = imread(path);

    return processMat(image);
}

int processMat(Mat image) {

    morphologic(image, image_morphed);

    RotatedRect rect;
    getContours(image_morphed, image_contours, rect);
    cutAndScale(image, image_cutted, rect);

    fastNlMeansDenoising(image_cutted, image_cutted);
//    imshow(windowName_cutted, image_cutted);

    Mat nonUniformIll = image_cutted.clone();
    nonUniformIlluminationMorph(image_cutted, nonUniformIll);

    Mat crcb = nonUniformIll.clone();
    filteringCrCb(nonUniformIll, crcb);

    Mat grayWorldFiltred = crcb.clone();
    grayWorldFilter(crcb, grayWorldFiltred);

    image_corrected = grayWorldFiltred.clone();
    gamma_correction(grayWorldFiltred, image_corrected);
    brightness_and_contrast_correction(image_corrected, image_corrected);

    image_sharpened = image_corrected.clone();
    sharpness(image_sharpened, image_sharpened, 6, 51, 2);
    adaptiveThreshold(image_sharpened, image_sharpened, 255., ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 81, 60);

    deSkew(image_sharpened, image_deskew);

    blur(image_deskew, image_deskew, Size(5,5));
    Mat morphStructure = getStructuringElement(MORPH_ELLIPSE, Size(2, 4));
    morphologyEx(image_deskew, image_deskew, MORPH_CLOSE, morphStructure);

    cv::imshow("Corrected", image_deskew);
    recognize(NULL, NULL);
}

nlohmann::json getRecognized() {
    return recognized;
}