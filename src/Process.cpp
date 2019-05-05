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

    imshow("Image original", image);
    imwrite("images/Image_original.jpg", image);
    // Используем морфологию для выделения объекта
    morphologic(image, image_morphed);

    // imshow("Image morphed", image_morphed);
    imwrite("images/Image_morphed.jpg", image_morphed);
    RotatedRect rect;
    // Получаем контур объекта на изображении для вырезки
    getContours(image_morphed, image_contours, rect);
    cutAndScale(image, image_cutted, rect);
    // imshow("Image cutted", image_cutted);
    imwrite("images/Image_cutted.jpg", image_cutted);


    // Шумоподавление из ОпенСВ
    fastNlMeansDenoising(image_cutted, image_cutted);
    // imshow("Image fastNlMeansDenoising", image_cutted);
    imwrite("images/Image_fastNlMeansDenoising.jpg", image_cutted);

    //
    Mat nonUniformIll = image_cutted.clone();
    nonUniformIlluminationMorph(image_cutted, nonUniformIll);
    // imshow("Image nonUniformIlluminationMorph", nonUniformIll);
    imwrite("images/Image_nonUniformIlluminationMorph.jpg", nonUniformIll);

    Mat crcb = nonUniformIll.clone();
    filteringCrCb(nonUniformIll, crcb);
    // imshow("Image filteringCrCb", crcb);
    imwrite("images/Image_filteringCrCb.jpg", crcb);

    Mat grayWorldFiltred = crcb.clone();
    grayWorldFilter(crcb, grayWorldFiltred);
    // imshow("Image grayWorldFilter", grayWorldFiltred);
    imwrite("images/Image_grayWorldFilter.jpg", grayWorldFiltred);

    image_corrected = grayWorldFiltred.clone();
    gamma_correction(grayWorldFiltred, image_corrected);
    // imshow("Image gamma_correction", image_corrected);
    imwrite("images/Image_gamma_correction.jpg", image_corrected);
    brightness_and_contrast_correction(image_corrected, image_corrected);
    // imshow("Image brightness_and_contrast_correction", image_corrected);
    imwrite("images/Image_brightness_and_contrast_correction.jpg", image_corrected);

    image_sharpened = image_corrected.clone();
    sharpness(image_sharpened, image_sharpened, 6, 51, 2);
    // imshow("Image sharpness", image_sharpened);
    imwrite("images/Image_sharpness.jpg", image_sharpened);
    adaptiveThreshold(image_sharpened, image_sharpened, 255., ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 81, 60);
    // imshow("Image adaptiveThreshold", image_sharpened);
    imwrite("images/Image_adaptiveThreshold.jpg", image_sharpened);

    deSkew(image_sharpened, image_deskew);
    // imshow("Image deSkew",  image_deskew);
    imwrite("images/Image_deSkew.jpg", image_deskew);

    blur(image_deskew, image_deskew, Size(5,5));
    // imshow("Image blur",  image_deskew);
    imwrite("images/Image_blur.jpg", image_deskew);
    Mat morphStructure = getStructuringElement(MORPH_ELLIPSE, Size(2, 4));
    morphologyEx(image_deskew, image_deskew, MORPH_CLOSE, morphStructure);
    // imshow("Image morphologyEx",  image_deskew);
    imwrite("images/Image_morphologyEx.jpg", image_deskew);

    imshow("Corrected", image_deskew);
    imwrite("images/Image_result.jpg", image_deskew);
    // recognize(NULL, NULL);
    //cv::imshow("Recognized", image_deskew);
}

nlohmann::json getRecognized() {
    return recognized;
}
