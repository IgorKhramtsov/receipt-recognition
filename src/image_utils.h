#ifndef A000_OCR_SERVICE_IMAGE_UTILS_H
#define A000_OCR_SERVICE_IMAGE_UTILS_H


#include <tesseract/baseapi.h>
#include "opencv2/imgproc.hpp"
#include "nlohmann/json.hpp"

void gamma_correction(cv::Mat &in, cv::Mat &out);
void brightness_and_contrast_correction(cv:: Mat &in, cv::Mat &out);
void clustering(cv::Mat &in, cv::Mat &out);
void gaussianBlur(cv::Mat &in, cv::Mat &out);
void grabCut(cv::Mat &in, cv::Mat &out);
void threshold(cv::Mat &in, cv::Mat &out);
void morphologic(cv::Mat &in, cv::Mat &out);
void getContours(cv::Mat &in, cv::Mat &contour_image, cv::RotatedRect &rect);
void findTextContours(cv::Mat &in, std::vector<std::vector<cv::Point>> &contours);
void cutAndScale(cv::Mat &in, cv::Mat &out, cv::RotatedRect &rect);
void equalizeIntensity(const cv::Mat& in, cv::Mat& out);
void colorBalance(cv::Mat& _inImg, cv::Mat& _outImg, float percent);
void grayWorldFilter(const cv::Mat& _src, cv::Mat& _dst);
void nonUniformIlluminationMorph(const cv::Mat &_src, cv::Mat &_dst);
void clahe(const cv::Mat &_src, cv::Mat &_dst);
void filteringCrCb(const cv::Mat &_src, cv::Mat &_dst);
void sharpness(const cv::Mat &src, cv::Mat &dst, float sigma, float threshold, float amount);
void deSkew(const cv::Mat &in, cv::Mat &dst);

void recognizeImage(const cv::Mat &img, tesseract::TessBaseAPI *ocr, nlohmann::json &rec);

#endif //A000_OCR_SERVICE_IMAGE_UTILS_H
