
#ifndef OCR_SERVICE_PROCESS_H
#define OCR_SERVICE_PROCESS_H

#include "tesseract/baseapi.h"
#include "nlohmann/json.hpp"
#include "opencv2/imgproc.hpp"

//using json = nlohmann::json;


const char windowName[] = "Main";
const char windowName_cutted[] = "Main Cutted";

extern tesseract::TessBaseAPI *ocr;

void recognize( int, void* );

int processFile(const char *path);
int processMat(cv::Mat image);

nlohmann::json getRecognized();

#endif //OCR_SERVICE_PROCESS_H
