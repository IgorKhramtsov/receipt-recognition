#include <opencv2/imgcodecs.hpp>
#include "OcrServiceController.h"
#include "cpprest/filestream.h"
#include "../../src/Process.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "cpprest/producerconsumerstream.h"

//using namespace std;
using namespace utility;                    // Common utilities like string conversions
using namespace web;                        // Common features like URIs.
using namespace http;                  // Common HTTP functionality
using namespace http::client;          // HTTP client features
using namespace concurrency::streams;       // Asynchronous streams

OcrServiceController::OcrServiceController() { }

OcrServiceController::OcrServiceController(utility::string_t url) : listener(url) {
    listener.support(methods::GET, std::bind(&OcrServiceController::handle_get, this, std::placeholders::_1 ));
    listener.support(methods::POST, std::bind(&OcrServiceController::handle_post, this, std::placeholders::_1 ));
}

OcrServiceController::~OcrServiceController() {}

void OcrServiceController::handle_get(http_request request) {
    std::cout << request.to_string() << std::endl;

    std::vector<utility::string_t> path_els = uri::split_path(uri::decode(request.relative_uri().path()));
    utf8string body("status ok \n");
    request.reply(status_codes::OK, body);

}

void OcrServiceController::handle_post(http_request request) {
    std::cout << request.to_string() << std::endl;

    std::vector<utility::string_t> path_els = uri::split_path(uri::decode(request.relative_uri().path()));
    std::string body = "Status: Error";
    status_code code = status_codes::BadRequest;

    if(path_els[0] == "api") {

        if(path_els[1] == "v1") {
            setlocale(LC_ALL, "C");
            ocr = new tesseract::TessBaseAPI();
            ocr->Init(NULL, "rus", tesseract::OEM_LSTM_ONLY);
            ocr->SetPageSegMode(tesseract::PSM_AUTO_ONLY);

            if (path_els[2] == "file") {
                Concurrency::streams::producer_consumer_buffer<unsigned char> trg;
                size_t _size = request.body().read_to_end(trg).get();
                unsigned char *buffer = new unsigned char[_size];
                trg.scopy(buffer, _size);

                cv::Mat image = cv::imdecode(cv::Mat(1, _size, CV_8UC1, buffer), CV_LOAD_IMAGE_UNCHANGED);

                processMat(image);
                nlohmann::json recognizedText = getRecognized();


                body = recognizedText.dump();
                code = status_codes::OK;
            } else {
                body = "Status: Wrong method.";
            }

        } else {
            body = "Status: Wrong version.";
        }

    } else {
        body = "Status: Wrong uri!";
    }


    std::cout << "BODY: " << body.c_str();
    request.reply(code, body);
}