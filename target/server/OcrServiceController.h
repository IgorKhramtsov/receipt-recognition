//
// Created by user on 15.08.18.
//

#ifndef OCR_SERVICE_OCRSERVICECONTROLLER_H
#define OCR_SERVICE_OCRSERVICECONTROLLER_H

#include <cpprest/http_listener.h>

using namespace web;
using namespace http;
using namespace http::experimental::listener;

class OcrServiceController {
public:
    OcrServiceController();
    OcrServiceController(utility::string_t url);
    virtual ~OcrServiceController();

    pplx::task<void>open() { return  listener.open();}
    pplx::task<void>close() { return  listener.close();}

private:
    void handle_get(http_request request);
    void handle_post(http_request request);
    http_listener listener;

};


#endif //OCR_SERVICE_OCRSERVICECONTROLLER_H
