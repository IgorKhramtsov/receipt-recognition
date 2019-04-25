#include <iostream>
#include <dirent.h>
#include <cpprest/http_client.h>
#include <iostream>
#include <cpprest/filestream.h>
#include <cpprest/producerconsumerstream.h>
#include "nlohmann/json.hpp"

using namespace std;
using namespace utility;                    // Common utilities like string conversions
using namespace web;                        // Common features like URIs.
using namespace web::http;                  // Common HTTP functionality
using namespace web::http::client;          // HTTP client features
using namespace concurrency::streams;       // Asynchronous streams
using namespace concurrency;

pplx::task<void> uploadFileToHttpServerAsync(const char *fileName, const char *uri) {
    using concurrency::streams::file_stream;
    using concurrency::streams::basic_istream;

    return file_stream<unsigned char>::open_istream(fileName).then([=](pplx::task<basic_istream<unsigned char>> previousTask){
        try {
            auto fileStream = previousTask.get();

            http_client client(uri);
            return client.request(methods::POST, "file", fileStream).then([fileStream](pplx::task<http_response> previousTask) {
                fileStream.close();
                auto response = previousTask.get();
                std::cout << "Server response code: " << response.status_code() << "." << std::endl;
                std::string resp_string = response.extract_string().get();

                if(response.status_code() == status_codes::OK) {
                    nlohmann::json resp_json = nlohmann::json::parse(resp_string.c_str());

                    int total = resp_json["data"]["count"];
                    for (int i = 0; i < total; i++) {
                        std::string line = resp_json["data"]["lines"][i]["text"].get<std::string>();
                        std::cout << "line " << i << ": " << line.c_str() << std::endl;
                    }
                }
                else if (response.status_code() == status_codes::BadRequest)
                    std::cout << "Bad request!" << std::endl << resp_string.c_str() << std::endl;
                else
                    std::cout << "Whoops! Something wrong with request." << std::endl << resp_string.c_str() << std::endl;
            });
        }
        catch (const std::system_error& e)
        {
            std::cout << e.what() << std::endl;

            // Return an empty task.
            return pplx::task_from_result();
        }
    });
}

int main( int argc, char ** argv ) {
    const char *api = "http://127.0.0.1:6502/api/v1/";

    const char *filename = argc >= 2 ? argv[1] : "../../../sample_images/1.jpg";

    uploadFileToHttpServerAsync(filename , api).wait();

    std::cout << "Press enter to exit." << std::endl;
    getchar();
    return 0;
}

