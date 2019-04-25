#include <fstream>
#include <iterator>
#include <algorithm>
#include <string>
#include <getopt.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <glob.h>

#include "CLogger.h"
#include "typedefs.h"
#include "Constants.h"
#include "Process.h"

using namespace std;

///----------------------------------------------------------------------------
//!
//! \brief Functions prints help information about the usage of the application
//!
//! \return void
//!
///----------------------------------------------------------------------------
void printHelp()
{
    puts(
            "  Receipt analyzer program help:\n\n"
            "  --help, -h             Print this help page\n"
            "  --input, -i            Input Image (Default: 'J_22.jpg')\n"
            "  --recurse, -r          If --input is a directory recurse all subdirs\n"
            "  --output, -o           Output image name for saving results\n"
            "  --data-file, -f        Data file name for saving results\n"
            "  --cluster-count, -c    kMeans cluster count\n"
            "  --no-images, -n        Do not show output images\n"
            "  --no-wait, -w          Do not show output images\n"
            "  --wait-key, -k         waitKey for this many mSec before continuing\n"
            "  --silent, -s           Print data to STDOUT.txt not console\n"
            "  --show-processing, -p  Show a second window with receipt being processed\n"
            "  --auto, -a             Auto - same as -r -n -w -s\n"
            "  --debug-level, -d      Debug level 1..9\n"
            "  --verbose, -v          Verbose output, sets --debug-level 9\n");
}


///----------------------------------------------------------------------------
//!
//! \brief Function parses command line
//!
//! \param argc int
//! \param argv char**
//! \return void
//!
///----------------------------------------------------------------------------
int parseCmdLine(int argc, char** argv, AppConfigT& config)
{
    int result;
    int opt_index;
    const char* short_opt = "hrnwspavi:o:f:c:k:d:"; // note that short options with arguments need :

    const struct option long_opt[] =
    {
        {"help",            no_argument,        0, 'h'},
        {"input",           required_argument,  0, 'i'},
        {"recurse",         no_argument,        0, 'r'},
        {"output",          required_argument,  0, 'o'},
        {"data-file",       required_argument,  0, 'f'},
        {"cluster-count",   required_argument,  0, 'c'},
        {"no-images",       no_argument,        0, 'n'},
        {"no-wait",         no_argument,        0, 'w'},
        {"wait-key",        required_argument,  0, 'k'},
        {"silent",          no_argument,        0, 's'},
        {"show-processing", no_argument,        0, 'p'},
        {"auto",            no_argument,        0, 'a'},
        {"debug",           required_argument,  0, 'd'},
        {"verbose",         no_argument,        0, 'v'},
        {0, 0, 0, 0}
    };

    while((result = getopt_long(argc, argv, short_opt, long_opt, &opt_index)) != -1)
    {
        switch(result)
        {
            case 'i':                                       // '-i' - input file/dir
                config.inputImgPath = std::string(optarg);
                break;

            case 'r':                                       // '-r' - recursion control
                config.recurse = true;
                break;

            case 'o':                                       // '-o' - output file
                config.outputImgPath = std::string(optarg);
                break;

            case 'f':                                       // '-f' - data file
                config.dataFilePath = std::string(optarg);
                break;

            case 'c':                                       // '-c' - kMeans cluster count
                config.clusterCount = atoi(optarg);
                break;

            case 'n':                                       // '-n' - flag don't show image
                config.imgShow = false;
                break;

            case 'w':                                       // '-w' - don't waitKey(0)
                config.wait = false;
                break;

            case 'k':                                       // '-k' - waitKey(arg mSec)
                config.waitKeyDelay = atoi(optarg);
                break;

            case 's':                                       // '-s' - flag to be silent
                config.silent = true;
                freopen("STDOUT.txt","w",stdout);           // redirect stdout to file
                break;

            case 'p':                                       // '-p' - flag to show processing window
                config.showPWindow = true;
                break;

            case 'a':                                       // '-a' - automatic processing
                config.recurse = true;               // '-r' - recurse
                config.imgShow = false;              // '-n' - don't show image
                config.wait    = false;              // '-w' - don't waitKey(0)
                config.silent  = true;               // '-s' - don't spew info
                freopen("STDOUT.txt","w",stdout);           // redirect stdout to file
                break;

            case 'd':                                       // '-d' - set debug level
                config.debugLevel = atoi(optarg);
                break;

            case 'v':                                       // '-v' - set debug level 9
                config.debugLevel = 9;
                break;

            case 'h':
            default:
                printHelp();
                exit(0);
        }
    }

    if ( config.debugLevel > 0 ) 
    {
        cout << "Application Config" << endl;
        cout << "clusterCount: " << config.clusterCount << endl;
        cout << "debugLevel:   " << config.debugLevel << endl;
        cout << "waitKeyDelay: " << config.waitKeyDelay << endl;
        cout << "imgShow:      " << BOOLSTR(config.imgShow) << endl;
        cout << "wait:         " << BOOLSTR(config.wait) << endl;
        cout << "silent:       " << BOOLSTR(config.silent) << endl;
        cout << "recurse:      " << BOOLSTR(config.recurse) << endl;
        cout << "showPWindow:  " << BOOLSTR(config.showPWindow) << endl;
        cout << "inputImgPath: " << config.inputImgPath << endl;
        cout << "outputImgPath:" << config.outputImgPath << endl;
        cout << "dataFilePath: " << config.dataFilePath << endl;
    } 
    return 0;
}

/*
    Abandon hope, all ye who enter here. Recursive code lurks below.

    If you don't understand why GNU is an infinite loop please leave now.

    This code is protected by dragon, this dragon...

            <>=======()
           (/\___   /|\\          ()==========<>_
                 \_/ | \\        //|\   ______/ \)
                   \_|  \\      // | \_/
                     \|\/|\_   //  /\/
                      (oo)\ \_//  /
                     //_/\_\/ /  |
                    @@/  |=\  \  |
                         \_=\_ \ |
                           \==\ \|\_
                        __(\===\(  )\
                       (((~) __(_/   |
                            (((~) \  /
                            ______/ /
                            '------'

    Don't let his cuteness fool you, he bites. I once saw him consume the CPU, 
    every byte of swap, and an entire SAN. Didn't even flinch.

    Proceed at your own risk.
*/

///----------------------------------------------------------------------------
//!
//! \brief Process a directory
//!
//! \param AppConfigT& config
//! 
//! Note the config.inputImgPath contains the directory name
//! Also note that this function is recursive if config.recurse is set true
//!
//! \return int
//!
///----------------------------------------------------------------------------
int processDirectory(AppConfigT& config)
{
    std::string directory = config.inputImgPath + '/' + '*';
    struct stat s;
    glob_t glob_result;

    cout << "Processing directory: " << directory << endl;

    glob(directory.c_str(),GLOB_TILDE,NULL,&glob_result);

    for(unsigned int i=0; i<glob_result.gl_pathc; ++i){

        config.inputImgPath = std::string(glob_result.gl_pathv[i]);

        if ( stat(glob_result.gl_pathv[i],&s) == 0 ) 
        {
            if ( s.st_mode & S_IFDIR )
            {
                if ( config.recurse)
                {
                    processDirectory(config);
                }
            }
            else if ( s.st_mode & S_IFREG )
            {
                ReceiptStructT receipt;
                receipt.config = config;
                processFile(receipt);
            }
            else
            {
                std::stringstream error;
                error << "Unknown file/directory type, mode: " << s.st_mode << " " << config.inputImgPath;
                cerr << error.str() << endl;
                LOG_ERROR(error.str());
            }
        }
        else 
        {
            std::stringstream error;
            error << "Can't stat " << config.inputImgPath;
            cerr << error.str() << endl;
            LOG_ERROR(error.str());
        }
    }
    return 0;
}


///----------------------------------------------------------------------------
//!
//! \brief The entry point of the application
//!
//! \param argc int - number of input parameters
//! \param argv char** - symbol array of the input parameters
//! \return int - application return value
//!
///----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    AppConfigT config;

    config.inputImgPath = "../../SampleImages/Dermatoscopic/J_22.jpg";
    config.dataFilePath = "stat_info_data.csv";

    parseCmdLine(argc, argv, config);

    /// Create a display window if we need it
    if(config.imgShow)
    {
        if (config.showPWindow) {
            cv::namedWindow("Processing Now", cv::WINDOW_AUTOSIZE);
            cv::moveWindow("Processing Now", 640, 0);
        }
        cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
        cv::moveWindow("Image", 0, 0);
    }

    struct stat s;
    if ( stat(config.inputImgPath.c_str(),&s) == 0 ) 
    {
        if ( s.st_mode & S_IFDIR )
        {
            return processDirectory(config);
        }
        else if ( s.st_mode & S_IFREG )
        {
            ReceiptStructT receipt;
            receipt.config = config;
            return processFile(receipt);
        }
        else
        {
            std::stringstream error;
            error << "Unknown file/directory type, mode: " << s.st_mode << " " << config.inputImgPath;
            cerr << error.str() << endl;
            LOG_ERROR(error.str());
        }
    }
    else 
    {
        std::stringstream error;
        error << "Can't stat " << config.inputImgPath;
        cerr << error.str() << endl;
        LOG_ERROR(error.str());
        return ER_IO;
    }

    return 0;
}
