#include "BorderDetectKmeans.h"
#include "CLogger.h"
#include "CStatistics.h"
#include "CTimeMeasure.h"
#include "Constants.h"
#include "Drawing.h"
#include "Geometry.h"
#include "HairRemoval.h"
#include "ImageUtils.h"
#include "RingArtifact.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "typedefs.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

using namespace cv;
using namespace std;

cv::Mat lastProcessedImg;

///----------------------------------------------------------------------------
//!
//! \brief This code will be called with a _lesion.config.inputImgPath
//!
//! You can do anything you like and have all the command line argument
//! controls of the skinview app
//!
//! \param LesionStructT& _lesion - the _lesion strut object
//! \return int
//!
///----------------------------------------------------------------------------
int processFile(LesionStructT& _lesion)
{
    cout << "Lesion: " << _lesion.config.inputImgPath << endl;

    _lesion.inputImg = cv::imread(_lesion.config.inputImgPath, CV_LOAD_IMAGE_COLOR); // Load the input image

    /// If image not found return ER_IO
    if (!_lesion.inputImg.data)
    {
        cerr << "Could not open or find the image" << endl;
        return ER_IO;
    }

    // control image size for display
    int WIDTH_MAX = 640;

    if (_lesion.inputImg.cols > WIDTH_MAX)
    {
        int rows = _lesion.inputImg.rows;
        int cols = _lesion.inputImg.cols;
        resize(_lesion.inputImg, _lesion.inputImg, Size(WIDTH_MAX, WIDTH_MAX * rows / cols));
    }

    /// Show what we started with.
    /// We do it like this to keep processed image displaying if --no-wait
    if (_lesion.config.imgShow)
    {
        if (_lesion.config.showPWindow)
        {
            cv::imshow("Processing Now", _lesion.inputImg);
        }
        if (lastProcessedImg.empty())
        {
            cv::imshow("Image", _lesion.inputImg);
            cerr << "Input image" << endl;
        }
        else
        {
            cv::imshow("Image", lastProcessedImg);
            /// this 100msec delay allows image to reliably render before thread lock from next step
            waitKey(100);
        }
    }

    ///----------------------------------------------------------------------------
    //!
    //! DO NOT CHANGE ABOVE THIS LINE (without a good reason)
    //!
    ///----------------------------------------------------------------------------

    /* 

   YOUR CODE GOES HERE :)
 
   Do set lastProcessedImg at the end of processing like:

   lastProcessedImg = your_output_image.clone()

   Particularly if you have a long running process, and are running auto and want to
   have a chance to look at the output while the next iteration runs....

*/

    LOG_INFO("Processed: " + _lesion.config.inputImgPath);

    ///----------------------------------------------------------------------------
    //!
    //! DO NOT CHANGE BELOW THIS LINE (without a good reason)
    //!
    ///----------------------------------------------------------------------------

    if (_lesion.config.wait && _lesion.config.imgShow)
    {
        /// avoid potential redirection of STDOUT and print to console
        cerr << "\nClick window and press a key to continue..." << endl;
        cv::waitKey(_lesion.config.waitKeyDelay);
    }
    return 0;
}
