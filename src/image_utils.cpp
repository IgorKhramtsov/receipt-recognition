#include "iostream"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "image_utils.h"
#include "../3dparty/nlohmann/json.hpp"
#include "boost/algorithm/string.hpp"


using namespace cv;

// Autocorrect gamma
void gamma_correction(Mat &in, Mat &out) {
    std::cout << "Gamma correction: " << std::endl;
    int brightness = 0;
    int brightness_row = 0;
    for(int y = 0; y < in.cols; y++) {
        for (int x = 0; x < in.rows; x++) {
            Vec3b color = in.at<Vec3b>(y, x);
            brightness_row += (color[0] + color[1] + color[2]) / 3;
        }
        brightness_row /= in.rows;
        brightness += brightness_row;
    }
    brightness /= in.cols;
    std::cout << "  Avg brightness: " << brightness << std::endl;

    //https://pdfs.semanticscholar.org/8150/961af293f8f54da684bdbda0b8d04bc7ec1b.pdf
    double gamma = -0.3 / log10(brightness / 255.);
    std::cout << "  Calculated gamma: " << gamma << std::endl;

    Mat lookupTable(1, 256, CV_8U);
    uchar *pointer = lookupTable.ptr();
    for(int i = 0; i < 256; i++)
        pointer[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);

    LUT(in, lookupTable, out);
}

// Grayscale image and adjust brightness and contrast
void brightness_and_contrast_correction(Mat &in, Mat &out) {
    std::cout << "Brightness and contrast correction: " << std::endl;
    double minGray = 0, maxGray = 0;

    Mat in_gray = Mat::zeros(in.size(), CV_8UC1);
    cvtColor(in, in_gray, CV_BGR2GRAY);
    std::cout << "  Image grayscaled." << std::endl;
    minMaxLoc(in_gray, &minGray, &maxGray);

    float inputRange = maxGray - minGray;
    double alpha = 255 / inputRange;
    int beta = -minGray * alpha;
    std::cout << "  Calculated alpha: " << alpha << std::endl;
    std::cout << "  Calculated beta: " << beta << std::endl;

    in_gray.convertTo(out, -1, alpha, beta);
}

void clustering(Mat &in, Mat &out) {
    out = Mat(in.size(), in.type());

    Mat samples(in.total(), 3, CV_32F);
    float *samples_ptr = samples.ptr<float>(0);

    auto cols_by_channels = in.cols * in.channels();
    for(int row = 0; row < in.rows; row++) {
        uchar *in_begin = in.ptr<uchar>(row);
        uchar *in_end = in_begin + cols_by_channels;
        while(in_begin < in_end) {
            samples_ptr[0] = in_begin[0];
            samples_ptr[1] = in_begin[1];
            samples_ptr[2] = in_begin[2];
            samples_ptr += 3;
            in_begin += 3;
        }
    }

    Mat bestLabels;
    Mat centers;
    kmeans(samples, 2, bestLabels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5, 1.0), 10,KMEANS_PP_CENTERS, centers);

    for(int row = 0; row < out.rows; row++) {
        uchar *out_begin = out.ptr<uchar>(row);
        uchar *out_end = out_begin + cols_by_channels;
        int *labels_ptr = bestLabels.ptr<int>(row * in.cols);

        while(out_begin < out_end) {
            int cluster_id = *labels_ptr;
            float *centers_ptr = centers.ptr<float>(cluster_id);
            out_begin[0] = centers_ptr[0];
            out_begin[1] = centers_ptr[1];
            out_begin[2] = centers_ptr[2];
            out_begin += 3;
            labels_ptr++;
        }
    }

    cvtColor(out, out, CV_BGR2GRAY);
    threshold(out, out, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);

}

void gaussianBlur(Mat &in, Mat &out) {
    GaussianBlur(in, out, Size(0, 0), 5, 5);
}

void grabCut(Mat &in, Mat &out) {
    out = Mat::zeros(in.size(), in.type());
    Mat mask = Mat::zeros(in.size(), CV_8UC1);
    Mat bgcModel;
    Mat fgModel;

    Rect rect(20, 20, in.rows - 140, in.cols - 140);

    grabCut(in, mask, rect, bgcModel, fgModel, 1, GC_INIT_WITH_RECT);
    in.copyTo(out, mask & 1);
    return;
}

void threshold(Mat &in, Mat &out) {
    cv::threshold(in, out, 16, 255, THRESH_BINARY);
}

void morphologic(Mat &in, Mat &out) {
    Mat in_resized;
    // Resize to speed up calculations
    resize(in, in_resized, Size( (float)in.cols / ((float)in.rows / 1600.) , 1600));
    if(in_resized.type() != CV_8UC1)
        cvtColor(in_resized, in_resized, COLOR_BGR2GRAY);

    // Blur to remove noises
    Mat image_corr_blur;
    blur(in_resized, image_corr_blur, Size(10, 10));

    Mat gradient;
    Mat morphStructure = getStructuringElement(MORPH_ELLIPSE, Size(8, 8));
    morphologyEx(image_corr_blur, gradient, MORPH_GRADIENT, morphStructure);
    threshold(gradient, gradient, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);

    Mat closed;
    morphStructure = getStructuringElement(MORPH_RECT, Size(15, 1));
    morphologyEx(gradient, out, MORPH_CLOSE, morphStructure);

    morphStructure = getStructuringElement(MORPH_RECT, Size(8, 8));
    morphologyEx(out, out, MORPH_ERODE, morphStructure);

    morphStructure = getStructuringElement(MORPH_RECT, Size(5, 10));
    morphologyEx(out, out, MORPH_OPEN, morphStructure);

    morphStructure = getStructuringElement(MORPH_RECT, Size(5, 70));
    morphologyEx(out, out, MORPH_DILATE, morphStructure);
    morphologyEx(out, out, MORPH_DILATE, morphStructure);

    resize(out, out, in.size());
}

void findTextContours(Mat &in, std::vector<std::vector<Point>> &contours) {

    Mat image;
    resize(in, image, Size( (float)in.cols / ((float)in.rows / 1600.) , 1600));
    Mat image_resized = image.clone();
    if(image.type() != CV_8UC1)
        cvtColor(image, image, COLOR_BGR2GRAY);
    blur(image, image, Size(10, 10));

    Mat morphStructure = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(image, image, MORPH_GRADIENT, morphStructure);
    threshold(image, image, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);


    morphStructure = getStructuringElement(MORPH_RECT, Size(15, 1));
    morphologyEx(image, image, MORPH_CLOSE, morphStructure);

    morphStructure = getStructuringElement(MORPH_RECT, Size(8, 8));
    morphologyEx(image, image, MORPH_ERODE, morphStructure);

//    resize(image, image, in.size());

    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    std::vector<std::vector<Point>> _contours;
    std::vector<Vec4i> hierarchy;
    findContours(image, _contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    for(int i = 0; i >= 0; i = hierarchy[i][0]) {
        Rect rect = boundingRect(_contours[i]);
        if(rect.height <= 10 || rect.width <= 10)
            continue;
        Mat maskROI(mask, rect);
        drawContours(mask, _contours, i, Scalar(255,255,255), CV_FILLED);
        float ratio = countNonZero(maskROI) / rect.width * rect.height;
        if(ratio > 0.45f)
            rectangle(image_resized, rect, Scalar(0, 255, 0), 2);
    }

    resize(image_resized, image_resized, Size( (float)in.cols / ((float)in.rows / 800.) , 800));
    //imshow("contt", image);
    imshow("Image contours", image_resized);


}

void getContours(Mat &in, Mat &contour_image, RotatedRect &rect) {
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    std::vector<Vec4i> hierarchy_hull;
    findContours( in, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    contour_image = Mat::zeros(in.size(), CV_8UC3);
    std::vector<std::vector<Point>> hull(contours.size());
    int hull_count = 0;
    for(int i = 0; i < contours.size(); i++) {
        Rect rect = boundingRect(contours[i]);
        // Skip small contours
        float image_scale = ((in.rows / 800.) + (in.cols / 600.)) / 2.;
        if(rect.height <= 10. * image_scale || rect.width <= 10. * image_scale) {
            continue;
        }

        drawContours(contour_image, contours, i, Scalar(250, 0, 0), 2, 8, hierarchy);
        convexHull( Mat(contours[i]), hull[hull_count++], false);
    }

    hull.resize(hull_count);
    int max_hull_index = 0;
    for(int i = 0; i < hull.size(); i++)
    {
        int hull_area = contourArea(hull[i]);
        if(hull_area >= contourArea(hull[max_hull_index]))
            max_hull_index = i;
    }
    drawContours(contour_image, hull, max_hull_index, Scalar(0, 255, 0), 2, 8, hierarchy_hull);

    rect = minAreaRect(hull[max_hull_index]);
}

void cutAndScale(Mat &in, Mat &out, RotatedRect &rect) {
    Mat in_clone = in.clone();

    Point2f box[4];
    rect.points(box);

//    if(rect.angle > 60)
//        rect.angle = rect.angle - 90;
//    else if(rect.angle < -60)
//        rect.angle = rect.angle + 90;
//
//    Mat rotationMatrix = getRotationMatrix2D(rect.center, rect.angle, 1);
//    // Rotate full image
//    warpAffine(in, in_clone, rotationMatrix, in.size());
//
//    // Rotate rectangle
//    std::vector<cv::Point2f> box_rotated(4);
//    transform(std::vector<Point2f>(std::begin(box), std::end(box)), box_rotated, rotationMatrix);
//    int x1 = max( min(box_rotated[0].x, min(box_rotated[1].x, min(box_rotated[2].x, box_rotated[3].x))), 0.f);
//    int x2 = min( max(box_rotated[0].x, max(box_rotated[1].x, max(box_rotated[2].x, box_rotated[3].x))), (float)in_clone.cols);
//    int y1 = max( min(box_rotated[0].y, min(box_rotated[1].y, min(box_rotated[2].y, box_rotated[3].y))), 0.f);
//    int y2 = min( max(box_rotated[0].y, max(box_rotated[1].y, max(box_rotated[2].y, box_rotated[3 ].y))), (float)in_clone.rows);

    int x1 = max( min(box[0].x, min(box[1].x, min(box[2].x, box[3].x))), 0.f);
    int x2 = min( max(box[0].x, max(box[1].x, max(box[2].x, box[3].x))), (float)in_clone.cols);
    int y1 = max( min(box[0].y, min(box[1].y, min(box[2].y, box[3].y))), 0.f);
    int y2 = min( max(box[0].y, max(box[1].y, max(box[2].y, box[3 ].y))), (float)in_clone.rows);
    Rect roi = Rect(Point(x1,y1), Point(x2,y2));

    out = in_clone(roi);

    if(out.rows < 1600 )
        resize(out, out, Size( (float)out.cols / ((float)out.rows / 1600.) , 1600));
    if(out.cols < 1200)
        resize(out, out, Size( 1200, (float)out.rows / ((float)out.cols / 1200.) ));
}

/*
    We do equalisation on BGR by converting to YCrCb where channel 0 is intensity
    and the other 2 channels carry the color info
*/
void equalizeIntensity(const Mat& in, Mat& out)
{
    if(in.channels() < 3)
        return;

    cv::Mat ycrcb;
    std::vector<Mat> channels;

    cvtColor(in, ycrcb, CV_BGR2YCrCb);

    split(ycrcb, channels);
    equalizeHist(channels[0], channels[0]);
    merge(channels, ycrcb);

    cvtColor(ycrcb, out, CV_YCrCb2BGR);
}


///----------------------------------------------------------------------------
//!
//! \brief This is the color balancing technique used in Adobe Photoshop's
//!        "auto levels" command. The idea is that in a well balanced photo,
//!        the brightest color should be white and the darkest black.
//!        Thus, we can remove the color cast from an image by scaling the
//!        histograms of each of the R, G, and B channels so that they span
//!        the complete 0-255 scale. In contrast to the other color balancing
//!        algorithms, this method does not separate the estimation and
//!        adaptation steps.
//!        In order to deal with outliers, Simplest Color Balance saturates a
//!        certain percentage of the image's bright pixels to white and dark
//!        pixels to black. The saturation level is an adjustable parameter
//!        that affects the quality of the output. Values around 0.01 are typical.
//!
//!        \param percent 0-100%
//!
//!        http://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html
//!
///----------------------------------------------------------------------------
void colorBalance(Mat& _inImg, Mat& _outImg, float percent)
{
    std::vector<Mat> split_colors;
    split(_inImg, split_colors);

    float half_percent = percent / 200.0f;

    for(int i = 0; i < 3; i++)
    {
        // find the low and high precentile values
        // (based on the input percentile)
        Mat imgFlat;
        split_colors[i].reshape(1,1).copyTo(imgFlat);

        cv::sort(imgFlat, imgFlat, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);

        int low_val  = imgFlat.at<uchar>(cvFloor(((float)imgFlat.cols) * half_percent));
        int high_val = imgFlat.at<uchar>(cvCeil(((float)imgFlat.cols) * (1.0 - half_percent)));

        // saturate below the low percentile and above the high percentile
        split_colors[i].setTo(low_val,  split_colors[i] < low_val);
        split_colors[i].setTo(high_val, split_colors[i] > high_val);

        // scale the channel
        normalize(split_colors[i], split_colors[i], 0, 255, NORM_MINMAX);
    }

    merge(split_colors, _outImg);
}


///----------------------------------------------------------------------------
//! \brief The following code implements the grey world algorithm.
//!        Grey world hypothesis assumes that the statistical mean of a scene
//!        is achromatic. Based on this assumption obtains an estimate for the
//!        scene illuminant to perform colour correction (white balancing)
//! \param _src - source image
//! \param _dst - sestination image
///----------------------------------------------------------------------------
void grayWorldFilter(const Mat& _src, Mat& _dst)
{
    cv::Scalar sumImg = sum(_src);

    // Normalise by the number of pixes in the image to obtain
    // an estimate for the illuminant
    cv::Scalar illum = sumImg / (_src.rows * _src.cols);

    // Split the image into different channels
    std::vector<cv::Mat> rgb_channels(3);

    cv::split(_src, rgb_channels);

    cv::Mat red   = rgb_channels[2];
    cv::Mat green = rgb_channels[1];
    cv::Mat blue  = rgb_channels[0];

    // Calculate scale factor for normalisation
    // (we can use 255 instead)
    double scale = (illum(0) + illum(1) + illum(2)) / 3;

    // Correct for illuminant (white balancing)
    red   = red * scale   / illum(2);
    green = green * scale / illum(1);
    blue  = blue * scale  / illum(0);

    // Assign the processed channels back into vector to use
    // in the cv::merge() function
    rgb_channels[0] = blue;
    rgb_channels[1] = green;
    rgb_channels[2] = red;

    /// Merge the processed colour channels
    merge(rgb_channels, _dst);
}


///----------------------------------------------------------------------------
//!
//! \brief 'Non uniform illumination' filter function
//! \param _src - source image
//! \param _dst - sestination image
//! \source http://answers.opencv.org/question/66125/object-detection-in-nonuniform-illumination/
//!
///----------------------------------------------------------------------------
void nonUniformIlluminationMorph(const cv::Mat &_src, cv::Mat &_dst)
{
    Mat brightness, src_hsv;
    std::vector<cv::Mat> hsv_planes;

    // GET THE BRIGHTNESS
    cv::cvtColor(_src, src_hsv, cv::COLOR_BGR2HSV);
    cv::split(src_hsv, hsv_planes);
    brightness = hsv_planes[2];
    bitwise_not(brightness, brightness);

    // REMOVE THE BACKGROUND
    Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1));
    Mat transformbuf = brightness.clone();
    dilate(transformbuf, transformbuf, element);
    erode(transformbuf, transformbuf, element);
    dilate(transformbuf, transformbuf, element);

//    adaptiveThreshold(transformbuf, transformbuf, 255., ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 7);
//    threshold(transformbuf, transformbuf, 0., 255., THRESH_BINARY | THRESH_OTSU);
//    imshow("asd", transformbuf);

    Scalar m = mean(transformbuf);
    brightness = brightness - transformbuf * 0.45;
    brightness = brightness + m(0) * 0.45;

    // BUILD THE DESTINATION
    bitwise_not(brightness, brightness);
    merge(hsv_planes, _dst);
    cvtColor(_dst, _dst, COLOR_HSV2BGR);

//    Mat cl = _dst.clone();
//    _dst = Mat::zeros(cl.size(), cl.type());
//    bitwise_not(_dst, _dst);
//    cl.copyTo(_dst, transformbuf & 1);
}

///----------------------------------------------------------------------------
//!
//! \brief 'Clahe' filter function
//! \param _src - source image
//! \param _dst - sestination image
//!
///----------------------------------------------------------------------------
void clahe(const cv::Mat &_src, cv::Mat &_dst)
{
    cv::Mat lab_image;
    cv::cvtColor(_src, lab_image, CV_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);

    clahe->apply(lab_planes[0], _dst);

    // Merge the the color planes back into an Lab image
    _dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

    // convert back to RGB
    cv::cvtColor(lab_image, _dst, CV_Lab2BGR);
}


///----------------------------------------------------------------------------
//!
//! \brief Filtering in chroma red and chroma blue components
//!
///----------------------------------------------------------------------------
void filteringCrCb(const cv::Mat &_src, cv::Mat &_dst)
{
    std::vector<cv::Mat> ycrcb_Channels(3);

    cvtColor(_src, _dst, CV_BGR2YCrCb);

    cv::split(_dst, ycrcb_Channels);

    GaussianBlur(ycrcb_Channels[1], ycrcb_Channels[1], cv::Size(3, 3), 3);
    GaussianBlur(ycrcb_Channels[2], ycrcb_Channels[2], cv::Size(3, 3), 3);

    merge(ycrcb_Channels, _dst);

    cvtColor(_dst, _dst, CV_YCrCb2BGR);
}

void sharpness(const Mat &src, Mat &dst, float sigma, float threshold, float amount) {
//    dst = Mat::zeros(src.size(), src.type());
    Mat blured;
    GaussianBlur(src, blured, Size(), sigma, sigma);
    Mat lowContrastMask = abs(src - blured) < threshold;
    dst = src * (1 + amount) + blured * (-amount);
    src.copyTo(dst, lowContrastMask);
}

void recognizeImage(const cv::Mat &img, tesseract::TessBaseAPI *ocr, nlohmann::json &rec) {
    Mat recognizeImage = img.clone();
    if(recognizeImage.type() == CV_8UC1)
        cvtColor(recognizeImage, recognizeImage, COLOR_GRAY2BGR);

    ocr->SetImage(recognizeImage.data, recognizeImage.cols, recognizeImage.rows, 3, recognizeImage.step);

    ocr->Recognize(0);

    tesseract::ResultIterator* ri = ocr->GetIterator();
    tesseract::PageIteratorLevel level = tesseract::RIL_TEXTLINE;
    int i = 0;
    if (ri != 0) {
        do {
            std::string word = ri->GetUTF8Text(level);
            float conf = ri->Confidence(level);
            int x1, y1, x2, y2;
            ri->BoundingBox(level, &x1, &y1, &x2, &y2);
            Rect rect(Point(x1, y1), Point(x2, y2));

            float image_scale = ((recognizeImage.rows / 800.) + (recognizeImage.cols / 600.)) / 2.;
            if(rect.height <= 10. * image_scale || rect.width <= 10. * image_scale)
                continue;
//            if(conf < 15)
//                continue;

            boost::erase_all(word, "\n");
            rec["data"]["lines"][i]["text"] = word;
            rec["data"]["lines"][i]["conf"] = conf;
            printf("ll: %s", rec["data"]["lines"][i]["text"].get<std::string>().c_str());


            rectangle(recognizeImage, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0), 2);
//            printf("line: '%s';  \tconf: %.2f; BoundingBox: %d,%d,%d,%d;\n",
//                   word.c_str(), conf, x1, y1, x2, y2);


            i++;
//            delete[] word;
        } while (ri->Next(level));
    }

    rec["data"]["count"] = i;

    resize(recognizeImage, recognizeImage, Size( (float)recognizeImage.cols / ((float)recognizeImage.rows / 900.) , 900));
    imshow("Recognized", recognizeImage);
}

void deSkew(const cv::Mat &in, cv::Mat &out) {
    Mat image_to_morph = in.clone();

    int left_right = in.size().width * 0.2;
    int top_bottom = in.size().height * 0.1;
    Rect Roi(left_right, top_bottom, in.size().width - left_right * 2, in.size().height - top_bottom * 2);

    image_to_morph = in(Roi);
    //cvtColor(image_sharpened, image_sharpened, COLOR_GRAY2BGR);

    Mat gradient;
    Mat morphStructure = getStructuringElement(MORPH_ELLIPSE, Size(1, 3));
    morphologyEx(image_to_morph, gradient, MORPH_GRADIENT, morphStructure);
    threshold(gradient, gradient, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);

    morphStructure = getStructuringElement(MORPH_RECT, Size(105, 1));
    morphologyEx(gradient, gradient, MORPH_CLOSE, morphStructure);

    morphStructure = getStructuringElement(MORPH_RECT, Size(11, 10));
    morphologyEx(gradient, gradient, MORPH_ERODE, morphStructure);

    std::vector<Vec4i> lines;
    HoughLinesP(gradient, lines, 1, CV_PI / 180, 100, gradient.cols / 2., 8);

    cv::Mat disp_lines(gradient.size(), CV_8UC1, cv::Scalar(0, 0, 0));
    double angle = 0.;
    unsigned nb_lines = lines.size();
    for (unsigned i = 0; i < nb_lines; ++i)
    {
        cv::line(disp_lines, cv::Point(lines[i][0], lines[i][1]),
                 cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0 ,0));
        angle += atan2((double)lines[i][3] - lines[i][1],
                       (double)lines[i][2] - lines[i][0]);
    }
    angle /= nb_lines; // mean angle, in radians.

    std::cout << "Skew: " << angle * 180 / CV_PI << std::endl;

    Mat rotationMatrix = getRotationMatrix2D(Point(in.size().width / 2, in.size().height / 2), angle * 180 / CV_PI, 1);
    // Rotate full image
//    Mat new_corr = Mat::zeros(in.size(), in.type());

    Size new_size(in.size().width * 1.2, in.size().height * 1.2);
    rotationMatrix.at<double>(0,2) += in.size().width * 0.1;
    rotationMatrix.at<double>(1,2) += in.size().height * 0.1;
    out = Mat::zeros(new_size, in.type());

    warpAffine(in, out, rotationMatrix,
               out.size(), INTER_LINEAR, BORDER_CONSTANT,
               Scalar(255, 255, 255));

//    cv::imshow("asd", disp_lines);
//    cv::imshow("Corrected", image_corrected);
}
