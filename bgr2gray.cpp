//
//  main.cpp
//  circle_detection
//
//  Created by YunSung-jae on 11/21/19.
//  Copyright © 2019 YunSung-jae. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int main()
{
    Mat img_color = imread("drug.jpg", IMREAD_COLOR);

    int height = img_color.rows;
    int width = img_color.cols;

    Mat img_grayscale(height, width, CV_8UC1);
    
    double t2 = (double)getTickCount();

    for (int y = 0; y < height; y++) {
        uchar* pointer_input = img_color.ptr<uchar>(y);
        uchar* pointer_ouput = img_grayscale.ptr<uchar>(y);

        for (int x = 0; x < width; x++) {

            uchar b = pointer_input[x * 3 + 0];
            uchar g = pointer_input[x * 3 + 1];
            uchar r = pointer_input[x * 3 + 2];

            pointer_ouput[x] = (r + g + b) / 3.0;
//            cout << "color :  " << dec<<int(pointer_ouput[x]) <<endl;

        }
    }

    t2 = ((double)getTickCount() - t2) / getTickFrequency();
    cout << "time ptr =  " << t2 << " sec" << endl;

    imshow("color", img_color);
    imshow("grayscale", img_grayscale);

    waitKey(0);
    return 0;
    
}


