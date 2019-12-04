//  circle_detection
//
//  Created by YunSung-jae on 11/21/19.
//  Copyright © 2019 YunSung-jae. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void gaussian(const Mat& input_gray_img, Mat& gauss_result);
void sobel(const Mat& image, Mat& sob_result, Mat& nmr_result, uchar thresh);

int main()
{
   //load img
    Mat input_gray_img = imread("lena.png",IMREAD_GRAYSCALE);
    //Mat input_gray_img = imread("drug.jpg",IMREAD_GRAYSCALE);

    Mat gauss_result, sob_result, nmr_result;

    gaussian(input_gray_img,gauss_result);
    sobel(gauss_result, sob_result, nmr_result, 24);

    imshow("gray", input_gray_img);
    imshow("gauss", gauss_result);
    imshow("sobel", sob_result);
    imshow("nmr", nmr_result);

    waitKey(0);
    return 0;
}

void gaussian(const Mat& input_gray_img, Mat& g_result){
    int height = input_gray_img.rows;
    int width = input_gray_img.cols;
    g_result.create(input_gray_img.size(), input_gray_img.type());

    Mat kernel_mat = (Mat_<float>(5,5) << 0.003765,	0.015019,	0.023792,	0.015019,	0.003765,
                        0.015019,	0.059912,	0.094907,	0.059912,	0.015019,
                        0.023792,	0.094907,	0.150342,	0.094907,	0.023792,
                        0.015019,	0.059912,	0.094907,	0.059912,	0.015019,
                        0.003765,	0.015019,	0.023792,	0.015019,	0.003765);
    // kernel size
    int kernel_size= kernel_mat.rows ;
    
    //apply filter
    for(int i = kernel_size/2 ; i < height; i++){
        //make img pointer
        const  uchar* imgptr[kernel_size];
        for(int j =0 ; j< kernel_size; j++){
            imgptr[j]=input_gray_img.ptr<uchar>(i + j-kernel_size/2);
        }

        uchar* output = g_result.ptr<uchar>(i);
        for(int j=0; j<width;j++){
            int actv=0;

            //by row
            for(int k=0; k<kernel_size;k++){
                //by col
                for(int l=0; l<kernel_size;l++){
                    //cout<< imgptr[k][j-kernel_size/2] <<endl;
                    actv+=imgptr[k][j-kernel_size/2]*kernel_mat.at<float>(k,l);
                }   
            }
            *output++= saturate_cast<uchar>(actv);
        }
    } 
    g_result.row(0).setTo(Scalar(0));
    g_result.row(g_result.rows-1).setTo(Scalar(0));
    g_result.col(0).setTo(Scalar(0));
    g_result.col(g_result.cols-1).setTo(Scalar(0));
}

void sobel(const Mat& image, Mat&sob_result, Mat& nmr_result, uchar thresh){ 
    //mask for each direction
    Mat mask_x = (Mat_<double>(3, 3) << 1, 0, -1, 
                                        2, 0, -2, 
                                        1, 0, -1); 
    Mat mask_y = (Mat_<double>(3, 3) << 1, 2, 1, 
                                        0, 0, 0, 
                                        -1, -2, -1); 
    int filterOffset = 3 / 2; 

    sob_result = Mat::zeros(image.rows - filterOffset * 2, image.cols - filterOffset * 2, image.type());
    nmr_result = Mat::zeros(image.rows - filterOffset * 2, image.cols - filterOffset * 2, image.type()); 
 
    double dx; 
    double dy; 
    double magnitude; 
    const double deg_factor=57.2957; // 180/pi
    
    for (int yimage = filterOffset; yimage < image.rows - filterOffset; yimage++){ 
        for (int ximage = filterOffset; ximage < image.cols - filterOffset; ximage++){ 
            dx = 0; 
            dy = 0; 
            for (int ymask = -filterOffset; ymask <= filterOffset; ymask++){ 
                for (int xmask = -filterOffset; xmask <= filterOffset; xmask++){ 
                    //x
                    dx += image.at<uchar>(yimage + ymask, ximage + xmask) * mask_x.at<double>(filterOffset + ymask, filterOffset + xmask); 
                    //y
                    dy += image.at<uchar>(yimage + ymask, ximage + xmask) * mask_y.at<double>(filterOffset + ymask, filterOffset + xmask); 
                } 
            } 

            magnitude = sqrt(pow(dy, 2) + pow(dx, 2)); 
            //threshold
            sob_result.at<uchar>(yimage - filterOffset, ximage - filterOffset) = ((magnitude > thresh) ? 255 : 0); 

            //non maxima suppression
            //if ((magnitude > thresh) && (yimage < (image.rows-filterOffset*2) ) && (ximage < (image.cols-filterOffset*2)))
            if (magnitude > thresh) 
            {

                int angle=atan2(dy,dx)*deg_factor;
                angle= ((angle <0) ? (angle+180):angle);
                nmr_result.at<uchar>(yimage, ximage)=255;
                //case0 angle0
                if (((0 <= angle) && (angle < 22.5)) || ((157.5 <= angle) && (angle <= 180))){
                    nmr_result.at<uchar>(yimage,ximage+1)=255;
                    nmr_result.at<uchar>(yimage,ximage-1)=255;
                }
                //case1 angle45
                else if (22.5 <= angle < 67.5){
                    nmr_result.at<uchar>(yimage+1,ximage-1)=255;
                    nmr_result.at<uchar>(yimage-1,ximage+1)=255;

                }
                //case2 angle90
                else if (67.5 <= angle < 112.5){
                    nmr_result.at<uchar>(yimage+1,ximage)=255;
                    nmr_result.at<uchar>(yimage-1,ximage)=255;

                }
                //case3 angle 135
                else if (112.5 <= angle < 157.5){
                    nmr_result.at<uchar>(yimage-1,ximage-1)=255;
                    nmr_result.at<uchar>(yimage+1,ximage+1)=255;

                }
             
            }

            

            
        } 
    } 

}
