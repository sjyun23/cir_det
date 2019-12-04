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

void gaussian(const Mat& input_gray_img, Mat& gauss_result);
void sobel(const Mat& image, Mat& sob_result, Mat& nmr_result, uchar thresh);
int find_edge_component(Mat& canvas, uchar ypnt, uchar xpnt, uchar edge_no);

int main()
{
   //load img
    Mat input_gray_img = imread("lena.jpg",IMREAD_GRAYSCALE);
    //Mat input_gray_img = imread("drug.jpg",IMREAD_GRAYSCALE);

    Mat gauss_result, sob_result, nmr_result;

    gaussian(input_gray_img,gauss_result);
    sobel(gauss_result, sob_result, nmr_result, 47);

    int b_sig=0;
    int edge_length=0;
    for(int i=0 ; i <input_gray_img.rows; i++ ){
        for(int j=0 ; j <input_gray_img.cols; j++ ){
            if (nmr_result.at<uchar>(i,j)==255){
                edge_length=find_edge_component(nmr_result, i, j, 1);
                b_sig=1;
                break;
                }
        }
        if (b_sig==1){
            break;}
    }

    cout<< edge_length<< endl;



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

    Mat kernel_mat = (Mat_<float>(5,5) << 0.003765,    0.015019,    0.023792,    0.015019,    0.003765,
                        0.015019,    0.059912,    0.094907,    0.059912,    0.015019,
                        0.023792,    0.094907,    0.150342,    0.094907,    0.023792,
                        0.015019,    0.059912,    0.094907,    0.059912,    0.015019,
                        0.003765,    0.015019,    0.023792,    0.015019,    0.003765);
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


void sobel(const Mat& image, Mat& sob_result, Mat& nmr_result, uchar thresh){
    //mask for each direction
    Mat mask_x = (Mat_<double>(3, 3) << -1, 0, 1,
                                        -1, 0, 1,
                                        -1, 0, 1);
    Mat mask_y = (Mat_<double>(3, 3) << 1, 1, 1,
                                        0, 0, 0,
                                        -1, -1, -1);
    int filterOffset = 3 / 2;

    sob_result = Mat::zeros(image.rows - filterOffset * 2, image.cols - filterOffset * 2, image.type());
    nmr_result = Mat::zeros(image.rows - filterOffset * 2, image.cols - filterOffset * 2, image.type());
    cout<<image.type()<<endl;
 
    double dx;
    double dy;
    double gradient;
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

            gradient = sqrt(pow(dy, 2) + pow(dx, 2));

            //thresholding & non maxima suppression
            if (gradient > thresh)
            {
                sob_result.at<uchar>(yimage - filterOffset, ximage - filterOffset) = 255;
                nmr_result.at<uchar>(yimage - filterOffset, ximage - filterOffset) = 255;

                int angle=atan2(dy,dx)*deg_factor;
                angle= ((angle <0) ? (angle+180):angle);
                //nmr_result.at<uchar>(yimage, ximage)=255;
                //case0 angle0
                if (((0 <= angle) && (angle < 22.5)) || ((157.5 <= angle) && (angle <= 180))){
                    nmr_result.at<uchar>(yimage - filterOffset,ximage - filterOffset+1)=0;
                    nmr_result.at<uchar>(yimage - filterOffset,ximage - filterOffset-1)=0;
                }
                //case1 angle45
                else if (22.5 <= angle < 67.5){
                    nmr_result.at<uchar>(yimage - filterOffset+1,ximage - filterOffset-1)=0;
                    nmr_result.at<uchar>(yimage - filterOffset-1,ximage - filterOffset+1)=0;

                }
                //case2 angle90
                else if (67.5 <= angle < 112.5){
                    nmr_result.at<uchar>(yimage - filterOffset+1,ximage - filterOffset)=0;
                    nmr_result.at<uchar>(yimage - filterOffset-1,ximage - filterOffset)=0;

                }
                //case3 angle 135
                else if (112.5 <= angle < 157.5){
                    nmr_result.at<uchar>(yimage - filterOffset-1,ximage - filterOffset-1)=0;
                    nmr_result.at<uchar>(yimage - filterOffset+1,ximage - filterOffset+1)=0;

                }
            }
        }
    }
}


int find_edge_component(Mat& canvas, uchar ypnt, uchar xpnt, uchar edge_no){
    //find edge
    int edge_length=1;
    bool end_of_edge=false;
    canvas.at<uchar>(ypnt, xpnt)=edge_no;

    while (end_of_edge==false)
    {
        int i=1, j=1;
        bool find_flag=false;
        for(i=1; i >-2; i--){
            for(j=1; j>-2;j--){
                if (canvas.at<uchar>(ypnt+i, xpnt+j)==255){
                    edge_length++;
                    canvas.at<uchar>(ypnt+i, xpnt+j)=edge_no;
                    find_flag=true;
                    ypnt=ypnt+i;
                    xpnt=xpnt+j;
                    break;
                }
            }
            if (find_flag==true){
                break;
            }
        }
        if (i==-2 && j==-2 && find_flag==false){
            end_of_edge = true;
            break;
        }
    }

    return edge_length;
}




