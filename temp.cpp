//
//  cd.cpp
//  circle_detection
//
//  Created by YunSung-jae on 11/21/19.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stack>


using namespace cv;
using namespace std;

void gaussian(const Mat& input_gray_img, Mat& gauss_result);
void sobel(const Mat& image, Mat& sob_result, Mat& nmr_result, Mat& grd_map, uchar thresh);
int find_edge_component(Mat& canvas, int ypnt, int xpnt, int edge_no);
Mat ed_anchor_nfa(const Mat& nmr_result,const Mat& grad_map, int anchor_thr, int anch_detail_ratio );


int main(){
    double t2 = (double)getTickCount();

   //load img
    Mat input_gray_img = imread("lena.jpg",IMREAD_GRAYSCALE);
    //Mat input_gray_img = imread("drug.jpg",IMREAD_GRAYSCALE);

    Mat gauss_result, sob_result, nmr_result, grad_map;

    gaussian(input_gray_img,gauss_result);
    
    int sobel_thr=55; //prewitt actually..
    sobel(gauss_result, sob_result, nmr_result, grad_map, sobel_thr);
    
    int anchor_thr=1, anch_detail_ratio=5;
    Mat anch_canvas=ed_anchor_nfa(nmr_result, grad_map, anchor_thr, anch_detail_ratio );

    t2 = ((double)getTickCount() - t2) / getTickFrequency();
    cout << "time ptr =  " << t2 << " sec" << endl;
    
    imshow("gray", input_gray_img);
    imshow("gauss", gauss_result);
    imshow("gradient map", grad_map);
    imshow("sobel/prewitt", sob_result);
    imshow("nonmaxima suppress", nmr_result);
    imshow("anchor map", anch_canvas);

    waitKey(0);
    return 0;
}

void gaussian(const Mat& input_gray_img, Mat& g_result){
    int height = input_gray_img.rows;
    int width = input_gray_img.cols;
    g_result.create(input_gray_img.size(), input_gray_img.type());

    //gaussian kernel 5*5 sigma 1
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

void sobel(const Mat& image, Mat& sob_result, Mat& nmr_result, Mat& grd_map, uchar thresh){
    //mask for each direction --prewitt
    Mat mask_x = (Mat_<double>(3, 3) << -1, 0, 1,
                                        -1, 0, 1,
                                        -1, 0, 1);
    Mat mask_y = (Mat_<double>(3, 3) << 1, 1, 1,
                                        0, 0, 0,
                                        -1, -1, -1);

    //mask for each direction  --sobel
//    Mat mask_x = (Mat_<double>(3, 3) << -1, 0, 1,
//                                        -2, 0, 2,
//                                        -1, 0, 1);
//    Mat mask_y = (Mat_<double>(3, 3) << 1, 2, 1,
//                                        0, 0, 0,
//                                        -1, -2, -1);
    
    int filterOffset = 3 / 2;

    sob_result = Mat::zeros(image.rows - filterOffset * 2, image.cols - filterOffset * 2, image.type());
    nmr_result = Mat::zeros(image.rows - filterOffset * 2, image.cols - filterOffset * 2, image.type());
    grd_map = Mat::zeros(image.rows - filterOffset * 2, image.cols - filterOffset * 2, image.type());

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
            if (gradient > thresh){
                grd_map.at<uchar>(yimage - filterOffset, ximage - filterOffset) = gradient;
                sob_result.at<uchar>(yimage - filterOffset, ximage - filterOffset) = 255;
                nmr_result.at<uchar>(yimage - filterOffset, ximage - filterOffset) = 255;

                if (yimage==0 || yimage==nmr_result.rows || ximage==0 || ximage==nmr_result.cols)
                {break;}
                int angle=atan2(dy,dx)*deg_factor;
                angle= ((angle <0) ? (angle+180):angle);
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

Mat ed_anchor_nfa(const Mat& nmr_result,const Mat& grad_map, int anchor_thr, int anch_detail_ratio ){
    int edge_length=0,edge_no=1;
    int total_edge_length = cv::countNonZero(nmr_result == 255);
    stack<int> y_st, x_st, length_st;
    stack<int> y_anc, x_anc;
    
    //counting edges
    Mat canvas;
    nmr_result.copyTo(canvas);
    for(int i=0 ; i <nmr_result.rows; i++ ){
        for(int j=0 ; j <nmr_result.cols; j++ ){
            if (nmr_result.at<uchar>(i,j)==255){
                edge_length=find_edge_component(canvas, i, j, 250);
                y_st.push(i);
                x_st.push(j);
                length_st.push(edge_length);
                edge_no++;
            }
        }
    }
    
    //cout<<"edge length total: "<<total_edge_length<<endl;
    //cout<<"no of edges: "<< edge_no<<endl;
    
    Mat anch_canvas;
    nmr_result.copyTo(anch_canvas);
    float edge_no_2nd=1;
    //check NFA until edge stack empty
    while (!y_st.empty()) {
        int length_single_edge= length_st.top();
        
            //check nfa condition
        if (((length_single_edge* edge_no)/ total_edge_length ) > anchor_thr){
            edge_length=find_edge_component(anch_canvas, y_st.top(), x_st.top(), edge_no_2nd);
            edge_no_2nd++;
            if(y_st.top()%anch_detail_ratio==0){
                y_anc.push(y_st.top());
                x_anc.push(x_st.top());
            }
        }
        y_st.pop(); x_st.pop(); length_st.pop();
    }
    
    //reducing anchors by detail ratio
    for(int i=0; i<anch_canvas.rows; i++){
        if(i%anch_detail_ratio!=0){
            anch_canvas.row(i)=0;
        }
    }
    
    anch_canvas.setTo(0, anch_canvas == 255);
    anch_canvas.setTo(255, anch_canvas > 0);
    
    //edge drawing
    Mat ed_canvas;
    anch_canvas.copyTo(ed_canvas);

    while (!y_anc.empty()) {
        bool finding_next_anchor=true;
        
        int ypnt= y_anc.top(),xpnt= x_anc.top();
        y_anc.pop();x_anc.pop();
        
        while (finding_next_anchor) {
            
            int i=1, j=1, temp=0, grad_max, grad;
            int direction_y=1,direction_x=1;
            
            grad_max=0;//grad_map.at<uchar>(ypnt+1,xpnt+1);
            grad= ed_canvas.at<uchar>(ypnt,xpnt);

            for(i=0; i >-2; i--){
                 for(j=0; j>-2;j--){
                     if(!(i==0 && j==0)){
                         temp= grad_map.at<uchar>(ypnt+i,xpnt+j);
                         if ((grad_max < temp) && (ed_canvas.at<uchar>(ypnt+i,xpnt+j) != 250 ) ){
                             grad_max=temp;
                             direction_y=i;
                             direction_x=j;
                         }
                     }
                 }
            }
                ypnt=ypnt+direction_y;
                xpnt=xpnt+direction_x;
            
                if (((ypnt) < 0) || ((ypnt) > ed_canvas.rows) || ((xpnt) < 0) || (xpnt > (ed_canvas.cols)) ){
                    finding_next_anchor = false;
                    break;
                }
                
                if (ed_canvas.at<uchar>(ypnt,xpnt)>=250 && grad_max==0){
                    finding_next_anchor=false;
                }else{
                    ed_canvas.at<uchar>(ypnt,xpnt)=250;
                }
        }
    }
    return ed_canvas;
}

int find_edge_component(Mat& canvas, int ypnt, int xpnt, int edge_no){
    //find edge
    int edge_length=1;
    bool end_of_edge=false;
    canvas.at<uchar>(ypnt, xpnt)=edge_no;

    if(edge_no >= 255){
        edge_no =edge_no%254 ;
    }
    
    while (end_of_edge==false){
        int i=1, j=1;
        bool find_flag=false;
        for(i=1; i >-2; i--){
            for(j=1; j>-2;j--){
                if (canvas.at<uchar>(ypnt+i, xpnt+j)==255 ){
                    if (((ypnt) < 0) || ((ypnt) > canvas.rows) || ((xpnt) < 0) || (xpnt > (canvas.cols)) ){
                        end_of_edge = true;
                        break;
                    }
                    edge_length++;
                    canvas.at<uchar>(ypnt+i, xpnt+j)=edge_no;
                    find_flag=true;
                    ypnt=ypnt+i;
                    xpnt=xpnt+j;
                    break;
                }
            }
            if ((find_flag==true) || (end_of_edge == true)){
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


