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
void sobel(const Mat& image, Mat& sob_result, Mat& nmr_result, Mat& grd_map, Mat& angle_map, uchar thresh);
int find_edge_component(Mat& canvas, int ypnt, int xpnt, int edge_no);
Mat edge_drawing(Mat& grad_map, Mat& nmr_result, Mat& anch_canvas, Mat& angle_map, Mat& edge_angle_map, int anchor_detail_ratio);


int main(){
    double t2 = (double)getTickCount();

   //load img
    //Mat input_gray_img = imread("lena.jpg",IMREAD_GRAYSCALE);
    //Mat input_gray_img = imread("drug.jpg",IMREAD_GRAYSCALE);
    //Mat input_gray_img = imread("peppers.png",IMREAD_GRAYSCALE);
    //Mat input_gray_img = imread("coins.jpg",IMREAD_GRAYSCALE);
    Mat input_gray_img = imread("circle2.jpg",IMREAD_GRAYSCALE);
    
    Mat gauss_result ;

    //gaussian(input_gray_img,gauss_result);
    GaussianBlur( input_gray_img, gauss_result, Size(5,5), 1, 1);

    
    int thr=20; //prewitt actually..
    Mat angle_map,nmr_result, grad_map , sob_result;
    sobel(gauss_result, sob_result, nmr_result, grad_map, angle_map, thr);

    int anchor_detail_ratio=4;
    Mat anch_canvas, edge_angle_map;
    Mat edge_canvas=edge_drawing(grad_map, nmr_result, anch_canvas, angle_map, edge_angle_map, anchor_detail_ratio );

    t2 = ((double)getTickCount() - t2) / getTickFrequency();
    cout << "time ptr =  " << t2 << " sec" << endl;
    
   // imshow("gray", input_gray_img);
    imshow("gauss", gauss_result);
    imshow("gradient map", grad_map);
    
    imshow("angle map", angle_map);

   // imshow("sobel/prewitt", sob_result);
    imshow("nonmaxima suppress", nmr_result);
    imshow("anch map", anch_canvas);
    imshow("edge map", edge_canvas);

    waitKey(0);
    return 0;
}

void gaussian(const Mat& input_gray_img, Mat& g_result){
    int height = input_gray_img.rows;
    int width = input_gray_img.cols;
    g_result.create(input_gray_img.size(), input_gray_img.type());

    //gaussian kernel 5*5 sigma 1
    Mat kernel_mat = (Mat_<double>(5,5) << 0.003765,    0.015019,    0.023792,    0.015019,    0.003765,
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
                    actv+=imgptr[k][j-kernel_size/2]*kernel_mat.at<double>(k,l);
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

void sobel(const Mat& image, Mat& sob_result, Mat& nmr_result, Mat& grd_map, Mat& angle_map, uchar thresh){
    //mask for each direction --prewitt
    Mat mask_x = (Mat_<double>(3, 3) << -1, 0, 1,
                                        -1, 0, 1,
                                        -1, 0, 1);
    Mat mask_y = (Mat_<double>(3, 3) << -1, -1, -1,
                                        0, 0, 0,
                                        1, 1, 1);
//
//    //mask for each direction  --sobel
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
    angle_map = Mat::zeros(image.rows - filterOffset * 2, image.cols - filterOffset * 2, image.type());
    
    double dx, dy;
    double gradient, angle;
    const double deg_factor=57.2957; // 180/pi
    
    for (int yimage = filterOffset; yimage < image.rows - filterOffset; yimage++){
        for (int ximage = filterOffset; ximage < image.cols - filterOffset; ximage++){
            dx = 0; dy = 0;
            for (int ymask = -filterOffset; ymask <= filterOffset; ymask++){
                for (int xmask = -filterOffset; xmask <= filterOffset; xmask++){
                    //x
                    dx += image.at<uchar>(yimage + ymask, ximage + xmask) * mask_x.at<double>(filterOffset + ymask, filterOffset + xmask);
                    //y
                    dy += image.at<uchar>(yimage + ymask, ximage + xmask) * mask_y.at<double>(filterOffset + ymask, filterOffset + xmask);
                }
            }
            gradient = sqrt(pow(dy, 2) + pow(dx, 2));
                if (gradient>=255){
                    //gradient=255;
                }
            if(gradient >= thresh){
                grd_map.at<uchar>(yimage - filterOffset, ximage - filterOffset) = gradient/2;
                sob_result.at<uchar>(yimage - filterOffset, ximage - filterOffset) = 255;

                angle=atan2(dy,dx)*deg_factor;
                angle= ((angle <0) ? (angle+180):angle);
                angle_map.at<uchar>(yimage-filterOffset,ximage-filterOffset)=angle;
            }
        }
    }
    
    //do Non-maxima suppression
    for (int yimage = filterOffset; yimage < image.rows - filterOffset; yimage++){
        for (int ximage = filterOffset; ximage < image.cols - filterOffset; ximage++){
                //thresholding & non maxima suppression
            gradient = grd_map.at<uchar>(yimage - filterOffset, ximage - filterOffset);
            angle = angle_map.at<uchar>(yimage-filterOffset,ximage-filterOffset);
            if(gradient >= thresh){
                double neighbor0=0 , neighbor1=0;
                //case0 angle0
                if (((0 <= angle) && (angle < 22.5)) || ((157.5 <= angle) && (angle <= 180))){
                    neighbor0 = grd_map.at<uchar>(yimage - filterOffset,ximage - filterOffset+1);
                    neighbor1 = grd_map.at<uchar>(yimage - filterOffset,ximage - filterOffset-1);
                }
                //case1 angle45
                else if ((22.5 <= angle) && (angle < 67.5)){
                    neighbor0 = grd_map.at<uchar>(yimage - filterOffset+1,ximage - filterOffset-1);
                    neighbor1 = grd_map.at<uchar>(yimage - filterOffset-1,ximage - filterOffset+1);
                }
                //case2 angle90
                else if ((67.5 <= angle) && (angle < 112.5)){
                    neighbor0 = grd_map.at<uchar>(yimage - filterOffset+1,ximage - filterOffset);
                    neighbor1 = grd_map.at<uchar>(yimage - filterOffset-1,ximage - filterOffset);
                }
                //case3 angle 135
                else if ((112.5 <= angle) && (angle < 157.5)){
                    neighbor0 = grd_map.at<uchar>(yimage - filterOffset-1,ximage - filterOffset-1);
                    neighbor1 = grd_map.at<uchar>(yimage - filterOffset+1,ximage - filterOffset+1);
                }

                if (gradient >= neighbor0 && gradient >=neighbor1){
                    nmr_result.at<uchar>(yimage - filterOffset, ximage - filterOffset)= gradient;
                }else{
                    nmr_result.at<uchar>(yimage - filterOffset, ximage - filterOffset)=0;
                }
                
            }
        }
    }
    
    nmr_result.row(1).setTo(Scalar(0));
    nmr_result.row(nmr_result.rows-1).setTo(Scalar(0));
    nmr_result.col(1).setTo(Scalar(0));
    nmr_result.col(nmr_result.cols-1).setTo(Scalar(0));
    
    grd_map.row(1).setTo(Scalar(0));
    grd_map.row(grd_map.rows-1).setTo(Scalar(0));
    grd_map.col(1).setTo(Scalar(0));
    grd_map.col(grd_map.cols-1).setTo(Scalar(0));
}

Mat edge_drawing(Mat& grad_map, Mat& nmr_result, Mat& anch_canvas, Mat& angle_map, Mat& edge_angle_map, int anchor_detail_ratio){

    Mat ed_canvas = Mat::zeros(nmr_result.rows, nmr_result.cols, nmr_result.type());
    edge_angle_map = Mat::zeros(nmr_result.rows, nmr_result.cols, nmr_result.type());

    nmr_result.copyTo(anch_canvas);
    //reducing anchors by detail ratio  by rows
    for(int i=0; i<anch_canvas.rows; i++){
        if(i%anchor_detail_ratio!=0){
            anch_canvas.row(i)=0;
        }
    }
    //reducing anchors by detail ratio  by cols
    for(int i=0; i<anch_canvas.cols; i++){
        if(i%anchor_detail_ratio!=0){
            anch_canvas.col(i)=0;
        }
    }

    anch_canvas.setTo(255, anch_canvas > 0);
    
    int edge_color=70;
    int anch_color=255;
    int ypnt, xpnt, ytemp, xtemp, temp_grad, grad_max, direction_y, direction_x, direction_from_y, direction_from_x;
    int length, line_no=0;
    int idx_y[6]={0}, idx_x[6]={0};
    double angle;
    bool finding_next_anchor;
    
    for(int yinit = 1 ; yinit< anch_canvas.rows; yinit++){
        for(int xinit =1; xinit< anch_canvas.cols; xinit++){
            if(anch_canvas.at<uchar>(yinit,xinit)==anch_color){
                if (edge_color > 230){
                    edge_color=70;
                }

                edge_color = edge_color+1;
                length=1;

                //next step
                finding_next_anchor = true;
                ypnt=yinit;
                xpnt=xinit;
                direction_from_y=0;
                direction_from_x=0;
                
                while(finding_next_anchor==true){
                   
                    //check direction
                    angle = double(angle_map.at<uchar>(ypnt, xpnt));

//                    int probey=438;
//                    int probex=746;
//                    int probey_next= 442;
//                    int probex_next=747;

                     //case0 angle0
                    if (((0 <= angle) && (angle < 22.5)) || ((157.5 <= angle) && (angle <= 180))){
                        idx_y[0] = 1;     idx_x[0] = 1;
                        idx_y[1] = 1;     idx_x[1] = 0;
                        idx_y[2] = 1;     idx_x[2] = -1;
                        idx_y[3] = -1;    idx_x[3] = -1;
                        idx_y[4] = -1;    idx_x[4] = 0;
                        idx_y[5] = -1;    idx_x[5] = 1;
                    }
                    //case1 angle45
                    else if ((22.5 <= angle) && (angle < 67.5)){
                        idx_y[0] = 1;    idx_x[0] = -1;
                        idx_y[1] = 0;    idx_x[1] = 1;
                        idx_y[2] = 1;    idx_x[2] = 0;
                        idx_y[3] = -1;   idx_x[3] = 1;
                        idx_y[4] = -1;   idx_x[4] = 0;
                        idx_y[5] = 0;    idx_x[5] = -1;

                    }
                     //case2 angle90
                    else if ((67.5 <= angle) && (angle < 112.5)){
                        idx_y[0] = 0;     idx_x[0] = 1;
                        idx_y[1] = 1;     idx_x[1] = 1;
                        idx_y[2] = -1;    idx_x[2] = 1;
                        idx_y[3] = 0;     idx_x[3] = -1;
                        idx_y[4] = 1;     idx_x[4] = -1;
                        idx_y[5] = -1;    idx_x[5] = -1;

                    }
                     //case3 angle 135
                    else if ((112.5 <= angle) && (angle < 157.5)){
                        idx_y[0] = 1;     idx_x[0] = 1;
                        idx_y[1] = 1;     idx_x[1] = 0;
                        idx_y[2] = 0;     idx_x[2] = 1;
                        idx_y[3] = -1;    idx_x[3] = -1;
                        idx_y[4] = -1;    idx_x[4] = 0;
                        idx_y[5] = 0;     idx_x[5] = -1;
                    }
                     
                    grad_max=0;
                    temp_grad=0;
                    direction_y=0;
                    direction_x=0;
                                    
                    if (ypnt>= grad_map.rows || xpnt>=grad_map.cols || ypnt < 1 || xpnt < 1){
                        finding_next_anchor=false;
                        break;
                    }

//int y1=300, x1=504, y0= 299, x0=486;

                    for (int i = 0; i<6; i++){
                        ytemp=ypnt+idx_y[i];
                        xtemp=xpnt+idx_x[i];
                        temp_grad=grad_map.at<uchar>(ytemp,xtemp);

/* if (ypnt >= y0 && xpnt >= x0 &&  y1 >= ypnt && x1 >= xpnt  ){
    cout<<idx_y[i]<<"  " <<idx_x[i]<<endl;
    cout<<temp_grad<<endl;
    cout << "y from " <<direction_from_y<<" xfrom"<< direction_from_x<< endl;
    if ( (direction_from_y != idx_y[i])) {
        cout <<"true"<<endl;
    }
    else {cout<<"false"<<endl;}

    if ( ((direction_from_x) != idx_x[i]) ){
        cout <<"true"<<endl;
    }
    else {cout<<"false"<<endl;}
} */
                        if (  (direction_from_y != idx_y[i]) || ((direction_from_x) != idx_x[i]) ){

                            if (grad_max < temp_grad){
                                grad_max = temp_grad;
                                direction_y = idx_y[i];
                                direction_x = idx_x[i];
                            }
                        }
                        //grad_map.at<uchar>(ytemp,xtemp)=0;

                    }
/* 
if (ypnt >= y0 && xpnt >= x0 &&  y1 >= ypnt && x1 >= xpnt  ){
    cout<<"ypnt is "<< ypnt<<" xpnt is " << xpnt<<endl;
    cout <<"current grad is "<< int(grad_map.at<uchar>(ypnt,xpnt))<<endl;
    cout <<"next grad is "<< grad_max<<endl;
    cout <<"direction for y is "<< direction_y<<" direction for x is "<< direction_x<<endl;
    cout <<"edge color is "<< edge_color<<endl<<endl;
} */

                    ypnt=ypnt+direction_y;
                    xpnt=xpnt+direction_x;
                    direction_from_y=direction_y*-1;
                    direction_from_x=direction_x*-1;

                    /* if((direction_x==0 && direction_y==0) || grad_max==0){
                        finding_next_anchor=false;
                        ed_canvas.at<uchar>(ypnt,xpnt)= edge_color;
                        if (length>=2){
                            ed_canvas.at<uchar>(yinit,xinit)=edge_color;
                            ed_canvas.at<uchar>(ypnt,xpnt)= edge_color;
                        }                        
                        break;
                    } */

                    //found next anchor -pass
                    if(anch_canvas.at<uchar>(ypnt, xpnt)==255){
                     /*    //ed_canvas.at<uchar>(ypnt,xpnt)= edge_color;
                        finding_next_anchor=false;
                        break; */
                    }
                        
                    if(ed_canvas.at<uchar>(ypnt, xpnt)==edge_color){
                        finding_next_anchor=false;
                        if (length<=2){
                            ed_canvas.at<uchar>(yinit,xinit)=0;
                        }
                        break;
                    }

                    //overwrite existing points
                    if(ed_canvas.at<uchar>(ypnt, xpnt)>0){
                  /*       ed_canvas.at<uchar>(ypnt,xpnt)= 252;
                        finding_next_anchor=false;
                        break; */
                    }
                    
                    //mark start point 
                    if(length==1){
                        ed_canvas.at<uchar>(yinit,xinit)=255;}
                    //make pixel to edge
                    ed_canvas.at<uchar>(ypnt,xpnt)= edge_color;
                    length++;                
                }// while
                if (length>3){
                    cout<<"line no : "<<line_no+1<<"   length : "<<length<<endl;
                    line_no++;
                }
            }// if
        }// for inner
    }//for outer
    cout<<"total line no "<<line_no<<endl;
    return ed_canvas;
}


int find_edge_component(Mat& canvas, int ypnt, int xpnt, int edge_no){
    //find edge
    int edge_length=1;
    bool end_of_edge=false;
    canvas.at<uchar>(ypnt, xpnt)=254;
   
    //finding edge elements
    while (end_of_edge==false){
        int i=1, j=1;
        bool find_flag=false;
        for(i=1; i >-2; i--){
            for(j=1; j>-2;j--){
                ypnt=ypnt+i;
                xpnt=xpnt+j;
                    //and if not on the canvas -> exit
                    if (((ypnt) < 0) || ((ypnt) >= canvas.rows) || ((xpnt) < 0) || (xpnt >= (canvas.cols)) ){
                        break;
                    }
                    //if new point is bright
                    else if (canvas.at<uchar>(ypnt, xpnt)==255 && (i!=0 && j!=0) ){
                        edge_length++;
                        canvas.at<uchar>(ypnt, xpnt)=edge_no;
                        find_flag=true;
                        break;
                    }
            }
            if (find_flag==true){
                break;
            }
        }
        //cannot find any or outside of canvas
        if (((i==-2 && j==-2) && find_flag==false) || (ypnt < 0) || (ypnt >= canvas.rows) || (xpnt < 0) || (xpnt >= canvas.cols)){
            end_of_edge = true;
            break;
        }
    }
    return edge_length;
}


