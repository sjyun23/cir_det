//
//  cd.cpp
//  circle_detection
//
//  Created by YunSung-jae on 11/21/19.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stack>
#include <tuple>



using namespace cv;
using namespace std;

void gaussian(const Mat& input_gray_img, Mat& gauss_result);
void sobel(const Mat& image, Mat& sob_result, Mat& nmr_result, Mat& grd_map, Mat& angle_map, uchar thresh);
int find_edge_component(Mat& canvas, int ypnt, int xpnt, int edge_no);
Mat edge_drawing(Mat& grad_map, Mat& nmr_result, Mat& anch_canvas, Mat& angle_map, Mat& edge_angle_map, Mat& circle_map, int anchor_detail_ratio, double circle_threshold);
Mat edCircle(Mat& edge_map, Mat& edge_angle_map );

tuple<double, double, double> circle_fit(int point_list[][2], int length_point_list);


int main(){
    double t2 = (double)getTickCount();

   //load img
    //Mat input_gray_img = imread("lena.jpg",IMREAD_GRAYSCALE);
    //Mat input_gray_img = imread("drug.jpg",IMREAD_GRAYSCALE);
    //Mat input_gray_img = imread("peppers.png",IMREAD_GRAYSCALE);
    Mat input_gray_img = imread("coins.jpg",IMREAD_GRAYSCALE);
    //Mat input_gray_img = imread("circle2.jpg",IMREAD_GRAYSCALE);
    
    Mat gauss_result;
    //gaussian(input_gray_img,gauss_result);
    GaussianBlur( input_gray_img, gauss_result, Size(5,5), 1, 1);

    //sharpen
    int thr=20; //prewitt actually..
    Mat angle_map,nmr_result, grad_map , sob_result;
    sobel(gauss_result, sob_result, nmr_result, grad_map, angle_map, thr);

    //edpf
    int anchor_detail_ratio=3;
    double circle_threshold=0.5;
    Mat anch_canvas, edge_angle_map, circle_map;
    Mat edge_result=edge_drawing(grad_map, nmr_result, anch_canvas, angle_map, edge_angle_map, circle_map, anchor_detail_ratio, circle_threshold);

    //edcircle
    //edCircle(edge_result, edge_angle_map);

    t2 = ((double)getTickCount() - t2) / getTickFrequency();

    cout << "time elapesed =  " << t2 << " sec" << endl;
    
    //imshow("gray", input_gray_img);
    //imshow("gaussian", gauss_result);
    //imshow("gradient map", grad_map);
    //imshow("angle map", angle_map);
    //imshow("sobel/prewitt", sob_result);
    //imshow("nonmaxima suppress", nmr_result);
    //imshow("anch map", anch_canvas);
    //imshow("edge map", edge_result);
    imshow("edge-angle map", edge_angle_map);
    imshow("circle map", circle_map);

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
    //sob_result = Mat::zeros(image.rows - filterOffset * 2, image.cols - filterOffset * 2, image.type());
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
                //sob_result.at<uchar>(yimage - filterOffset, ximage - filterOffset) = 255;

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

Mat edge_drawing(Mat& grad_map, Mat& nmr_result, Mat& anch_canvas, Mat& angle_map, Mat& edge_angle_map, Mat& circle_map, int anchor_detail_ratio, double circle_threshold){

    Mat ed_canvas = Mat::zeros(nmr_result.rows, nmr_result.cols, nmr_result.type());
    edge_angle_map = Mat::zeros(nmr_result.rows, nmr_result.cols, nmr_result.type());
    circle_map = Mat::zeros(nmr_result.rows, nmr_result.cols, nmr_result.type());


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
    
    int edge_color=40;
    int anch_color=255;
    int ypnt, xpnt, ytemp, xtemp, temp_grad, grad_max;
    int direction_y, direction_x, direction_from_y, direction_from_x;
    int travel_length;
    int idx_y[6]={0}, idx_x[6]={0};
    double angle;
    bool finding_next_anchor;

    stack<int>stack_circle_center_x;
    stack<int>stack_circle_center_y;
    stack<int>stack_radius;
    stack<int>stack_arc_length;

    for(int yinit = anchor_detail_ratio ; yinit< anch_canvas.rows; yinit=yinit+anchor_detail_ratio){
        for(int xinit =anchor_detail_ratio; xinit< anch_canvas.cols; xinit=xinit+anchor_detail_ratio){
            if(anch_canvas.at<uchar>(yinit,xinit)==anch_color){

                if (edge_color > 230){
                    edge_color=40;
                }

                edge_color = edge_color+1;
                travel_length=1;

                //next step
                finding_next_anchor = true;
                ypnt=yinit;
                xpnt=xinit;
                direction_from_y=0;
                direction_from_x=0;
            
                stack<int>ystack_line;
                stack<int>xstack_line;


                while(finding_next_anchor==true){
                   
                    //check direction
                    angle = double(angle_map.at<uchar>(ypnt, xpnt));

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

                    for (int i = 0; i<6; i++){
                        ytemp=ypnt+idx_y[i];
                        xtemp=xpnt+idx_x[i];
                        temp_grad=grad_map.at<uchar>(ytemp,xtemp);

                        if (  (direction_from_y != idx_y[i]) || ((direction_from_x) != idx_x[i]) ){

                            if (grad_max < temp_grad){
                                grad_max = temp_grad;
                                direction_y = idx_y[i];
                                direction_x = idx_x[i];
                            }
                        }
                    }

                    ypnt=ypnt+direction_y;
                    xpnt=xpnt+direction_x;

                    //if head meets tail, stop.    
                    if(ed_canvas.at<uchar>(ypnt, xpnt)==edge_color){
                        finding_next_anchor=false;
                        if (travel_length<=2){
                            ed_canvas.at<uchar>(yinit,xinit)=0;
                        }
                        break;
                    }

                    //make pixel to edge
                    ed_canvas.at<uchar>(ypnt,xpnt)= edge_color;
                    edge_angle_map.at<uchar>(ypnt,xpnt)=angle_map.at<uchar>(ypnt,xpnt);

                    ystack_line.push(ypnt);
                    xstack_line.push(xpnt);

                    travel_length++;

                    direction_from_y=direction_y*-1;
                    direction_from_x=direction_x*-1;

                }// while

                    //check length
                if(ystack_line.size()>=10){
                    int ypnt_stack_size=ystack_line.size();
                    int xpnt, ypnt, xpnt_before, ypnt_before, dy, dx;
                    int delta,delta_angle_edges=0;
                    int angle_position_before=0;
                    int segment_table[ypnt_stack_size][3];

                    stack<int>changepoint_index;


                    for(int i =0 ; i<ypnt_stack_size ; i++){
                        for(int j =0 ; j<3; j++){
                            segment_table[i][j]=0;
                        }
                    }

                    int length_list[ypnt_stack_size];
                    for(int i =0 ; i<ypnt_stack_size ; i++){
                        length_list[i]=0.0;                        
                    }

                    //get line elements from stack and put in array
                    for(int i =0 ; i<ypnt_stack_size ; i++){
                        segment_table[i][0]=ystack_line.top();
                        ystack_line.pop();
                        segment_table[i][1]=xstack_line.top();
                        xstack_line.pop();

                        if (i>0){
                            ypnt= segment_table[i][0];
                            xpnt= segment_table[i][1];
                            ypnt_before= segment_table[i-1][0];
                            xpnt_before= segment_table[i-1][1];
                            
                            delta=abs(angle_map.at<uchar>(ypnt,xpnt)-angle_map.at<uchar>(ypnt_before,xpnt_before));
                            if ( delta > 90){
                                delta=(180-delta);
                            }                                                           

                            if (6>delta){

                            }else if (60 > delta ){
                                //angle between edges 
                                delta_angle_edges = angle_map.at<uchar>(segment_table[i][0],segment_table[i][1])-angle_map.at<uchar>(segment_table[angle_position_before][0],segment_table[angle_position_before][1]);
                                if (delta_angle_edges < -90){
                                    delta_angle_edges += 180;
                                }else if(delta_angle_edges > 90){
                                    delta_angle_edges -= 180;
                                }
                                segment_table[i][2]= delta_angle_edges;

                                if(segment_table[i][2] * segment_table[angle_position_before][2] < 0 ){
                                    //put in stack
                                    changepoint_index.push(i);
                                }

                                //write edge length
                                for(int j=angle_position_before+1; j <= i; j++){
                                    dy=segment_table[j][0]-segment_table[j-1][0];
                                    dx=segment_table[j][1]-segment_table[j-1][1];
                                    length_list[i]+=sqrt(pow(dx,2)+pow(dy,2));
                                }

                                angle_position_before=i;
                            }else if (60<delta ){
                                changepoint_index.push(i);

                            }                               
                        }      
                    }//for 

                    tuple<double, double, double> circle_info;
                    double circle_center_y=0.0, circle_center_x=0.0, radius=0.0, arc_length;
                    int old_idx=0, idx=0, cnt=1;

                    int changepoint_index_size=changepoint_index.size();

                    if(changepoint_index_size>0){

                        int change_point_list[changepoint_index_size+1]; 
                        
                        while(!changepoint_index.empty()){
                            idx=changepoint_index.top();
                            changepoint_index.pop();
                            change_point_list[changepoint_index_size-cnt]=idx;
                            cnt++;
                        }
                        change_point_list[changepoint_index_size]=ypnt_stack_size-1;

                        arc_length=0.0;
                        for (int i= 0; i<=changepoint_index_size; i++){
                            idx=change_point_list[i];
                            
                            if ((idx-old_idx)<=10){
                                old_idx=idx;
                                continue;
                            }

                            int point_list[idx-old_idx][2];

                            for (int j=old_idx+1; j < idx; j++){
                                point_list[j-old_idx-1][0]=segment_table[j][0];
                                point_list[j-old_idx-1][1]=segment_table[j][1];
                                arc_length+=length_list[j];
                                //circle_map.at<uchar>(segment_table[j][0],segment_table[j][1])=30;
                            }
                            circle_info = circle_fit(point_list, idx-old_idx);
                            circle_center_y = int(get<0>(circle_info));
                            circle_center_x = int(get<1>(circle_info));
                            radius = int(get<2>(circle_info));

                            //cout<<circle_center_x<<" "<<circle_center_y<<" "<<radius<<" "<<arc_length<<endl;

                            old_idx=idx;

                        }

                    }else if (changepoint_index_size==0){
                        int point_list[ypnt_stack_size][2];

                        for(int i=0; i < ypnt_stack_size; i++){
                                point_list[i][0]=segment_table[i][0];
                                point_list[i][1]=segment_table[i][1];
                                arc_length+=length_list[i];

                                circle_map.at<uchar>(segment_table[i][0],segment_table[i][1])=30;
                        }
                        circle_info = circle_fit(point_list, ypnt_stack_size);
                        circle_center_y = int(get<0>(circle_info));
                        circle_center_x = int(get<1>(circle_info));
                        radius = int(get<2>(circle_info));
                    }
                    
                    if (get<0>(circle_info) < circle_map.rows && get<1>(circle_info) < circle_map.cols && get<2>(circle_info) > 5){
                        if (get<0>(circle_info) >0 && get<1>(circle_info) >0){
                            
                            circle_center_y = int(get<0>(circle_info));
                            circle_center_x = int(get<1>(circle_info));
                            
                            if(circle_map.at<uchar>(circle_center_y, circle_center_x)==0){
                            circle_map.at<uchar>(circle_center_y, circle_center_x)=100;//100;
                            }else{circle_map.at<uchar>(circle_center_y, circle_center_x)+=10;
                            }
                        }
                    } 
                                                   

                }// if stack empty   
            }// if
        }// for inner
    }//for outer
    return ed_canvas;
}

tuple<double, double, double> circle_fit(int point_list[][2], int length_point_list){
    double center_y=0, center_x=0, radius=0.0;
    double u, v, detMat;
    double uu=0.0, vv=0.0, uv=0.0, uuu=0.0, vvv=0.0, uvv=0.0, vuu=0.0;
    double x_acc = 0.0, y_acc=0.0, x_mean=0.0, y_mean=0.0;
    int x, y;

    for (int i = 0; i < length_point_list; i++){
        //cout<<point_list[i][0]<<"   " <<point_list[i][1]<<endl;
        x=point_list[i][1];  //x
        y=point_list[i][0];  //y

        x_acc+=x;
        y_acc+=y;
    }

    x_mean=x_acc/length_point_list;
    y_mean=y_acc/length_point_list;

    for (int i = 0; i < length_point_list; i++)
        {
            //cout<<point_list[i][0]<<"   " <<point_list[i][1]<<endl;
            u=point_list[i][1]-x_mean;  //x
            v=point_list[i][0]-y_mean;  //y

            uu+=pow(u,2);
            uv+=u*v;
            vv+=pow(v,2);

            uuu+=pow(u,3);
            vvv+=pow(v,3);
            uvv+=u*pow(v,2);
            vuu+=v*pow(u,2);
        }

        detMat = uu*vv-uv*uv;
        center_x = (vv*(uuu+uvv) - uv*(vvv+vuu)) * 1/(2*detMat);
        center_y = (uu*(vvv+vuu) - uv*(uuu+uvv)) * 1/(2*detMat);
        radius = sqrt(pow(center_x,2) + pow(center_y,2) + (uu+vv)/length_point_list);
        
        center_x+=x_mean;
        center_y+=y_mean; 

        if (detMat==0){
            radius=0;
        }
        
//cout<<center_x<<" "<< center_y <<" "<< radius <<endl;

    return make_tuple(center_y, center_x, radius);
}


