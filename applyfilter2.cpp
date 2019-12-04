
#include <math.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
int main()
{
Mat InImg = imread("lena.png", IMREAD_COLOR);

int height = InImg.rows;
int width = InImg.cols;

Mat img_grayscale(height, width, CV_8UC1);
Mat OrgImg(height, width, CV_8UC1,0);

int m_HistoArr[256];

int MaskGaussian[3][3]={{1,2,1},
                    {2,4,2},
                    {1,2,1}};
int heightm1=height-1;//중복계산을 피하려고
int widthm1=width-1;//중복계산을 피하려고
int mr,mc;
int newValue;
int i,j;



 for(i=1; i<heightm1; i++)
 {
  for(j=1; j<widthm1; j++)
  {
   newValue=0; //0으로 초기화
   for(mr=0;mr<3;mr++)
    for(mc=0;mc<3;mc++)
     newValue += (MaskGaussian[mr][mc]*InImg.at<uchar>[i+mr-1][j+mc-1]);
   newValue /= 16; //마스크의 합의 크기로 나누기:값의 범위를 0에서 255로 함
   OrgImg[i][j]=newValue;//BYTE값으로 변환 
  }
 }

imshow("Output", OrgImg);

waitKey(0);
return 0;

}
