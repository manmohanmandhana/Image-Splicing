#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


/*////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   1) 32x32 overlapping local windows. Each window will have 63 channels corrosponding to 8x8 DCT kernal
   2) Estimate the image for each channel.    
   3) Compute 4 image moments for each of the K channels
   4) Compute the varience and Kurtosis for each local window for each channel
   5) Estimate local variences for each of the local windows 
*////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double moment1(){
    int ii = i-15; 
    int jj = j-15;
    double sum = 0;
    for(int k=ii;k<ii+32;k++){
        for(int kk=jj;kk<jj+32;kk++){
            sum = sum + image.at<uchar>(k,kk);
        }
    }
    sum = sum/(32*32);
    return sum;
}

double moment2(int i, int j, Mat image){
    int ii = i-15; 
    int jj = j-15;
    double sum = 0;
    for(int k=ii;k<ii+32;k++){
        for(int kk=jj;kk<jj+32;kk++){
            sum = sum + (image.at<uchar>(k,kk))*(image.at<uchar>(k,kk));
        }
    }
    sum = sum/(32*32);
    return sum;
}

double moment3(int i, int j, Mat image){
    int ii = i-15; 
    int jj = j-15;
    double sum = 0;
    for(int k=ii;k<ii+32;k++){
        for(int kk=jj;kk<jj+32;kk++){
            sum = sum + (image.at<uchar>(k,kk))*(image.at<uchar>(k,kk))*(image.at<uchar>(k,kk));
        }
    }
    sum = sum/(32*32);
    return sum;
}

double moment4(int i, int j, Mat image){
    int ii = i-15; 
    int jj = j-15;
    double sum = 0;
    for(int k=ii;k<ii+32;k++){
        for(int kk=jj;kk<jj+32;kk++){
            sum = sum + (image.at<uchar>(k,kk))*(image.at<uchar>(k,kk))*(image.at<uchar>(k,kk))*(image.at<uchar>(k,kk));
        }
    }
    sum = sum/(32*32);
    return sum;
}

// Kurtosis Computation
double compute_kurtosis(double mu1, double mu2, double mu3, double mu4){
    double kurt;
    kurt = mu4 - 4*mu3*mu1 + 6*mu2*mu1*mu1 - 3*mu1*mu1*mu1*mu1;
    kurt = kurt/(mu2*mu2 - 2*mu2*mu1*mu1 + mu1*mu1*mu1*mu1);
    kurt = kurt - 3;
    return kurt;
}

// Varience Computation
double compute_varience(double mu1, double mu2){
    double var;
    var = mu2 - mu1*mu1;
    return var;
}

// Final Estimation of local varience
double estimate_localvarience(){}


int main()
{
    // Step 1 to divede the image into 8x8 DCT decomposition windows and hence K=63
    // divide into 32x32 windows and then do the 8x8 window division and perform DCT decomposition
    // Do we need to conver to grayscale???
    Mat img_input;
    img_input = imread("1.jpg",1);
    cvtColor(img_input,img_input,CV_BGR2GRAY);
    int totrows = img_input.rows;
    int totcols = img_input.cols;
    
    // now we need 32x32 overlapping windows for the image
    // Yet to apply it on the whole of the image

    Mat window_img = img_input(cv::Rect(0,0,32,32)); // the first window of 32x32
    cout << window_img.rows << window_img.cols << endl;

    //Mat temp = Mat::zeros( window_img.size(), window_img.type());
    Mat temp8 = Mat::zeros( 8,8, window_img.type());
    
    // this represents the stack of 64 images in each local window 
    double local_channels[32][32][64] = 0;

    int k = 64;
    int m[k];
    
    // a local window analysis starts
    for(int i = 0;i<25;i++){
        for(int j = 0;j<25;j++){
            temp8 = window_img(cv::Rect(i,j,8,8));
            
            // DCT of the window here
            dct(temp8,temp8,0);
            // Mat to uchar array
            uchar *nice = temp8.data;
            for(int l=0;l<k;l++){ m[l] = nice[l]; }            

            // The 1st channel is redundent as it is the DC component
            for(int ii=i;ii<i+8;ii++){
                for(int jj=j;jj<j+8;jj++){
                    for(k = 0;k<64;k++){
                        local_channels[ii][jj][k] = local_channels[ii][jj][k] + m[k];
                    }
                }
            } 
        }
    }
    
    //Now we process this window through the desired functions
    //int temp[32][32]=0;
    for(int k=1;k<64;k++){
       
       // all 63 AC channels, compute an image for each of them first
       // int array to image??
          
    }
    
    waitKey(0);
    return 0;

}

