#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


/*////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   1) Into K channels using AC filters from DCT decomposition    
   2) Divide the image into proper windows
   3) Compute 4 image moments for each of the K channels
   4) Compute the varience and Kurtosis for each local window for each channel
   5) Estimate local variences for each of the local windows 
*////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double moment1(int i, int j, Mat image){
    int ii = i-15; 
    int jj = j-15;
    double sum = 0;
    for(int k=ii,k<ii+32,k++){
        for(int kk=jj,kk<jj+32,kk++){
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
    for(int k=ii,k<ii+32,k++){
        for(int kk=jj,kk<jj+32,kk++){
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
    for(int k=ii,k<ii+32,k++){
        for(int kk=jj,kk<jj+32,kk++){
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
    for(int k=ii,k<ii+32,k++){
        for(int kk=jj,kk<jj+32,kk++){
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
    Mat img_input;
    img_input = imread('1.jpg',1);
    totrows = img_input.rows;
    totcols = img_input.cols;
    

}

