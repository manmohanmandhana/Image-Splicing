#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;
/** @function main */

/*////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   1) Into K channels using AC filters from DCT decomposition    
   2) Divide the image into proper windows
   3) Compute image integrals for 4 image moments for each of the K channels
   4) Compute the varience and Kurtosis for each local window for each channel
   5) Estimate local variences for each of the local windows 
*////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double compute_kurtosis(){}
double compute_varience(){}
double estimate_localvarience(){}


int main()
{
  
}

