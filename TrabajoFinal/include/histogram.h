#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <limits>

using namespace cv;
using namespace std;

namespace lbp {

template <typename _Tp>
void histogram_(const Mat& src, Mat& hist, int numPatterns);

template <typename _Tp>
double chi_square_(const Mat& histogram0, const Mat& histogram1);

void spatial_histogram(const Mat& src, Mat& spatialhist, int numPatterns, const Size& window, int overlap=0);

void spatial_histogram(const Mat& src, Mat& spatialhist, int numPatterns, int gridx=8, int gridy=8, int overlap=0);
void histogram(const Mat& src, Mat& hist, int numPatterns);
double chi_square(const Mat& histogram0, const Mat& histogram1);

Mat histogram(const Mat& src, int numPatterns);
Mat spatial_histogram(const Mat& src, int numPatterns, const Size& window, int overlap=0);
Mat spatial_histogram(const Mat& src, int numPatterns, int gridx=8, int gridy=8, int overlap=0);
}
#endif
