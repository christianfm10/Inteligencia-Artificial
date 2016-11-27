//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//#include "opencv2/img_hash.hpp"
////
////#include "iostream"
////#include <stdio.h>
////#include <math.h>
////#include <stdlib.h>
////#include <stdio.h>
//using namespace cv;
////
////
//Mat __Sobel(Mat dest){
//   Mat src;
//   Mat src_gray;
//   Mat grad;
//  int scale = 1;
//  int delta = 0;
//  int ddepth = CV_16S;
//  GaussianBlur( dest, src, Size(3,3), 0, 0, BORDER_DEFAULT );
//  cvtColor( src, src_gray, CV_BGR2GRAY );
//  Mat grad_x, grad_y;
//  Mat abs_grad_x, abs_grad_y;
//  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
//  convertScaleAbs( grad_x, abs_grad_x );
//  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
//  convertScaleAbs( grad_y, abs_grad_y );
//  addWeighted( abs_grad_x, 0.3, abs_grad_y, 0.3, 0, grad );
//    return grad;
//
//
//}
//////using namespace cv;
//////using namespace std;
////////int hamming(Figura const &input, Figura const &target)
////////{
////////    cv::Mat inHash;
////////    cv::Mat outHash;
////////    img_hash::averageHash(input.imagen, inHash);
////////    img_hash::averageHash(target.imagen, outHash);
////////    int const averageMismatch = norm(inHash, outHash, NORM_HAMMING);
////////    return averageMismatch;
//////////    std::cout<<"averageMismatch : "<<averageMismatch<<std::endl;
////////}
//////
//////int xGradient(Mat image, int x, int y)
//////{
//////    return image.at<uchar>(y-1, x-1) +
//////                2*image.at<uchar>(y, x-1) +
//////                 image.at<uchar>(y+1, x-1) -
//////                  image.at<uchar>(y-1, x+1) -
//////                   2*image.at<uchar>(y, x+1) -
//////                    image.at<uchar>(y+1, x+1);
//////}
//////
//////// Computes the y component of the gradient vector
//////// at a given point in a image
//////// returns gradient in the y direction
//////
//////int yGradient(Mat image, int x, int y)
//////{
//////    return image.at<uchar>(y-1, x-1) +
//////                2*image.at<uchar>(y-1, x) +
//////                 image.at<uchar>(y-1, x+1) -
//////                  image.at<uchar>(y+1, x-1) -
//////                   2*image.at<uchar>(y+1, x) -
//////                    image.at<uchar>(y+1, x+1);
//////}
//////int main()
//////{
////////int scale = 1;
////////int delta = 0;
////////int ddepth = CV_16S;
////////    Mat src_base, hsv_base;
////////    Mat src_test1, hsv_test1;
////////    Mat src_test2, hsv_test2;
////////    Mat hsv_half_down;
//////    Mat img,dst;
//////    Mat src2;
//////    img = imread("50.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//////    cout<<img.col(0);
////////    cout<<src1.col(2);
////////        cout<<img.row(0);
////////        cout<<img.size().width;
//////////        for(unsigned i=0;i<10000;i++)
//////////                img.at<uchar>(i)=0;
////////
////////        Scalar intensity = img.at<uchar>(0);
//////        int s = img.at<uchar>(0);
//////        cout<<s<<endl;
//////    cout<<(int)img.at<uchar>(4,0)<<endl;
////////    int sobel_x[3][3];
////////    int sobel_y[3][3];
//    sobel_x[0][0]=-1;
//    sobel_x[0][1]= 0;
//    sobel_x[0][2]= 1;
//    sobel_x[1][0]=-2;
//    sobel_x[1][1]= 0;
//    sobel_x[1][2]= 2;
//    sobel_x[2][0]=-1;
//    sobel_x[2][1]= 0;
//    sobel_x[2][2]= 1;
//
//    sobel_y[0][0]=-1;
//    sobel_y[0][1]=-2;
//    sobel_y[0][2]=-1;
//    sobel_y[1][0]= 0;
//    sobel_y[1][1]= 0;
//    sobel_y[1][2]= 0;
//    sobel_y[2][0]= 1;
//    sobel_y[2][1]= 2;
//    sobel_y[2][2]= 1;
////////    for(unsigned i=0;i<3;i++){
////////        for(unsigned j=0;j<3;j++)
////////            cout<<sobel_y[i][j];
////////        cout<<endl;
////////    }
////////    int pixel_x=0;
////////    int pixel_y=0;
////////    int val=0;
//////////    for x in 1..img.width-2
//////////  for y in 1..img.height-2
//    for(int x=1;   x<img.size().width-2;x++){
//        for(int y=1;   y<img.size().height-2;y++){
//    pixel_x = (sobel_x[0][0] * img.at<uchar>(x-1,y-1)) + (sobel_x[0][1] * img.at<uchar>(x,y-1)) + (sobel_x[0][2] * img.at<uchar>(x+1,y-1)) +
//              (sobel_x[1][0] * img.at<uchar>(x-1,y))   + (sobel_x[1][1] * img.at<uchar>(x,y))   + (sobel_x[1][2] * img.at<uchar>(x+1,y)) +
//              (sobel_x[2][0] * img.at<uchar>(x-1,y+1)) + (sobel_x[2][1] * img.at<uchar>(x,y+1)) + (sobel_x[2][2] * img.at<uchar>(x+1,y+1));
////    cout<<pixel_x<<endl;
//    pixel_y = (sobel_y[0][0] * img.at<uchar>(x-1,y-1)) + (sobel_y[0][1] * img.at<uchar>(x,y-1)) + (sobel_y[0][2] * img.at<uchar>(x+1,y-1)) +
//              (sobel_y[1][0] * img.at<uchar>(x-1,y))   + (sobel_y[1][1] * img.at<uchar>(x,y))   + (sobel_y[1][2] * img.at<uchar>(x+1,y)) +
//              (sobel_y[2][0] * img.at<uchar>(x-1,y+1)) + (sobel_y[2][1] * img.at<uchar>(x,y+1)) + (sobel_y[2][2] * img.at<uchar>(x+1,y+1));
////////
//////////    val = Math.sqrt((pixel_x * pixel_x) + (pixel_y * pixel_y)).ceil
////////    val = ceil(sqrt((pixel_x * pixel_x) + (pixel_y * pixel_y)));
////////    img.at<uchar>(x,y)=val << 24 | val << 16 | val << 8 | 0xff;
//////////    edge[x,y] = ChunkyPNG::Color.grayscale(val)
////////        }
////////    }
//////////    val=2;
////////    val= val << 24 | val << 16 | val << 8 | 0xff;
////////    cout<<val;
//////dst = img.clone();
//////int gx, gy, sum;
//////        for(int y = 0; y < img.rows; y++)
//////            for(int x = 0; x < img.cols; x++)
//////                dst.at<uchar>(y,x) = 0.0;
//////
//////        for(int y = 1; y < img.rows - 1; y++){
//////            for(int x = 1; x < img.cols - 1; x++){
//////                gx = xGradient(img, x, y);
//////                gy = yGradient(img, x, y);
//////                sum = abs(gx) + abs(gy);
//////                sum = sum > 255 ? 255:sum;
//////                sum = sum < 0 ? 0 : sum;
//////                dst.at<uchar>(y,x) = sum;
//////            }
//////        }
//////
//////    namedWindow( "Imagen original", CV_WINDOW_AUTOSIZE );
//////    imshow( "Imagen original", dst );
//////
////////  end
////////end
////////sobel_x = [[-1,0,1],
////////           [-2,0,2],
////////           [-1,0,1]]
////////
////////sobel_y = [[-1,-2,-1],
////////           [0,0,0],
////////           [1,2,1]]
//////
////////    cout<<src1.col(1);
//////
////////    Mat img_bw = src1 > 128;
//////
////////    src2 = imread("1.jpg", CV_LOAD_IMAGE_COLOR);
////////    namedWindow( "Imagen original", CV_WINDOW_AUTOSIZE );
////////    imshow( "Imagen original", src2 );
////////    cv::Mat inHash;
////////    Mat grad_x, grad_y;
////////Mat abs_grad_x, abs_grad_y;
////////Mat grad;
//////
///////// Gradient X
////////Sobel( src1, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
/////////// Gradient Y
////////Sobel( src1, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
////////
////////    convertScaleAbs( grad_x, abs_grad_x );
////////convertScaleAbs( grad_y, abs_grad_y );
////////addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
////////namedWindow( "Original image", CV_WINDOW_AUTOSIZE );
////////    imshow( "Original image", grad ); cvMatc
////////    cv::Mat outHash;
////////    img_hash::averageHash(src1, inHash);
////////
//////
//////    waitKey(0);
////////
////////    Mat grey;
////////    cvtColor(src1, grey, CV_BGR2GRAY);
////////    Mat grey2;
////////    cvtColor(src2, grey2, CV_BGR2GRAY);
////////
////////    Mat sobelx;
////////    Sobel(grey, sobelx, CV_32F, 1, 0);
////////    Mat sobelx2;
////////    Sobel(grey2, sobelx2, CV_32F, 1, 0);
////////
////////    double minVal, maxVal;
////////    minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
////////    cout << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl;
////////    double minVal2, maxVal2;
////////    minMaxLoc(sobelx2, &minVal2, &maxVal2); //find minimum and maximum intensities
////////    cout << "minVal2 : " << minVal2 << endl << "maxVal2 : " << maxVal2 << endl;
////////
////////    Mat draw;
////////    sobelx.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
////////    Mat draw2;
////////    sobelx2.convertTo(draw2, CV_8U, 255.0/(maxVal2 - minVal2), -minVal2 * 255.0/(maxVal2 - minVal2));
////////
////////    namedWindow("image", CV_WINDOW_AUTOSIZE);
////////    imshow("image", draw);
////////    namedWindow("imagenes", CV_WINDOW_AUTOSIZE);
////////    imshow("imagenes", draw2);
////////
////////
////////    cvtColor( src1, hsv_base, COLOR_BGR2HSV );
////////    cvtColor( src2, hsv_test1, COLOR_BGR2HSV );
////////
////////    int h_bins = 50; int s_bins = 60;
////////    int histSize[] = { h_bins, s_bins };
////////
////////    float h_ranges[] = { 0, 180 };
////////    float s_ranges[] = { 0, 256 };
////////
////////    const float* ranges[] = { h_ranges, s_ranges };
////////
////////    int channels[] = { 0, 1 };
////////    MatND hist_base;
////////    MatND hist_test1;
////////
////////    calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
////////    normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
////////
////////    calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
////////    normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );
////////
////////    for( int i = 0; i < 4; i++ )
////////    {
////////        int compare_method = i;
////////        double base_base = compareHist( hist_base, hist_base, compare_method );
////////        double base_test1 = compareHist( hist_base, hist_test1, compare_method );
////////
////////        printf( " Method [%d] Perfect, Base-Test : %f, %f \n", i, base_base,  base_test1  );
////////    }
////////    printf( "Done \n" );
////////    waitKey(0);
//////    return 0;
//////}
////
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//
//
//
//
//int xGradient(Mat image, int x, int y)
//{
//    return image.at<uchar>(y-1, x-1) + 2*image.at<uchar>(y, x-1) + image.at<uchar>(y+1, x-1) - image.at<uchar>(y-1, x+1) - 2*image.at<uchar>(y, x+1) - image.at<uchar>(y+1, x+1);
//}
//
//
//
//int yGradient(Mat image, int x, int y)
//{
//    return image.at<uchar>(y-1, x-1) + 2*image.at<uchar>(y-1, x) + image.at<uchar>(y-1, x+1) - image.at<uchar>(y+1, x-1) - 2*image.at<uchar>(y+1, x) - image.at<uchar>(y+1, x+1);
//}
//Mat _Sobel(Mat dest){
//    Mat dst;
//    cvtColor( dest, dest, CV_BGR2GRAY );
//    dst = dest.clone();
//    int gx, gy, sum;
//        for(int y = 0; y < dest.rows; y++)
//            for(int x = 0; x < dest.cols; x++)
//                dst.at<uchar>(y,x) = 0.0;
//
//        for(int y = 1; y < dest.rows - 1; y++){
//            for(int x = 1; x < dest.cols - 1; x++){
//                gx = xGradient(dest, x, y);
//                gy = yGradient(dest, x, y);
////                sum = abs(gx) + abs(gy);
//                sum = ceil(sqrt(gx*gx + gy*gy));
//                sum = sum > 255 ? 240:sum;
//                sum = sum < 150 ? 0 : sum;
//                dst.at<uchar>(y,x) = sum;
//            }
//        }
//    return dst;
//}
//int main( int argc, char** argv )
//{
//
//  Mat src;
//  Mat sketch;
//  Mat dest_foto;
//  Mat dest_sketch;
//  Mat grad;
//  Mat grad2;
//  char* window_name = "Sobel-dibujo";
//  char* window_name2= "Sobel-foto";
//  char* window_name3 = "Foto";
//  char* window_name4= "Dibujo";
//  char* window_name5= "Sobel1";
//  char* window_name6= "Sobel2";
//
//
//  src = imread( "1.jpg", CV_LOAD_IMAGE_COLOR );
//  sketch = imread( "2.jpg", CV_LOAD_IMAGE_COLOR );
//  imshow(window_name3,src);
//  imshow(window_name4,sketch);
//
//  if( !src.data || !sketch.data )
//  { return -1; }
//
//
//  grad= __Sobel(sketch);
//  imshow(window_name5,grad);
//  adaptiveThreshold(grad,dest_sketch,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,5,10);
//  imshow(window_name,dest_sketch);
//  grad2= __Sobel(src);
//  imshow(window_name6,grad2);
//  adaptiveThreshold(grad2,dest_foto,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,3,3.5);
//  imshow(window_name2,dest_foto);
//  waitKey(0);
//
//  return 0;
//  }
//#include "opencv2/core/core.hpp"
//#include "opencv2/contrib/contrib.hpp"
//#include "opencv2/highgui/highgui.hpp"
//
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <string>
//using namespace cv;
//using namespace std;
//
//static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
//    std::ifstream file(filename.c_str(), ifstream::in);
//    if (!file) {
//        string error_message = "No valid input file was given, please check the given filename.";
//        CV_Error(CV_StsBadArg, error_message);
//    }
//    string line, path, classlabel;
//    while (getline(file, line)) {
//        stringstream liness(line);
//        getline(liness, path, separator);
//        getline(liness, classlabel);
//        if(!path.empty() && !classlabel.empty()) {
//            images.push_back(imread(path, 0));
//            labels.push_back(atoi(classlabel.c_str()));
//        }
//    }
//}
//
int main(int argc, char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.

    if (argc != 2) {
        cout << "usage: " << argv[0] << " <csv.ext>" << endl;
        exit(1);
    }
    // Get the path to your CSV.

    string fn_csv = string(argv[1]);
//    string fn_csv = "1.jpg2.jpg";
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = images[0].rows;
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::FaceRecognizer on) and the test data we test
    // the model with, do not overlap.
//    Mat testSample = images[images.size() - 1];
//    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();
    string a;
    cout<<"Ingrese consulta:"<<endl;
    cin>>a;
    string Query =a+".jpg";
    int testLabel = 100;
//    cout<<Query;
    string Dir="imagenes/"+Query;
//    cout<<Dir;
    Mat testSample = imread(Dir, 0);
    // The following lines create an LBPH model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    //
    // The LBPHFaceRecognizer uses Extended Local Binary Patterns
    // (it's probably configurable with other operators at a later
    // point), and has the following default values
    //
    //      radius = 1
    //      neighbors = 8
    //      grid_x = 8
    //      grid_y = 8
    //
    // So if you want a LBPH FaceRecognizer using a radius of
    // 2 and 16 neighbors, call the factory method with:
    //
    //      cv::createLBPHFaceRecognizer(2, 16);
    //
    // And if you want a threshold (e.g. 123.0) call it with its default values:
    //
    //      cv::createLBPHFaceRecognizer(1,8,8,8,123.0)
    //

    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer(2,16);
    model->train(images, labels);
    // The following line predicts the label of a given
    // test image:
    int predictedLabel = model->predict(testSample);
    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
    Mat Resultado;
    Mat Sketch;
//    cout<<to_string(predictedLabel)+".jpg"<<endl;
    string nameImage =to_string(predictedLabel)+".jpg";
    string nameSketch =Dir;

    Resultado = imread(nameImage, CV_LOAD_IMAGE_COLOR);
    namedWindow("Resultado", CV_WINDOW_AUTOSIZE);
    imshow("Resultado", Resultado);
    Sketch = imread(nameSketch, CV_LOAD_IMAGE_COLOR);
    namedWindow("Sketch", CV_WINDOW_AUTOSIZE);
    imshow("Sketch", Sketch);

    string result_message = format("Respuesta = %d / Consulta = %d.", predictedLabel, testLabel);
    cout << result_message << endl;
    // Sometimes you'll need to get/set internal model data,
    // which isn't exposed by the public cv::FaceRecognizer.
    // Since each cv::FaceRecognizer is derived from a
    // cv::Algorithm, you can query the data.
    //
    // First we'll use it to set the threshold of the FaceRecognizer
    // to 0.0 without retraining the model. This can be useful if
    // you are evaluating the model:
    //
    model->set("threshold", 0.0);
    // Now the threshold of this model is set to 0.0. A prediction
    // now returns -1, as it's impossible to have a distance below
    // it
//    predictedLabel = model->predict(testSample);
//    cout << "Predicted class = " << predictedLabel << endl;
    // Show some informations about the model, as there's no cool
    // Model data to display as in Eigenfaces/Fisherfaces.
    // Due to efficiency reasons the LBP images are not stored
    // within the model:
    cout << "Model Information:" << endl;
    string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
            model->getInt("radius"),
            model->getInt("neighbors"),
            model->getInt("grid_x"),
            model->getInt("grid_y"),
            model->getDouble("threshold"));
    cout << model_info << endl; model
    // We could get the histograms for example:
    vector<Mat> histograms = model->getMatVector("histograms");
    // But should I really visualize it? Probably the length is interesting:
    cout << "Size of the histograms: " << histograms[0].total() << endl;
    waitKey(0);
    return 0;
}

//#include "opencv2/core/core.hpp"
//#include "opencv2/contrib/contrib.hpp"
//#include "opencv2/highgui/highgui.hpp"
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "lbp.h"
#include "histogram.h"

using namespace cv;

int main(int argc, const char *argv[]) {
	int deviceId = 0;
	if(argc > 1)
		deviceId = atoi(argv[1]);

	VideoCapture cap(0);

	if(!cap.isOpened()) {
    	cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
    	return -1;
    }

	// initial values
    int radius = 1;
    int neighbors = 8;

    // windows
    namedWindow("original",CV_WINDOW_AUTOSIZE);
    namedWindow("lbp",CV_WINDOW_AUTOSIZE);
//    namedWindow("hist",CV_WINDOW_AUTOSIZE);

    // matrices used
    Mat frame; // always references the last frame
    Mat dst; // image after preprocessing
    Mat lbp; // lbp image

    // just to switch between possible lbp operators
    vector<string> lbp_names;
    lbp_names.push_back("LBP Extendido"); // 0
    lbp_names.push_back("LBP Original"); // 1
    lbp_names.push_back("LBP-Sobel"); // 2
    int lbp_operator=0;

    bool running=true;
    int a=0;
    while(running) {
        if(a==0){
//            system("PAUSE");
            waitKey(2000);
            a=1;
    }
    	cap >> frame;

//std::cout << frame.channels()<<std::endl;std::cout << dst.channels()<<std::endl;
    	cvtColor(frame, dst, CV_BGR2GRAY);
//        dst = frame.clone();
    	GaussianBlur(dst, dst, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
    	// comment the following lines for original size
if (frame.cols == 0) {
     cout << "Error reading file "  << endl;
     return -1;
}
if (dst.cols == 0) {
     cout << "Error reading file "  << endl;
     return -1;
}
    	resize(frame, frame, Size(), 0.5, 0.5);
    	resize(dst,dst,Size(), 0.5, 0.5);
    	//
    	switch(lbp_operator) {
    	case 0:
    		lbp::ELBP(dst, lbp, radius, neighbors); // use the extended operator
    		break;
    	case 1:
    		lbp::OLBP(dst, lbp); // use the original operator
    		break;
    	case 2:
    		lbp::VARLBP(dst, lbp, radius, neighbors);
    		break;
    	}
    	// now to show the patterns a normalization is necessary
    	// a simple min-max norm will do the job...
    	normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);
    	imshow("original", frame);
    	imshow("lbp", lbp);
//    	Mat Hist;
//        lbp::spatial_histogram(dst,Hist,255,Size(30,30),10);
//        cout<<Hist;
//        lbp::histogram_(lbp,Hist,255); const Size
//    	imshow("hist", Hist);
//            system("PAUSE");

    	char key = (char) waitKey(20);

    	// exit on escape
    	if(key == 27)
    		running=false;

    	// to make it a bit interactive, you can increase and decrease the parameters
    	switch(key) {
    	case 'q': case 'Q':
    		running=false;
    		break;
    	// lower case r decreases the radius (min 1)
    	case 'r':
    		radius-=1;
    		radius = std::max(radius,1);
    		cout << "radio=" << radius << endl;
    		break;
    	// upper case r increases the radius (there's no real upper bound)
    	case 'R':
    		radius+=1;
    		radius = std::min(radius,32);
    		cout << "radio=" << radius << endl;
    		break;
    	// lower case p decreases the number of sampling points (min 1)
    	case 'p':
    		neighbors-=1;
    		neighbors = std::max(neighbors,1);
    		cout << "Vecinos=" << neighbors << endl;
    		break;
    	// upper case p increases the number of sampling points (max 31)
    	case 'P':
    		neighbors+=1;
    		neighbors = std::min(neighbors,31);
    		cout << "Vecinos=" << neighbors << endl;
    		break;
    	// switch between operators
    	case 'o': case 'O':
    		lbp_operator = (lbp_operator + 1) % 3;
    		cout << "Cambiar operador " << lbp_names[lbp_operator] << endl;
    		break;
    	case 's': case 'S':
    		imwrite("original.jpg", frame);
    		imwrite("lbp.jpg", lbp);
    		cout << "Screenshot (operator=" << lbp_names[lbp_operator] << ",radio=" << radius <<",radio=" << neighbors << ")" << endl;
    		break;
    	default:
    		break;
    	}

    }
    	return 0; // success
}
