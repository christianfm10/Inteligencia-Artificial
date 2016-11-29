
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "lbp.h"
#include "histogram.h"
using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "Input no valido.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "usar: " << argv[0] << " <csv.ext>" << endl;
        exit(1);
    }
   string fn_csv = string("names.csv");
    string fn_csv2 = string("namesc.csv");

   vector<Mat> images;
    vector<int> labels;
    vector<Mat> images1;
    vector<int> labels1;

    try {
        read_csv(fn_csv, images, labels);
        read_csv(fn_csv2, images1, labels1);
    } catch (cv::Exception& e) {
        cerr << "Error abriendo archivo \"" << fn_csv << "\". Razon: " << e.msg << endl;
        exit(1);
    }
    if(images.size() <= 1) {
        string error_message = "Se necesita al menos 2 imagenes";
        CV_Error(CV_StsError, error_message);
    }
    if(images1.size() <= 1) {
        string error_message = "Se necesita al menos 2 imagenes";
        CV_Error(CV_StsError, error_message);
    }
     int height = images[0].rows;

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

    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer(2,16);
    Ptr<FaceRecognizer> model1 = createLBPHFaceRecognizer(2,16);
    model->train(images, labels);
    model1->train(images1, labels1);

    int predictedLabel = model->predict(testSample);

    Mat Resultado;
    Mat Sketch;
    Mat dst;
    Mat lbp;
    string nameImage =to_string(predictedLabel)+".jpg";
    string nameSketch =Dir;

    Resultado = imread(nameImage, CV_LOAD_IMAGE_COLOR);
    Sketch = imread(nameSketch, CV_LOAD_IMAGE_COLOR);
    namedWindow("Sketch", CV_WINDOW_AUTOSIZE);
    imshow("Sketch", Sketch);
    cvtColor(Resultado, dst, CV_BGR2GRAY);
    GaussianBlur(dst, dst, Size(7,7), 5, 3, BORDER_CONSTANT);
    resize(dst,dst,Size(), 0.5, 0.5);
//PATRON BINARIO LOCAL
    lbp::ELBP(dst, lbp, 2, 16);
   	normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);


    model->set("threshold", 0.0);

    cout << "Informacion:" << endl;
    string model_info = format("\tLBPH(radio=%i, vecinos=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
            model->getInt("radius"),
            model->getInt("neighbors"),
            model->getInt("grid_x"),
            model->getInt("grid_y"),
            model->getDouble("threshold"));
    cout << model_info << endl;
    vector<Mat> histograms = model->getMatVector("histograms");
    vector<Mat> histograms1 = model1->getMatVector("histograms");

    	double result = 0.0;

    int query=std::stoi( a );
    double resultadoOP=1000;
    int labelOP=1000;
//chi cuadrado
       for(int j=0; j < histograms.size();j++){
            result=0.0;
            for(int i=0; i < histograms[0].cols; i++) {
                double a = histograms[j].at<float>(0,i) - histograms1[query-1].at<float>(0,i);
                double b = histograms[j].at<float>(0,i) + histograms1[query-1].at<float>(0,i);
                if(abs(b) > numeric_limits<double>::epsilon()) {
                    result+=(a*a)/b;
                }
            }
            if(resultadoOP>result){
                resultadoOP=result;
                labelOP=j;
            }
            cout<<"Resultado "<<j+1<<" :"<<result<<endl;
       }
       cout<<labelOP+1<<endl;
    namedWindow("ResultadoOP", CV_WINDOW_AUTOSIZE);
    string nameOP;
    nameOP=to_string(labelOP+1)+".jpg";
    string ok=nameOP;
    cout<<nameOP<<endl;
    Mat RespOP;
    RespOP = imread(ok, CV_LOAD_IMAGE_COLOR);
    imshow("ResultadoOP", RespOP);
//Cantidad de aciertos



//int aciertos=0;
//    for(int k=0;k<histograms.size();k++){
//        double resultadoOP=1000;
//        int labelOP=1000;
//        for(int j=0; j < histograms.size();j++){
//                result=0.0;
//                for(int i=0; i < histograms[0].cols; i++) {
//                    double a = histograms[j].at<float>(0,i) - histograms1[k].at<float>(0,i);
//                    double b = histograms[j].at<float>(0,i) + histograms1[k].at<float>(0,i);
//                    if(abs(b) > numeric_limits<double>::epsilon()) {
//                        result+=(a*a)/b;
//                    }
//                }
//                if(resultadoOP>result){
//                    resultadoOP=result;
//                    labelOP=j;
//                }
////                cout<<"Resultado "<<j+1<<" :"<<result<<endl;
//           }
//            cout<<"Acerto :"<<k+1<<"?"<<endl;
//           if(labelOP==k){
//                cout<<"si"<<endl;
//                aciertos++;
//           }
//            else{
//                cout<<"NO"<<endl;
//            }
//    }
    string result_message = format("Respuesta = %d / Consulta = %d.", labelOP+1, testLabel);
    cout << result_message << endl;
    waitKey(0);
    return 0;
}
