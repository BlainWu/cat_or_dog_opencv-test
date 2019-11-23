#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main(int argc,char **argv) {
    String model="../data/model_iter_50000.caffemodel"; //caffe模型文件
    String net_txt="../data/deploy.prototxt";//网络结构文件
    String test_pic="../test_pics/3.jpg";//测试图片
    Net cat_dog=readNet(net_txt,model);

    Mat pic_raw=imread(test_pic),pic;
    pic=pic_raw;
    resize(pic_raw,pic,Size(256,256));//将图片大小归一

    Mat blob = blobFromImage(pic,1.0,Size(256,256));//将图片转换成数据流
    cat_dog.setInput(blob,"data");//给模型输入
    Mat prob=cat_dog.forward("prob");//得到模型输出
    Mat prob_show =prob.reshape(1,1);
    cout<<prob_show<<endl;
    putText(pic_raw, format("It is a :%s", (prob_show.at<float>(0, 0) > prob_show.at<float>(0, 1) ? "Cat" : "Dog")),Point(2,20),
            4, 0.8, Scalar(0, 0, 255), 2);

    imshow("result",pic_raw);
    waitKey(0);
    return 0;
}