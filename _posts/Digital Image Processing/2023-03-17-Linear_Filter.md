---
title: Linear Filter(Gaussian, Sobel)
date: 2023-03-17 00:00:00 +09:00
categories: [Digital Image Processing]
tags:
  [
    Computer Vision,
    Digital Image Processing,
    Gaussian Filter,
    Sobel Filter
  ]
pin: true
---

# ▪ 9x9 Gaussian filter를 구현하고 결과를 확인할 것
▪ 9x9 Gaussian filter를 적용했을 때 히스토그램이 어떻게 변하는지 확인할 것
▪ 영상에 Salt and pepper noise를 주고, 구현한 9x9 Gaussian filter를 적용해볼 것

## [Code]

```cpp
#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

int myKernelConv9x9(uchar* arr, int kernel[][9], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	//특정 화소의 모든 이웃화소에 대해 계산하도록 반복문 구성
	for (int j = -4; j <= 4; j++) {
		for (int i = -4; i <= 4; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				//영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 4][j + 4];
				sumKernel += kernel[i + 4][j + 4];
			}
		}
	}
	if (sumKernel != 0) {// 합이 1로 정규화되도록 해 영상의 밝기변화 방지
		return sum / sumKernel;
	}
	else {
		return sum;
	}
}

Mat myGaussianFilter(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;
	int kernel[9][9] = { 1, 2, 2, 2, 2, 2, 2, 2, 1,
						2, 3, 4, 4, 4, 4, 4, 3, 2,
						2, 4, 5, 6, 6, 6, 5, 4, 2,
						2, 4, 6, 7, 8, 7, 6, 4, 2,
						2, 4, 6, 8, 9, 8, 6, 4, 2,
						2, 4, 6, 7, 8, 7, 6, 4, 2,
						2, 4, 5, 6, 6, 6, 5, 4, 2,
						2, 3, 4, 4, 4, 4, 4, 3, 2,
						1, 2, 2, 2, 2, 2, 2, 2, 1};//9x9Gaussian 마스크
	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = myKernelConv9x9(srcData, kernel, x, y, width, height);
			//앞서 구현한 convolution에 마스크 배열을 입력해 사용
		}
	}

	return dstImg;
}

Mat GetHistogram(Mat src) {
	Mat histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;

	//히스토그램 계산
	calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

	//히스토그램 plot
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat()); //정규화

	for (int i = 1; i < number_bins; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);// 값과 값을 잇는 선을 그리는 방식으로 plot
	}

	return histImage;
}

void SpreadSalts(Mat img, int num) {
	//num : 점을 찍을 개수
	for (int n = 0; n < num; n++) {
		int x = rand() % img.cols;
		int y = rand() % img.rows;

		if (x % 2 == 0) {//랜덤한 변수 x가 홀수냐 짝수냐에 따라 검은색점 또는 하얀색 점을 찍어 salt and pepper noise를 만든다.
			img.at<uchar>(y, x) = 255;
		}
		else {
			img.at<uchar>(y, x) = 0;
		}
	}
}

int main() {
	Mat original_img = imread("D:/study/디영처/3주차/gear.jpg", 0); //흑백으로 사진을 불러온다.
	Mat noise_img = imread("D:/study/디영처/3주차/gear.jpg", 0); //흑백으로 사진을 불러온다.
	imshow("original image", original_img);
	Mat original_his = GetHistogram(original_img);
	imshow("original histogram", original_his);

	Mat org_Gaussian_img = myGaussianFilter(original_img);
	imshow("9x9 gaussian fiter image(original)", org_Gaussian_img);
	Mat org_Gaussian_his = GetHistogram(org_Gaussian_img);
	imshow("9x9 gaussian fiter histogram(original)", org_Gaussian_his);

	SpreadSalts(noise_img, 500);
	imshow("salt and pepper noise image", noise_img);

	Mat noise_Gaussian_img = myGaussianFilter(noise_img);
	imshow("9x9 gaussian fiter image(noise)",noise_Gaussian_img);
	
	waitKey(0);// 키 입력 대기(0: 키가 입력될 때 까지 프로그램 멈춤)
	destroyAllWindows();// 이미지 출력창 종료
	return 0;
}
```

## [구현 방법 및 결과]

## 1-1) 9x9 Gaussian filter를 구현하고 결과를 확인할 것

강의노트에 나와있는 Gaussian 필터는 3x3형태의 마스크 배열로 되어있습니다. 이번 실습에서는 9x9형태의 마스크 배열을 적용해야하므로 kernel의 크기도 kernel[9][9]로 변경하고, Convolution연산을 하는함수도 9x9 Gaussian filter연산을 수행하기 위해 바꿨습니다. 모든 화소에 대해 연산하기 위해 for문의 범위를 늘려 filter의 크기를 확장하였습니다. 그 결과는 아래와 같습니다.

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/35684442-b406-4397-b8ef-787925bf0ad2)

위는 왼쪽에 원본사진과 오른쪽에 9x9 Gaussian filter를 적용한 사진입니다. 확실이 사진이 흐릿해진 것을 볼 수 있고, 실습에서 3x3 filter를 사용했을 때에는 육안상으로 흐릿해진 것을 거의 확인할 수 없었지만, filter의 크기가 커져 확실하게 달라진 결과를 볼 수 있었습니다.

## 1-2) 9x9 Gaussian filter를 적용했을 때 히스토그램이 어떻게 변하는지 확인할 것

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4483f124-a4b0-44f0-b499-8bb7cdb8ab7d)

위 결과는 왼쪽이 원본사진의 히스토그램, 오른쪽에 9x9 Gaussian Filter를 적용한 사진의 히스토그램입니다. 히스토그램의 분포가 원본에 비해 부드러워진 것을 볼 수 있습니다. 이는 convolution연산을 통해 Blur효과가 나타났다는 것을 직관적으로 볼 수 있습니다.

## 1-3) 영상에 Salt and pepper noise를 주고, 구현한 9x9 Gaussian filter를 적용해볼 것

영상에 Salt and pepper noise를 주는 방법은 지난주 실습에서 SpreadSalts함수를 이용했던 방법과 동일하게 하였는데, if문을 활용하여 random하게 검은색 또는 흰색 점을 원하는 갯수만큼 noise로 줄 수 있도록 하였습니다. 그리고 그 사진에 9x9 Gaussian filter를 적용하였고 그 결과는 다음과 같습니다.

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/97217685-7a6e-4c5b-b508-6a7e19672826)

왼쪽은 노이즈가 추가된 image이고, 오른쪽은 이 이미지에 9x9 Gaussian filter를 적용한 결과입니다. noise는 총 500개를 추가하였고, filter를 적용한 결과를 통해 볼 수 있듯이 Blurry 효과를 통해 noise가 없어진 것을 볼 수 있습니다.

# ▪ 45도와 135도의 대각 에지를 검출하는 Sobel filter를 구현하고 결과를 확인할
것

## [Code]

```cpp
#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

int myKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	//특정 화소의 모든 이웃화소에 대해 계산하도록 반복문 구성
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				//영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}
	if (sumKernel != 0) {// 합이 1로 정규화되도록 해 영상의 밝기변화 방지
		return sum / sumKernel;
	}
	else {
		return sum;
	}
}

Mat mySobelFilter(Mat srcImg) {
	int kernel45[3][3] = { -2, -1, 0,
							-1, 0, 1,
							0, 1, 2};//45도방향 Sobel 마스크

	int kernel135[3][3] = { 0, 1, 2,
							-1, 0, 1,
							-2, -1, 0};//135도방향 Sobel 마스크
	//마스크 합이 0이 되므로 1로 정규화하는 과정은 필요 없음
	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = (abs(myKernelConv3x3(srcData, kernel45, x, y, width, height)) +
									  abs(myKernelConv3x3(srcData, kernel135, x, y, width, height))) / 2;
			// 두 에지 결과의 절대값 합 형태로 최종결과 도출
		}
	}
	return dstImg;
}

void main() {
	Mat src_img = imread("D:/study/디영처/3주차/gear.jpg", 0);
	imshow("original image", src_img);
	Mat sobel_img = mySobelFilter(src_img);
	imshow("sobel filter image", sobel_img);
	waitKey(0);
	destroyAllWindows();
}
```

## [구현 방법 및 결과]

Sobel마스크의 방향을 다르게 하는 것 외에는 실습시간에 했던 방향과 모두 같았습니다. 따라서 마스크의 방향을 45도와 135도에 대한 edge를 구하기 위해 코드를 수정하였고, 그 두 결과를 더하는 형태로 코딩하였습니다. 그 결과는 아래와 같습니다.

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ef5791cc-03b0-4435-9d59-c88a29758a50)

위 결과를 통해 sobel filter의 특성에 맞게 edge가 잘 검출된 것을 확인할 수 있었습니다.

# 컬러영상에 대한 Gaussian pyramid를 구축하고 결과를 확인할 것

## [Code]

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

Mat mySampling(Mat srcImg) {
	int width = srcImg.cols / 2;
	int height = srcImg.rows / 2;
	Mat dstImg(height, width, CV_8UC3);
	//가로 세로가 입력 영상의 절반인 3채널 영상을 먼저 생성
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[(y * width + x) * 3] = srcData[((y * 2)*(width * 2) + (x * 2)) * 3];
			dstData[(y * width + x)*3+1] = srcData[((y * 2)*(width * 2) + (x * 2)) * 3+1];
			dstData[(y * width + x)*3+2] = srcData[((y * 2)*(width * 2) + (x * 2)) * 3+2];
		}
	}
	return dstImg;
}

int myKernelConv9x9(uchar* arr, int kernel[][9], int x, int y, int width, int height, char color) {
	int sum = 0;
	int sumKernel = 0;

	//특정 화소의 모든 이웃화소에 대해 계산하도록 반복문 구성
	for (int j = -4; j <= 4; j++) {
		for (int i = -4; i <= 4; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {//영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
				//컬러영상에 대한 1차원 배열이므로 길이가 3배이다. 또한 순서가 B, G, R순으로 구성되기 때문에 각 color에 맞도록 conv하기 위해 if문을 사용하였다.
				if (color == 'B') {//B channel에 대한 연산
					sum += arr[((y + j) * width + (x + i))*3] * kernel[i + 4][j + 4];
					sumKernel += kernel[i + 4][j + 4];
				}
				else if (color == 'G') {//G channel에 대한 연산
					sum += arr[((y + j) * width + (x + i)) * 3+1] * kernel[i + 4][j + 4];
					sumKernel += kernel[i + 4][j + 4];
				}
				else if(color=='R') {//R channel에 대한 연산
					sum += arr[((y + j) * width + (x + i)) * 3+2] * kernel[i + 4][j + 4];
					sumKernel += kernel[i + 4][j + 4];
				}
			}
		}
	}
	if (sumKernel != 0) {// 합이 1로 정규화되도록 해 영상의 밝기변화 방지
		return sum / sumKernel;
	}
	else {
		return sum;
	}
}

Mat myGaussianFilter(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;
	int kernel[9][9] = { 1, 2, 2, 2, 2, 2, 2, 2, 1,
						 2, 3, 4, 4, 4, 4, 4, 3, 2,
					     2, 4, 5, 6, 6, 6, 5, 4, 2,
						 2, 4, 6, 7, 8, 7, 6, 4, 2,
						 2, 4, 6, 8, 9, 8, 6, 4, 2,
						 2, 4, 6, 7, 8, 7, 6, 4, 2,
						 2, 4, 5, 6, 6, 6, 5, 4, 2,
						 2, 3, 4, 4, 4, 4, 4, 3, 2,
						 1, 2, 2, 2, 2, 2, 2, 2, 1};// 9x9 Gaussian 마스크
	Mat dstImg(srcImg.size(), CV_8UC3);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[(y * width + x)*3] = myKernelConv9x9(srcData, kernel, x, y, width, height, 'B');
			dstData[(y * width + x)*3+1] = myKernelConv9x9(srcData, kernel, x, y, width, height, 'G');
			dstData[(y * width + x)*3+2] = myKernelConv9x9(srcData, kernel, x, y, width, height, 'R');
			//앞서 구현한 convolution에 마스크 배열을 입력해 사용
		}
	}

	return dstImg;
}

vector<Mat> myGaussianPyramid(Mat srcImg) {
	vector<Mat> Vec; //여러 영상을 모아서 저장하기 위해 STL의 vector 컨테이너 사용
	Vec.push_back(srcImg);
	for (int i = 0; i < 4; i++) {
		srcImg = mySampling(srcImg); //앞서 구현한 down sampling
		srcImg = myGaussianFilter(srcImg); //앞서 구현한 Gaussian filtering
		Vec.push_back(srcImg);//Vector 컨테이너에 하나씩 처리 결과를 삽입
	}
	return Vec;
}

void main() {
	Mat src_img = imread("D:/study/디영처/3주차/gear.jpg", 1); //컬러이미지로 불러온다.
	imshow("original image", src_img);

	vector<Mat> myGP = myGaussianPyramid(src_img);
	int itr = myGP.size();// vector의 크기만큼 for문을 돌리기 위함
	cout << "총" << itr << "개의 Gaussian Pyramid" << endl;
	Mat result;
	for (int i = 0; i < itr; i++) {//각 vector에 저장된 Gaussian Pyramid Feature를 출력한다.
		result = myGP[i];
		string fname = "Gaussian Pyramid" + to_string(i) + ".png";
		imshow(fname, result);
	}
	waitKey(0);
	destroyAllWindows();

}
```

## [구현 방법 및 결과]

Gaussian filter를 적용하며 Downsampling을 진행하는 코드입니다. 각 단계의 piramid, 즉 downsampling이 진행될 때마다 해상도가 절반이 됩니다. 여기서 처음에 결과를 잘못냈는데, 컬러 영상을 기준으로 Gaussian pyramid를 적용해야하는 것에 오류를 범했습니다. 컬러 영상을 1차원 배열로 나타낼 때, 이전까지의 과제(grayscale)과 다르게 B,G,R순으로 차례로 픽셀 정보가 있기 때문에 이를 고려하여 코딩하였습니다. myKernelConv9x9함수에서도 각 색깔에 대해 따로 처리하기 위해 char color라는 변수를 받아 각 색의 정보에 따라 따로 정보를 처리하였습니다. 이 결과를myGaussianPyramid 함수에서 sampling과 filter연산을 실행하여 vetor에 결과값을 넣습니다. 그 결과는 아래에서 모두 확인할 수 있습니다.

![Untitled 4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/d4bcdec0-bef0-4f2f-aa0e-b2b6b8d4e9ed)

각 결과들은 원본에서 해상도를 낮추고 filter를 적용한 image입니다.

# 컬러영상에 대한 Laplacian pyramid를 구축하고 결과를 확인할 것

## [Code]

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

Mat mySampling(Mat srcImg) {
	int width = srcImg.cols / 2;
	int height = srcImg.rows / 2;
	Mat dstImg(height, width, CV_8UC3);
	//가로 세로가 입력 영상의 절반인 3채널 영상을 먼저 생성
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[(y * width + x) * 3] = srcData[((y * 2) * (width * 2) + (x * 2)) * 3];
			dstData[(y * width + x) * 3 + 1] = srcData[((y * 2) * (width * 2) + (x * 2)) * 3 + 1];
			dstData[(y * width + x) * 3 + 2] = srcData[((y * 2) * (width * 2) + (x * 2)) * 3 + 2];
		}
	}
	return dstImg;
}

int myKernelConv9x9(uchar* arr, int kernel[][9], int x, int y, int width, int height, char color) {
	int sum = 0;
	int sumKernel = 0;

	//특정 화소의 모든 이웃화소에 대해 계산하도록 반복문 구성
	for (int j = -4; j <= 4; j++) {
		for (int i = -4; i <= 4; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {//영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
				//컬러영상에 대한 1차원 배열이므로 길이가 3배이다. 또한 순서가 B, G, R순으로 구성되기 때문에 각 color에 맞도록 conv하기 위해 if문을 사용하였다.
				if (color == 'B') {//B channel에 대한 연산
					sum += arr[((y + j) * width + (x + i)) * 3] * kernel[i + 4][j + 4];
					sumKernel += kernel[i + 4][j + 4];
				}
				else if (color == 'G') {//G channel에 대한 연산
					sum += arr[((y + j) * width + (x + i)) * 3 + 1] * kernel[i + 4][j + 4];
					sumKernel += kernel[i + 4][j + 4];
				}
				else if (color == 'R') {//R channel에 대한 연산
					sum += arr[((y + j) * width + (x + i)) * 3 + 2] * kernel[i + 4][j + 4];
					sumKernel += kernel[i + 4][j + 4];
				}
			}
		}
	}
	if (sumKernel != 0) {// 합이 1로 정규화되도록 해 영상의 밝기변화 방지
		return sum / sumKernel;
	}
	else {
		return sum;
	}
}

Mat myGaussianFilter(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;
	int kernel[9][9] = { 1, 2, 2, 2, 2, 2, 2, 2, 1,
						 2, 3, 4, 4, 4, 4, 4, 3, 2,
						 2, 4, 5, 6, 6, 6, 5, 4, 2,
						 2, 4, 6, 7, 8, 7, 6, 4, 2,
						 2, 4, 6, 8, 9, 8, 6, 4, 2,
						 2, 4, 6, 7, 8, 7, 6, 4, 2,
						 2, 4, 5, 6, 6, 6, 5, 4, 2,
						 2, 3, 4, 4, 4, 4, 4, 3, 2,
						 1, 2, 2, 2, 2, 2, 2, 2, 1 };// 9x9 Gaussian 마스크
	Mat dstImg(srcImg.size(), CV_8UC3);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[(y * width + x) * 3] = myKernelConv9x9(srcData, kernel, x, y, width, height, 'B');
			dstData[(y * width + x) * 3 + 1] = myKernelConv9x9(srcData, kernel, x, y, width, height, 'G');
			dstData[(y * width + x) * 3 + 2] = myKernelConv9x9(srcData, kernel, x, y, width, height, 'R');
			//앞서 구현한 convolution에 마스크 배열을 입력해 사용
		}
	}

	return dstImg;
}

vector<Mat> myLaplacianPyramid(Mat srcImg) {
	vector<Mat> Vec;

	for (int i = 0; i < 5; i++) {
		if (i != 4) {
			Mat highImg = srcImg; // 수행하기 이전 영상을 백업
			srcImg = mySampling(srcImg);
			srcImg = myGaussianFilter(srcImg);
			Mat lowImg = srcImg;
			resize(lowImg, lowImg, highImg.size());
			//작아진 영상을 백업한 영상의 크기로 확대
			Vec.push_back(highImg - lowImg + 128);
			//차 영상을 컨테이너에 삽입
			//128을 더해준 것은 차 영상에서 오버플로우를 방지하기 위함
		}
		else {
			Vec.push_back(srcImg);
		}
	}
	return Vec;
}

void main() {
	Mat src_img = imread("D:/study/디영처/3주차/gear.jpg", 1);
	Mat dst_img;

	vector<Mat> VecLap = myLaplacianPyramid(src_img);
	// Laplacian pyramid 확보
	reverse(VecLap.begin(), VecLap.end());
	//작은 영상부터 처리하기 위해 vector의 순서를 반대로

	for (int i = 0; i < VecLap.size(); i++) {
		//Vector의 크기만큼 반복
		if (i == 0) {
			dst_img = VecLap[i];
		}
		else {
			resize(dst_img, dst_img, VecLap[i].size());
			//작은 영상을 확대
			dst_img = dst_img + VecLap[i] - 128;
			//차 영상을 다시 더해 큰 영상을 복원
			//오버플로우 방지용으로 더했던 128을 다시 빼줌
		}
		string fname = "Laplacian Pyramid" + to_string(i) + ".png";
		imshow(fname, dst_img);
	}
	waitKey(0);
	destroyAllWindows();
}
```

## [구현 방법 및 결과]

Laplacian pyramid를 활용하여 영상을 복원하는 실습을 하였습니다. Laplacian pyramid에는 낮은 해상도의 확장한 영상과 높은 해상도의 영상 간의 차 영상을 저장하는 방식인데, 이때 오버플로우 방지를 위해 128를 더하고 저장하고 이후에 다시 빼는 작업을 해주었습니다. Laplacian pyramid를 vector로 선언하고 각 차 영상을 저장하였습니다. 이후 차 영상을 사용하여 영상을 복원하였고, 그 결과는 아래와 같습니다.

![Untitled 5](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/032a75df-d2bb-4b7b-b3e5-cee62d75f8c0)

위는 복원을 하는 과정을 보여주고, 최종적으로 복원된 영상이 오른쪽 사진입니다. 

![gear](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/54ca8069-41d7-483c-af57-11b94cd97c33)

위는 원래의 원본 영상인데, 복원된 사진이 원본과 거의 유사한 것을 볼 수 있었습니다.