---
title: Median and Edge Filtering
date: 2023-04-08 00:00:00 +09:00
categories: [Digital Image Processing]
tags:
  [
    Computer Vision,
    Digital Image Processing,
    Median Filtering,
    Edge Filtering
  ]
pin: true
---

# 1. salt_pepper2.png에 대해서 3x3, 5x5의 Mean 필터를 적용해보고 결과를 분석할 것
(잘 나오지 않았다면 그 이유와 함께 결과를 개선해볼 것)

## 1-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1); //결과를 저장하기 위한 선언

	int wd = src_img.cols; int hg = src_img.rows; //source img의 가로, 세로 길이
	int kwd = kn_size.width; int khg = kn_size.height; // kernel의 가로, 세로 길이
	int rad_w = kwd / 2; int rad_h = khg / 2; //가장자리 indexing을 피하기 위해 정의한 변수
	int size = kwd * khg; //kernel을 1차원 배열로 나타냈을 때의 길이
	uchar* src_data = (uchar*)src_img.data; //src_img.data는 Mat 객체 src_img의 첫 번째 픽셀 데이터의 포인터를 반환
	uchar* dst_data = (uchar*)dst_img.data; //void* 타입의 포인터로 반환되므로, 이를 uchar* 타입의 포인터로 
	uchar value = 0; //중간 값을 저장하기 위한 변수
	float* table = new float[size](); //커널 테이블 동적 할당

	//픽셀 인덱싱 (가장자리 제외)
	for (int c = rad_w; c < wd - rad_w; c++) {
		for (int r = rad_h; r < hg - rad_h; r++) {
			//커널 테이블에 주변 픽셀값 저장
			for (int i = 0; i < kwd; i++) {
				for (int j = 0; j < khg; j++) {
					table[i * khg + j] = src_data[(r + j - rad_h) * wd + c + i - rad_w];//1차원 이미지 data에 대한 kernel의 위치 계산
				}
			}
			//커널 테이블의 중간값 계산
			sort(table, table + size);// kernel의 index를 오름차순으로 정렬
			value = table[size / 2];// 중앙 값을 value에 저장
			dst_data[r * wd + c] = value;// dst_data에 해당 값을 저장
		}
	}
	delete[] table;
}
void doMedianEx() {
	cout << "--- doMedianEX() --- " << endl;
	//입력
	Mat src_img = imread("D:/study/디영처/6주차/salt_pepper2.png", 0);
	if (!src_img.data) printf("No image data \n");
	//Median 필터링 수행
	Mat dst_img1;
	Mat dst_img2;
	myMedian(src_img, dst_img1, Size(3, 3));// 3x3 filter 수행
	myMedian(src_img, dst_img2, Size(5, 5));// 5x5 filter 수행
	//출력
	Mat result_img1;
	Mat result_img2;
	hconcat(src_img, dst_img1, result_img1);
	imshow("doMedianEx()_3x3kernel", result_img1); //3x3 filter 결과
	hconcat(src_img, dst_img2, result_img2);
	imshow("doMedianEx()_5x5kernel", result_img2); //5x5 filter 결과
	waitKey(0);
}

void main() {
	doMedianEx();
}
```

## 1-2) 실험결과

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/9d8157cb-1d23-424e-919a-e79d9bb1d138)

## 1-3)구현 과정

- **`void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size) :`** 이 함수에서 필터링 사이즈를 입력 받고, 중앙값을 찾고 필터링 합니다. sort함수를 사용하여 만약 kernel이 3x3이라면 9개의 값들을 오름차순으로 놓고 그 가운데 값을 선정하여 filtering을 하였습니다.
- **`void doMedianEx()`**: 이 함수에서 실제로 filtering을 진행합니다. filter는 3x3과 5x5 2가지로 구현하여 그 결과를 확인하였습니다.

## 1-4)결과 분석

3x3과 5x5 Median Filter를 적용하였을 때의 결과를 각각 확인하였습니다. 비교해보았을 때, 3x3 filter에 비해 5x5의 filter의 결과가 잘나왔습니다. 3x3 filter는 9개, 즉 filtering을 하는 pixel을 포함하여 9개의 값 중 중앙 값을 선택하기 때문에, 너무 낮거나 높은 pixel의 값이 선택될 가능성이 낮습니다. 하지만, 5x5 filter는 25개의 값 중 중앙 값을 선택하기 때문에 3x3 filter에 비해 훨씬 가능성이 낮습니다. 따라서 3x3 filter의 결과를 본다면 원본 사진에 비해 많이 filtering되었지만 아직은 높고 낮은 noise들이 남아 있는 것을 볼 수 있었습니다. 하지만 5x5에서는 noise가 없습니다.

따라서 3x3 filter의 성능을 향상시키는 방법은 kernel의 크기를 늘리는 것이라는 것을 알 수 있었습니다.

# 2. rock.png에 대해서 Bilateral 필터를 적용해볼 것 (아래 table을 참고하여 기존
gaussian 필터와의 차이를 분석해볼 것)

## 2-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

double gaussian(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));//가우시안 식을 계산
}

float distance(int x, int y, int i, int j) {
	return float(sqrt(pow(x-i, 2) + pow(y-j, 2)));// 해당 pixel과의 거리를 계산하는 역할
}

void bilateral(const Mat& src_img, Mat& dst_img,
	int c, int r, int diameter, double sig_r, double sig_s) {
	int radius = diameter / 2; //diameter는 filter의 지름, 즉 radius는 filter의 반지름

	double gr, gs, wei;
	double tmp = 0.;//정규화 전의 값을 저장하는 변수
	double sum = 0.;//정규화에 이용되는 변수

	// <커널 인덱싱>
	for (int kc = -radius; kc <= radius; kc++) {
		for (int kr = -radius; kr <= radius; kr++) {
			double dis = distance(c, r, c + kc, r + kr); //kernel 내에서 filtering하는 픽셀과 주변 픽셀과의 거리를 구한다.
			//filtering하는 픽셀의 밝기와 유사할 수록 큰 값을 갖는다.
			gr = gaussian(src_img.at<uchar>(c + kc, r + kr) - src_img.at<uchar>(c, r), sig_r); 
			//filtering하는 픽셀과 거리가 가까울 수록 큰값을 갖는다.
			gs = gaussian(dis, sig_s);
			wei = gr * gs;//weight를 구한다.
			tmp += src_img.at<uchar>(c + kc, r + kr) * wei;//weight와 pixel의 밝기 값을 곱하여 가중치가 적용된 픽셀의 밝기값을 구한다.
			sum += wei;//정규화를 위해 weight를 모두 더한다.
		}
	}
	dst_img.at<double>(c, r) = tmp / sum; //정규화를 한다.
}

void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	Mat guide_img = Mat::zeros(src_img.size(), CV_64F);//filtering결과를 저장하는 변수
	int wh = src_img.cols; int hg = src_img.rows;//source이미지의 가로 세로
	int radius = diameter / 2; 

	//<픽셀 인덱싱(가장자리 제외)
	for (int c = radius + 1; c < hg - radius; c++) {// c와 r은 filtering하는 위치를 의미한다. 
		for (int r = radius + 1; r < wh - radius; r++) {//가장자리는 indexing하지 않기 위해서 radius+1부터 filtering
			bilateral(src_img, guide_img, c, r, diameter, sig_r, sig_s);//모든 pixel에 대해 bilateral filter를 적용한다.
		}
	}
	guide_img.convertTo(dst_img, CV_8UC1); // Mat type 변환
}

void doBilateralEx() {
	cout << "--- doBilateralEX() --- " << endl;

	Mat src_img = imread("D:/study/디영처/6주차/rock.png", 0);
	if (!src_img.data) printf("No image data \n");
	
	Mat dst_img1;
	Mat result_img1;
	myBilateral(src_img, dst_img1, 5, 25.0, 50.0);
	hconcat(src_img, dst_img1, result_img1);
	imshow("sig_r=25, sig_s=50(기준)", result_img1);

	Mat dst_img2;
	Mat result_img2;
	myBilateral(src_img, dst_img2, 5, 75.0, 50.0);
	imshow("sig_r=75, sig_s=50", dst_img2);

	Mat dst_img3;
	myBilateral(src_img, dst_img3, 5, 225.0, 50.0);
	imshow("sig_r=225, sig_s=50", dst_img3);

	Mat dst_img4;
	myBilateral(src_img, dst_img4, 5, 25.0, 125.0);
	imshow("sig_r=25, sig_s=125", dst_img4);

	Mat dst_img5;
	myBilateral(src_img, dst_img5, 5, 25.0, 999999);
	imshow("sig_r=25, sig_s=999999", dst_img5);

	waitKey(0);
}
void main() {
	doBilateralEx();
}
```

## 2-2) 실험 결과

- Left: 원본사진 / Right: sig_r=25, sig_s=50

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/d27cee8a-d201-49c1-89f1-3287e31bfb0a)

위 사진은 원본 사진과 실습에서 나오는 조건 sig_r=25, sig_s=50을 적용한 사진 입니다.

edge가 보전되면서 Smoothing이 된 것을 볼 수 있습니다.

---

- sig_s=50(고정), sig_r=25, 50, 999999 순

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/a2705622-aca2-4a54-9af8-84d97aa0c0da)

위 결과는 sig_s는 50으로 고정하고 sig_r값을 점점 늘렸을 때의 결과 사진입니다. 

---

- sig_r=25(고정), sig_s=50, 150, 450 순

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/803afd60-1a1f-4e4c-a395-ea9004a63b24)

위 결과는 sig_r은 25로 고정하고 sig_s값을 점점 늘렸을 때의 결과 사진입니다. 

## 2-3) 구현 과정

- **`double gaussian(float x, double sigma):`** 이 함수는 시그마 값에 따라 gaussian식을 직접 계산할 수 있는 함수입니다.
- `**float distance(int x, int y, int i, int j):**` 이 함수는 pixel로부터 다른 pixel까지의 거리를 구하는 함수입니다.
- **`void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s):`** 이 함수는 실질적으로 bilateral filter의 동작을 하는 함수입니다.

![Untitled 4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/372ad9f1-8eeb-486d-8813-2b4da4d3b094)

위 식을 함수로 구현한 코드로, filtering하는 픽셀과의 거리와 밝기 차이에 따라 가중치를 다르게 합니다.

- **`void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s):`** 위에서 정의된 bilateral함수를 통해 전체 이미지에 대해 연산을 진행하는 함수입니다.
- **`void doBilateralEx():`** sig_r과 sig_s값을 다르게 하여 실제로 결과를 확인하는 함수입니다.

## 2-4) 결과 분석

위에서 sig_r과 sig_s값을 변화시키면서 값을 확인했습니다. 우선 기본적인 sig_r=20, sig_s=50의 filtering결과를 보았을 때, 적절하게 edge가 유지되면서 smooting이 된 것을 확인할 수 있었습니다. 따라서 이번에는 sig_r과 sig_s값에 따른 변화를 보았습니다. sig_s를 고정하고 sig_r을 값을 늘렸을 때, 점점 더 smooth해지는 것을 볼 수 있었습니다. 이는 sig_r의 값이 늘어나면서, 밝기 값에 대한 기준이 줄어드는 것을 의미함으로 더 blurry해지는 것이라고 예상할 수 있었고 실제로 결과도 그렇게 나왔습니다. 반면 sig_r값을 작게 고정하고 sig_s를 늘렸을 때에는 생각보다 많은 변화는 없었습니다. 하지만 이론적으로는 더 blurry해지는 것이 맞고, 아마 그럴 것이라고 예상합니다. 하지만 sig_r값이 미치는 영향력이 더 크다는 것을 알 수 있었습니다.

gaussian filter 하나만을 적용했을 때와 비교해보자면, gaussian filter만을 적용했을 때에는 edge가 없어지는 것을 볼 수 있었는데, bilateral filter는 edge를 유지하면서 blurry해 지는 것을 볼 수 있었습니다.

# 3. OpenCV의 Canny edge detection 함수의 파라미터를 조절해 여러 결과를 도출하고 파라미터에 따라서 처리시간이 달라지는 이유를 정확히 서술할 것

## 3-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

void followEdges(int x, int y, Mat& magnitude, int tUpper, int tLower, Mat& edges) {
	edges.at<float>(y, x) = 255;

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			if ((i != 0) && (j != 0) && (x + i >= 0) && (y + j >= 0) &&
				(x + i <= magnitude.cols) && (y + j <= magnitude.rows)) {

				if ((magnitude.at<float>(y + j, x + i) > tLower) &&
					(edges.at<float>(y + j, x + i) != 255)) {
					followEdges(x + i, y + j, magnitude, tUpper, tLower, edges);
					//재귀적 방법으로 이웃 픽셀에서 불확실한 edg를 찾아 edge로 규정
				}
			}
		}
	}
}

void edgeDetect(Mat& magnitude, int tUpper, int tLower, Mat& edges) {
	int rows = magnitude.rows;
	int cols = magnitude.cols;

	edges = Mat(magnitude.size(), CV_32F, 0.0);

	//<픽셀 인덱싱>
	for (int x = 0; x < cols; x++) {
		for (int y = 0; y < rows; y++) {
			if (magnitude.at<float>(y, x) >= tUpper) {
				followEdges(x, y, magnitude, tUpper, tLower, edges);
				//edge가 확실하면 이와 연결된 불확실한 edge를 탐색
			}
		}
	}
}

void nonMaximumSuppression(Mat& magnitudeImage, Mat& directionImage) {
	Mat checkImage = Mat(magnitudeImage.rows, magnitudeImage.cols, CV_8U);

	MatIterator_<float>itMag = magnitudeImage.begin<float>();
	MatIterator_<float>itDirection = directionImage.begin<float>();
	MatIterator_<unsigned char>itRet = checkImage.begin<unsigned char>();
	MatIterator_<float>itEnd = magnitudeImage.end<float>();

	for (; itMag != itEnd; ++itDirection, ++itRet, ++itMag) {
		const Point pos = itRet.pos();

		float currentDirection = atan(*itDirection) * (180 / 3.142);

		while (currentDirection < 0)currentDirection += 180;

		*itDirection = currentDirection;

		if (currentDirection > 22.5 && currentDirection <= 67.5) {
			if (pos.y > 0 && pos.x > 0 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x - 1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows-1&&pos.x<magnitudeImage.cols-1&& *itMag<=magnitudeImage.at<float>(pos.y+1,pos.x+1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
		else if (currentDirection > 67.5 && currentDirection <= 112.5) {
			if (pos.y > 0 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows - 1 && *itMag <= magnitudeImage.at<float>(pos.y + 1, pos.x)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
		else if (currentDirection > 112.5 && currentDirection <= 157.5) {
			if (pos.y > 0 && pos.x<magnitudeImage.cols-1 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x+1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows - 1 && pos.x>0 && *itMag <= magnitudeImage.at<float>(pos.y + 1, pos.x-1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
		else {
			if (pos.x > 0 && *itMag <= magnitudeImage.at<float>(pos.y, pos.x - 1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.x < magnitudeImage.cols - 1 && *itMag <= magnitudeImage.at<float>(pos.y, pos.x + 1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
	}
}

void myCanny(const Mat& src_img, Mat& dst_img, int tUpper,int tLower) {

	//<Gaussian filter 기반 노이즈 제거>
	Mat blur_img;
	GaussianBlur(src_img, blur_img, Size(3, 3), 1.5);

	//<Sobel edge detection>
	Mat magX = Mat(src_img.rows, src_img.cols, CV_32F);
	Mat magY = Mat(src_img.rows, src_img.cols, CV_32F);
	Sobel(blur_img, magX, CV_32F, 1, 0, 3);
	Sobel(blur_img, magY, CV_32F, 0, 1, 3);

	Mat sum = Mat(src_img.rows, src_img.cols, CV_64F);
	Mat prodX = Mat(src_img.rows, src_img.cols, CV_64F);
	Mat prodY = Mat(src_img.rows, src_img.cols, CV_64F);
	multiply(magX, magX, prodX);
	multiply(magY, magY, prodY);
	sum = prodX + prodY;
	sqrt(sum, sum);

	Mat magnitude = sum.clone();

	//<Non-maximum suppression>
	Mat slopes = Mat(src_img.rows, src_img.cols, CV_32F);
	divide(magY, magX, slopes);
	//gradient의 방향 계싼
	nonMaximumSuppression(magnitude, slopes);

	//<Edge tracking by hysteresis>
	edgeDetect(magnitude, tUpper, tLower, dst_img);
	dst_img.convertTo(dst_img, CV_8UC1);
	
}

void doCannyEx() {
	cout << "--- doCannyEx() -- \n" << endl;
	clock_t start, end;

	//입력
	Mat src_img = imread("D:/study/디영처/6주차/edge_test.jpg", 0);
	if (!src_img.data) printf("No image data \n");

	Mat dst_img1;
	Mat result_img;
	start = clock();
	myCanny(src_img, dst_img1,200,100);
	end = clock();
	printf("threshold: 100-200: %ld ms\n", (end - start));
	hconcat(src_img, dst_img1, result_img);
	imshow("threshold: 100-200", result_img);

	Mat dst_img2;
	start = clock();
	myCanny(src_img, dst_img2, 180, 120);
	end = clock();
	printf("threshold: 155-255: %ld ms\n", (end - start));
	imshow("threshold: 155-255", dst_img2);

	Mat dst_img3;
	start = clock();
	myCanny(src_img, dst_img3, 300, 200);
	end = clock();
	printf("threshold: 200-300: %ld ms\n", (end - start));
	imshow("threshold: 200-300", dst_img3);

	Mat dst_img4;
	start = clock();
	myCanny(src_img, dst_img4, 400, 300);
	end = clock();
	printf("threshold: 300-400: %ld ms\n", (end - start));
	imshow("threshold: 300-400", dst_img4);

	Mat dst_img5;
	start = clock();
	myCanny(src_img, dst_img5, 300, 100);
	end = clock();
	printf("threshold: 100-300: %ld ms\n", (end - start));
	imshow("threshold: 100-300", dst_img5);

	Mat dst_img6;
	start = clock();
	myCanny(src_img, dst_img6, 400, 200);
	end = clock();
	printf("threshold: 200-400: %ld ms\n", (end - start));
	imshow("threshold: 200-400", dst_img6);

	waitKey(0);
	
}
void main() {
	doCannyEx();
}
```

우선 강의노트에 써있는대로 실험을 진행하기 위해 위처럼 코딩하였지만 예외처리가 계속 발생하여 opencv에 있는 canny edge detection함수를 사용하였습니다. 그 코딩은 아래와 같습니다.

```cpp
void doCannyEx() {
	cout << "--- doCannyEx() -- \n" << endl;
	clock_t start, end;

	//입력
	Mat src_img = imread("D:/study/디영처/6주차/edge_test.jpg", 0);
	if (!src_img.data) printf("No image data \n");

	Mat dst_img1;
	Mat result_img;
	start = clock();
	//myCanny(src_img, dst_img1,200,100);
	Canny(src_img, dst_img1, 100, 200);
	end = clock();
	printf("threshold: 100-200: %ld ms\n", (end - start));
	hconcat(src_img, dst_img1, result_img);
	imshow("threshold: 100-200", result_img);

	Mat dst_img2;
	start = clock();
	//myCanny(src_img, dst_img2, 100, 0);
	Canny(src_img, dst_img2, 0, 100);
	end = clock();
	printf("threshold: 0-100: %ld ms\n", (end - start));
	imshow("threshold: 0-100", dst_img2);

	Mat dst_img3;
	start = clock();
	//myCanny(src_img, dst_img3, 300, 200);
	Canny(src_img, dst_img3, 200, 300);
	end = clock();
	printf("threshold: 200-300: %ld ms\n", (end - start));
	imshow("threshold: 200-300", dst_img3);

	Mat dst_img4;
	start = clock();
	//myCanny(src_img, dst_img4, 150, 100);
	Canny(src_img, dst_img4, 150, 200);
	end = clock();
	printf("threshold: 150-200: %ld ms\n", (end - start));
	imshow("threshold: 150-200", dst_img4);

	Mat dst_img5;
	start = clock();
	//myCanny(src_img, dst_img5, 50, 0);
	Canny(src_img, dst_img5, 50, 100);
	end = clock();
	printf("threshold: 50-100: %ld ms\n", (end - start));
	imshow("threshold: 50-100", dst_img5);

	Mat dst_img6;
	start = clock();
	//myCanny(src_img, dst_img6, 250, 200);
	Canny(src_img, dst_img6, 250, 300);
	end = clock();
	printf("threshold: 250-300: %ld ms\n", (end - start));
	imshow("threshold: 250-300", dst_img6);
	
	waitKey(0);
	
}
void main() {
	doCannyEx();
}
```

## 3-2) 실험 결과

![Untitled 5](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/196d2d64-ce9c-4125-9228-1a2db3066ac3)

![Untitled 6](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/318e7e30-ea1a-4041-ae6b-0f08468c70ba)

![Untitled 7](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8dfb0951-b721-48c3-b988-75a3a4109cb0)

![Untitled 8](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8e5f5ef6-5148-4bea-884b-6d68cf7b9bac)

## 3-3) 구현 과정

- **`void followEdges(int x, int y, Mat& magnitude, int tUpper, int tLower, Mat& edges):`** 이 함수는 재귀적인 방법으로 non-maximum suppression을 통해 불확실한 edge로 분명하게 만드는 함수입니다.
- **`void edgeDetect(Mat& magnitude, int tUpper, int tLower, Mat& edges):`** 이 함수는  followEdges 함수를 호출하여 입력 이미지의 gradient magnitude를 받고, 입력된 threshold값에 따라서 edge를 추출하고, 추출된 edge를 연결합니다.
- `**void nonMaximumSuppression(Mat& magnitudeImage, Mat& directionImage):**`  이 함수는 edgeDetect함수의 결과로 나온 edge 픽셀들을 분명한 edge로 하는 역할을 합니다. 픽셀들의 기울기를 계산하고, 기울기 방향에 따라 인접한 픽셀들과 비교하여 최대값인 경우에만 edge를 유지하고, 그 외에는 억제합니다.
- **`void myCanny(const Mat& src_img, Mat& dst_img, int tUpper,int tLower):`** 분명한 edge를 유지하는 최종적인 역할을 하는 함수 입니다.

따라서 high threshold 이상의 분명한 edge를 검출하고, low threshold와 high threshold 사이의 불분명한 edge에서 분명한 edge를 구별해내는 코드입니다.

- 하지만 결국 결과가 나오지 않아 OpenCV의 Canny함수를 사용하여 직접 실험하였습니다.

## 3-4) 결과 분석

위 결과는 각 threshold마다의 실행 결과를 나타낸 값입니다. 직접 실행을 해보기 전 결과를 예상해보았을 때, 범위가 넓을수록, 그리고 high threshold가 낮을수록 많은 연산이 있기 때문에 시간이 오래걸릴 것이라고 예상했습니다. 

![Untitled 9](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/882ba421-2c19-479f-a0f6-ec8d198f5e7d)

실제로 결과를 살펴보겠습니다.

- 0-100: 39ms
- 100-200: 29ms
- 200-300:13ms

위를 보았을 때, 범위를 그대로 유지하고, high threshold의 값이 늘어날 수록 처리시간이 줄어드는 것을 볼 수 있습니다. 이는 제가 예상한 그대로의 결과이고, high threshold의 크기가 늘어날 수록 분명한 edge로 검출되는 양이 적기 때문에 당연한 결과입니다.

- 0-100: 39ms → 50-100: 19ms
- 100-200: 29ms →150-200: 18ms
- 200-300:13ms → 250-300: 10ms

위 결과를 보았을 때, high threshold를 유지하고, 범위를 줄였는데, 실험 전 예상 했듯이 처리시간이 줄어들었습니다. 이는 범위가 줄어들었으므로 분명하지 않은 edge를 분명한 edge로 바꾸는 연산이 줄어들었기 때문입니다.

따라서 파라미터, 즉 threshold에 따라 처리시간이 달라지는 이유는, 

1) High Threshold의 크기가 커지면, 분명한 edge의 수가 적어지기 때문

2) Threshold의 범위, 즉 불분명한 edge가 검출되는 범위가 줄어들기 때문

입니다.

위의 결과들이 모두 예상한대로 나왔고, 정상적으로 동작하는 것을 확인할 수 있었습니다.