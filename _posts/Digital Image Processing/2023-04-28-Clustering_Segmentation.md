---
title: Clustering and Segmentation
date: 2023-04-28 00:00:00 +09:00
categories: [Digital Image Processing]
tags:
  [
    Computer Vision,
    Digital Image Processing,
    Mean-Shift,
    Segmentation
  ]
pin: true
---

# 1. Mean-shift OpenCV와 low-level 구현 2개 적용해보고 결과 분석

# Mean-shift OpenCV 구현

## 1-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

void exCvMeanShift() {
	Mat img = imread("D:/study/디영처/9주차/fruits.png");
	if (img.empty()) exit(-1);
	cout << "----- exCvMeanShift() -----" << endl;

	resize(img, img, Size(256, 256), 0, 0, CV_INTER_AREA);
	imshow("Src", img);
	imwrite("exCvMeanShift_src.jpg", img);

	pyrMeanShiftFiltering(img, img, 8, 16);

	imshow("Dst", img);
	waitKey();
	destroyAllWindows();
	imwrite("exCvMeanShift_dst.jpg", img);
}

void main() {
	exCvMeanShift();
}
```

## 1-2) 실험결과

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/37b5261e-c18f-471a-8aaa-1c1798cbc47e)

## 1-3)구현 과정

- 이번 과정의 구현은 OpenCv에서 지원하는 Mean Shift Filtering을 사용하였습니다. Meanshift Filtering의 동작과정에 대해서는 low-level로 코드를 구현한 구현과정에 명시했습니다.

# Mean-shift low-level 구현

## 1-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

class Point5D {
	// Mean shift 구현을 위한 전용 포인트(픽셀) 클래스
public:
	float x, y, l, u, v; // 포인트의 좌표와 LUV 값

	void accumPt(Point5D); // 포인트 축적
	void copyPt(Point5D); // 포인트 복사
	float getColorDist(Point5D); // 색상 거리 계산
	float getSpatialDist(Point5D); // 좌표 거리 계산
	void scalePt(float); // 포인트 스케일링 함수 (평균용)
	void setPt(float, float, float, float, float); // 포인트값 설정함수
	void printPt();
};

void Point5D::accumPt(Point5D Pt) {
	x += Pt.x;
	y += Pt.y;
	l += Pt.l;
	u += Pt.u;
	v += Pt.v;
}

void Point5D::copyPt(Point5D Pt) {
	x = Pt.x;
	y = Pt.y;
	l = Pt.l;
	u = Pt.u;
	v = Pt.v;
}

float Point5D::getColorDist(Point5D Pt) {
	return sqrt(pow(l - Pt.l, 2) +
				pow(u - Pt.u, 2) +
				pow(v - Pt.v, 2));
}

float Point5D::getSpatialDist(Point5D Pt) {
	return sqrt(pow(x - Pt.x, 2) + pow(y - Pt.y, 2));
}

void Point5D::scalePt(float scale) {
	x *= scale;
	y *= scale;
	l *= scale;
	u *= scale;
	v *= scale;
}

void Point5D::setPt(float px, float py, float pl, float pa, float pb) {
	x = px;
	y = py;
	l = pl;
	u = pa;
	v = pb;
}

void Point5D::printPt() {
	cout << x << " " << y << " " << l << " " << u << " " << v << endl;
}

class MeanShift {
	/* Mean shift 클래스 */
public:
	float bw_spatial = 8; // Spatial bandwidth
	float bw_color = 16; // Color bandwidth
	float min_shift_color = 0.1; // 최소 컬러변화
	float min_shift_spatial = 0.1; // 최소 워치변화
	int max_steps = 10; // 최대 반복횟수
	vector<Mat> img_split; //채널별로 분할되는 Mat
	MeanShift(float, float, float, float, int); // Bandwidth 설정을 위한 생성자
	void doFiltering(Mat&); // Mean shift filtering 함수
};

MeanShift::MeanShift(float bs, float bc, float msc, float mss, int ms) {
	/* 생성자 */
	bw_spatial = bs;
	bw_color = bc;
	max_steps = ms;
	min_shift_color = msc;
	min_shift_spatial = mss;
}

void MeanShift::doFiltering(Mat& Img) {
	int height = Img.rows;
	int width = Img.cols;
	split(Img, img_split);

	Point5D pt, pt_prev, pt_cur, pt_sum;

	int pad_left, pad_right, pad_top, pad_bottom;
	size_t n_pt, step;

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {

			pad_left = (col - bw_spatial) > 0 ? (col - bw_spatial) : 0;
			pad_right = (col + bw_spatial) < width ? (col + bw_spatial) : width;
			pad_top = (row - bw_spatial) > 0 ? (row - bw_spatial) : 0;
			pad_bottom = (row + bw_spatial) < height ? (row + bw_spatial) : height;

			// <현재 픽셀 세팅>
			pt_cur.setPt(row, col,
				(float)img_split[0].at<uchar>(row, col),
				(float)img_split[1].at<uchar>(row, col),
				(float)img_split[2].at<uchar>(row, col));
			
			// <주변 픽셀 탐색>
			step = 0;
			do {
				pt_prev.copyPt(pt_cur);
				pt_sum.setPt(0, 0, 0, 0, 0);
				n_pt = 0;
				for (int hx = pad_top; hx < pad_bottom; hx++) {
					for (int hy = pad_left; hy < pad_right; hy++) {
						pt.setPt(hx,hy,
							(float)img_split[0].at<uchar>(hx, hy),
							(float)img_split[1].at<uchar>(hx, hy),
							(float)img_split[2].at<uchar>(hx, hy));

						//<Color bandwidth 안에서 축적>
						if (pt.getColorDist(pt_cur) < bw_color) {
							pt_sum.accumPt(pt);
							n_pt++;
						}
					}
				}

				// <축적 결과를 기반으로 현재픽셀 갱신>
				pt_sum.scalePt(1.0 / n_pt); //축적결과 평균
				pt_cur.copyPt(pt_sum);
				step++;
			}
			while((pt_cur.getColorDist(pt_prev)>min_shift_color)&&
				(pt_cur.getSpatialDist(pt_prev)>min_shift_spatial)&&
				(step < max_steps));
			// 변화량 최소조건을 만족할 때 까지 반복
			// 최대 반복횟수 조건도 포함

			// <결과 픽셀 갱신>
			Img.at<Vec3b>(row, col) = Vec3b(pt_cur.l, pt_cur.u, pt_cur.v);
		}
	}
}

void exMyMeanShift() {
	Mat img = imread("D:/study/디영처/9주차/fruits.png");
	if (img.empty()) exit(-1);
	cout << "----- exMyMeanShift() -----" << endl;

	resize(img, img, Size(256, 256), 0, 0, CV_INTER_AREA);
	imshow("Src", img);
	imwrite("exMyMeanShift_src.jpg", img);

	cvtColor(img, img, CV_RGB2Luv);

	MeanShift MSProc(8, 16, 0.1, 0.1, 10);
	MSProc.doFiltering(img);

	cvtColor(img, img, CV_Luv2RGB);

	imshow("Dst", img);
	waitKey();
	destroyAllWindows();
	imwrite("exMyMeanShift_dst.jpg", img);
}

void main() {
	exMyMeanShift();
}
```

## 1-2) 실험결과

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/9ad4f7ed-3fc5-487d-8ac9-4acde2701d2a)

## 1-3)구현 과정

- **`class Point5D :`** Meanshift구현을 위한 픽셀 클래스를 정의합니다. 색상 유사도 측정, 거리 측정 등 픽셀 단위의 연산을 수행하기 위한 함수들이 구현되어 있습니다.
- **`class MeanShift:`** 실질적으로 Meanshift를 진행하기 위한 Class입니다. Class내부에는 Meanshift를 진행하는 함수와 그에 필요한 파라미터 값들이 저장되어 있습니다.
- **`MeanShift::doFiltering :`** 실질적으로 Meanshift clustering을 수행하는 함수입니다. 각 픽셀별로 filtering을 수행합니다. 이 때, padding내에 있는 픽셀들과 색상 유사도와 거리에 따라 Clustering을 진행합니다. 데이터의 밀도가 가장 높은 곳으로 중심을 이동시키고, 이때 정해진 최소 조건을 만족할 때까지 반복합니다.
이 함수의 과정을 통해 Meanshift Clustering이 진행됩니다.

## 1-4)결과 분석

1번은 Meanshift clustering을 사용하여 image segmentation을 진행하는 과제였습니다. Meanshift clustering은 데이터 값의 밀도가 높은 곳을 찾아 군집화를 진행하는 것이라고 이론시간에 배웠습니다. 그래서 이론적으로는 어떤 알고리즘을 사용하는지, 어떻게 segmentation이 진행될지 그려졌지만 직관적으로 이해하지는 못했었습니다. 하지만 OpenCv와 특히 low-level로 구현된 Meanshift clustering코드를 작성해보면서 원리를 직관적으로 알 수 있었습니다. 강의노트에 나와있는 코드를 이해하는 것은 시간이 조금 걸렸지만, 코드를 한줄씩 이해해보려고 노력하면서, Meanshift clustering에서 어떤 동작을 하는 코드인지를 이해하고 알고리즘 자체를 직관적으로 이해할 수 있는 기회가 되었습니다.

지난 과제에서 K-Mean clustering을 이용하여 오늘 실습에서 사용한 fruits.png파일을 image segmentation해보았는데, K값에 따라 Segmentation의 결과값이 달라지고 성능도 너무 자세하게 Segmentation되거나 성능이 좋지 않게 나오는 경우가 많았는데, 이번에 사용한 Meanshift clustering을 사용하여 image segmentation을 진행했을 때에는 성능이 좋게 image segmentation이 된 것을 볼 수 있었습니다.

1번은 Meanshift clustering의 원리에 대한 직관적인 이해와 코드를 통해 분석, 그리고 K-Mean clustering과의 실질적인 차이에  대해 더욱 자세히 배울 수 있는 실습이었습니다.

# 2. Grab cut의 처리가 잘 되는 영상과 잘 되지 않는 영상을 찾아 실험해보고 그 이유를 서술할 것

## 2-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

Mat GrabCut(Mat img, int x1, int y1, int x2, int y2) {
	Rect rect = Rect(Point(x1, y1), Point(x2, y2));

	Mat result, bg_model, fg_model;
	grabCut(img, result, 
			rect, 
			bg_model, fg_model, 
			5, GC_INIT_WITH_RECT);

	compare(result, GC_PR_FGD, result, CMP_EQ);
	// GC_PR_FGD: GrabCut class foreground 픽셀
	// CMP_EQ: comapre 옵션(equal)

	Mat mask(img.size(), CV_8UC3, Scalar(255, 255, 255));
	img.copyTo(mask, result);
	imshow("Mask", mask);
	return result;
}

void hamster_grab() {
	Mat Src = imread("D:/study/디영처/9주차/hamster.jpg", 1);
	imshow("Src", Src);
	Mat Dst = GrabCut(Src, 400, 20, 800, 420);
	imshow("Dst", Dst);
}

void zebra_grab1() {
	Mat Src = imread("D:/study/디영처/9주차/zebra.jpg", 1);
	imshow("Src", Src);
	Mat Dst = GrabCut(Src, 100, 70, 540, 365);
	imshow("Dst", Dst);
}

void zebra_grab2() {
	Mat Src = imread("D:/study/디영처/9주차/zebra.jpg", 1);
	imshow("Src", Src);
	Mat Dst = GrabCut(Src, 100, 70, 540, 370);
	imshow("Dst", Dst);
}

void cheetah_grab() {
	Mat Src = imread("D:/study/디영처/9주차/cheetah.jpg", 1);
	imshow("Src", Src);
	Mat Dst = GrabCut(Src, 100, 50, 610, 309);
	imshow("Dst", Dst);
}

void cameleon_grab() {
	Mat Src = imread("D:/study/디영처/9주차/cameleon.jpg", 1);
	imshow("Src", Src);
	Mat Dst = GrabCut(Src, 80, 50, 520, 380);
	imshow("Dst", Dst);
}

void snake_grab() {
	Mat Src = imread("D:/study/디영처/9주차/snake.jpg", 1);
	imshow("Src", Src);
	Mat Dst = GrabCut(Src, 100, 50, 1050, 760);
	imshow("Dst", Dst);
}

void whitetiger_grab() {
	Mat Src = imread("D:/study/디영처/9주차/white_tiger.jpg", 1);
	imshow("Src", Src);
	Mat Dst = GrabCut(Src, 0, 0, 600,574);
	imshow("Dst", Dst);
}

int main() {
	//hamster_grab();
	//zebra_grab1();
	//zebra_grab2();
	//cheetah_grab();
	//cameleon_grab();
	snake_grab();
	//whitetiger_grab();
	waitKey(0);
	destroyAllWindows();
	return 0;
}
```

## 2-2) 실험 결과

### hamster

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/aa6be7ba-4bc6-4651-990d-a09481a5a358)

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/6b516b14-5051-4f3a-b94b-c2b78443b825)

![Untitled 4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/0d62d2c3-faec-440e-96ef-d4af98f9fa47)

### zebra(배경이 포함되지 않은)

![Untitled 5](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/c83dd764-80e0-486e-82c7-e1e0ca87816b)

![Untitled 6](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/edcceb0b-4fae-41a3-9621-c760eb3719ce)

### zebra(배경이 포함된)

![Untitled 7](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/c0453778-f05d-4aa6-b600-ab16d8e6e28a)

![Untitled 8](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/16336b8e-b9dd-414e-a89a-ec5484d70388)

### cheetah

![Untitled 9](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/1f61caf4-2ae4-455e-ac15-2d0dc6edbf3c)

![Untitled 10](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/b278cb3d-3da6-4b6c-805c-8d8613014178)

### Cameleon

![Untitled 11](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/960c64f4-ce25-4eaf-a7db-e2fc0ad54395)

![Untitled 12](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/1250bc2d-2689-4453-b692-2038cddb7c37)

### snake

![Untitled 13](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/765616fe-7b3a-4b6c-b02e-a62e5fe137b6)

![Untitled 14](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/d8a29c25-f420-4fad-8810-a3f3b0a6da9c)

![Untitled 15](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/2816ac86-a4f5-4fae-8241-e6f2ab0f0dc5)

### white tiger

![Untitled 16](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/a5f19c12-6a8e-43c4-8f74-b80e349ff9d9)

![Untitled 17](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/7d39fdbc-250d-4b44-ad4f-b80afd14b742)

## 2-3) 구현 과정

- 2번에 대한 구현은 OpenCv를 활용하여 Grab Cut을 할 수 있도록 코드를 작성하였습니다. bounding box를 지정하고, bounding box 안에서 object를 찾는 동작을 하는 함수입니다.

## 2-4) 결과 분석

- 우선 실험을 진행하기 전, 단순하게 생각해보았을 때, Grab Cut이 잘되고 잘되지 않는 원인 중 가장 큰 부분은 배경, 즉 Background에 달렸다고 생각했습니다. 따라서 결과를 어느 정도 예측하고 Grab Cut이 잘될 것 같은 동물 사진과 잘 되지 않을 것 같은 동물사진들을 각각 준비해보았습니다. 물론 제 예측이 틀린 것은 아니지만 제가 예측하지 못했던 변수들이 있었습니다. 따라서 잘된 영상과 잘 되지 않는 영상의 결과를 보고 그 원인에 대해서 분석해보았습니다.

### [햄스터 사진]

![Untitled 18](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/62e71120-ae1a-4f89-b04d-c8e41cd760d9)

- 우선 햄스터 사진에 대한 Grab Cut은 결과가 잘나왔습니다. 물론 햄스터가 아닌 해바라기 씨도 추출되었지만, 이 정도면 Grab Cut의 결과가 잘나왔다고 볼 수 있습니다. 이는 햄스터는 흰색이고, 배경과 색이 많이 차이나기 때문에 좋은 결과가 나왔습니다. 눈이 추출되지 않는 이유도 이 때문입니다.

### [얼룩말 사진]

![Untitled 19](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4b79fc23-200a-441f-ac71-aeb366abbc08)

**`Mat Dst = GrabCut(Src, 100, 70, 540, 365);`**

![Untitled 20](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/35843228-5fd3-4e49-bb60-924710c7d259)

**`Mat Dst = GrabCut(Src, 100, 70, 540, 370);`**

- 얼룩말 사진에 대한 실험은 2가지를 진행하였습니다. 배경이 거의 없이 추출된 영상과 배경이 많이 포함된 영상입니다. 위 두 결과의 차이는 바로 bounding box입니다. 아래의 결과를 도출한 bounding box의 영역이 더 많은 배경 부분을 포함하고 있습니다. 따라서 이번 결과를 통해 알 수 있었던 것은 bounding box에 background가 최대한 포함되지 않도록, 즉 정교한 Bounding box가 필요하다는 점입니다. 하지만 box형태로는 한계가 있고 다른 방법을 사용한다면 얼룩말은 배경과 다소 다르기 때문에 Grab cut처리가 잘 될 것이라고 생각합니다. 하지만 얼룩말의 발굽 부분은 배경에 가려져있기도 하고, 색상도 다소 비슷하여 Grab cut이 잘 되지 않았습니다.
- 여기에서 알 수 있었던 것은 Grab cut의 성능이 좋기 위해서는 Bounding box가 중요하다는 것이지만, 더 나아가 생각해본다면 물체에 텅 비어있는 부분, 즉 구멍이나 공간이 있으면 Grab cut의 성능이 떨어진다고 생각해볼 수 있었습니다. 따라서 얼룩말의 다리 공간 때문에 Grab cut의 성능이 좋지 않음을 알 수 있습니다. 하지만 얼룩말의 Grab cut성능이 좋지 않은 가장 큰 이유는 얼룩말 다리 부분이 배경과 유사한 색을 지니고 있다는 점이라고 분석할 수 있었습니다.

### [치타 사진]

![Untitled 21](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/bcb7c6b9-dce3-4752-94ef-0eda6b2c2179)

- 치타사진은 Grab cut처리가 잘될 것이라고 예상했고, bounding box에 배경이 다소 많이 포함되기는 하지만, 주변 배경과 치타의 색상이 많은 차이가 나기도 하고, edge가 잘 보존되어 있어서 Grab cut의 성능이 좋은 것을 알 수 있었습니다.

### [카멜레온 사진]

![Untitled 22](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/62e86b5d-3a94-462f-9641-f1edd1f4378c)

- 카멜레온 사진에 대한 실습은 가장 의외의 결과를 냈던 실습이었습니다. source img를 보면 알 수 있듯이 카멜레온의 색과 주변 배경과의 색상이 비슷하기 때문에 Grab cut의 성능이 매우 떨어질 것이라고 예상했습니다. 하지만, 결과를 볼 수 있듯이 의외로 좋은 성능의 Grab cut처리가 된 것을 볼 수 있습니다. 그 원인을 보았을 때, 뒷 배경이 blurry하고 카멜레온의 edge가 분명하기 때문일 것이라고 분석했습니다. 하지만 새롭게 알게 되었던 것은, 몸체와 달리 꼬리 부분은 Grab cut이 되지 않은 것을 볼 수 있는데, 이는 그림자 때문입니다. 그림자도 어떻게 본다면 색상을 어둡게 하는 역할을 하기 때문에, 색상의 차이로 인해 Grab cut이 잘 처리되지 않았다고 볼 수 있지만 그림자, 배경에 가려짐 등 이미지 내에서의 환경으로 인해 Grab cut의 성능이 달라질 수 있다는 것을 새삼 깨달을 수 있었습니다.

### [뱀 사진]

![Untitled 23](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/2853493f-0592-4566-92c4-348184ee5b32)

- 뱀 사진에 대한 실습은 가장 Grab cut 성능이 안좋을 것이라고 예측하고 실험했던 사진입니다. 뱀과 배경의 색이 매우 유사하여 Grab cut의 성능이 좋지 않을 것이라고 예측할 수 있었고, 결과도 그렇게 나왔습니다. 결과를 보면 눈이 덮인 흰색 배경을 제외하면 뱀이 Grab cut된 것이 아닌, 배경 전체가 Grab cut되었다는 것을 알 수 있습니다. 따라서 Grab cut의 성능에 가장 많은 영향을 미치는 것은 전경과 배경의 색상 차이라는 것을 다시 한번 깨달을 수 있었습니다.

### [백호 사진]

![ffff](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/db3f2318-d903-4391-ada9-f7bec9757aa3)

- 백호 사진은 가장 Grab cut의 성능이 좋을 것이라고 예측했던 사진 중 하나 입니다. 백호의 흰색과 주변의 초록색의 색상이 매우 다르기 때문입니다. 결과 또한 예측처럼 Grab cut성능이 매우 좋게 나온 것을 볼 수 있었습니다. 여기서 수염 부분이 포함되지 않은 것을 확인할 수 있었는데, 이는 그림자와 같은 원리로 이미지 내에서의 빛에 따라 Grab cut의 성능이 달라질 수 있음을 다시 한번 확인할 수 있었던 실습입니다.

2번은 위 다양한 동물 사진을 통한 Grab cut실습을 통해 Grab cut의 성능을 좌우하는 요소에 대해 분석하고 직관적으로 깨달을 수 있는 실습이었습니다. 제가 분석한 Grab cut의 성능을 판가름하는 요소는 다음과 같습니다.

1. **전경과 배경의 색상 차이**
2. **이미지 내의 그림자와 같은 빛의 요소**
3. **전경의 edge 보존 여부**
4. **이미지 내에서 물체가 배경 요소로 인해 가려진 정도**
5. **Bounding box의 정교성**

저는 실습 전 ‘전경과 배경의 색상 차이’만이 Grab cut의 성능을 판가름한다고 단순하게 생각했었는데, 실습을 진행하면서 Grab cut의 성능을 좌우하는 요소들이 생각보다 많다는 것을 알 수 있었고, 실험 결과를 통해 직접 확인해보면서 직관적으로 깨달을 수 있었습니다.