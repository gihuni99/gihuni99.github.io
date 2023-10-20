---
title: Color Processing and Clustering
date: 2023-04-14 00:00:00 +09:00
categories: [Digital Image Processing]
tags:
  [
    Computer Vision,
    Digital Image Processing,
    Clustering,
	Color Conversion
  ]
pin: true
---

# 1. 임의의 과일 사진을 입력했을 때 해당 과일의 색을 문자로 출력하고 과일 영역을 컬러로 정확히 추출하는 코드를 구현 (BGR to HSV와 inRange() 함수는 직접 구현할 것)

## 1-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

Mat MyBgr2Hsv(Mat src_img) {
	double b, g, r, h, s, v;
	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			b = (double)src_img.at<Vec3b>(y, x)[0];
			g = (double)src_img.at<Vec3b>(y, x)[1];
			r = (double)src_img.at<Vec3b>(y, x)[2];

			double my_max = max(max(r, g), b);
			double my_min = min(min(r, g), b);

			v = my_max;

			s = (v != 0) ? (my_max - my_min) / my_max : 0;
			if (my_max == r) h = 30 * ((g - b) / (my_max - my_min));
			else if (my_max == g) h = 30 * (2 + (b - r) / (my_max - my_min));
			else if (my_max == b) h = 30 * (4 + (r - g) / (my_max - my_min));
	

			if (h < 0) h = h + 180;

			s = s * 255;

			dst_img.at<Vec3b>(y, x)[0] = (uchar)h;
			dst_img.at<Vec3b>(y, x)[1] = (uchar)s;
			dst_img.at<Vec3b>(y, x)[2] = (uchar)v;

		}
	}
	return dst_img;
}

int inRange(Mat hsv_img, Scalar min, Scalar max, Mat dst_img) {

	int count = 0;
	double h,s,v;
	for (int y = 0; y < hsv_img.rows; y++) {
		for (int x = 0; x < hsv_img.cols; x++) {
			h = hsv_img.at<Vec3b>(y, x)[0];
			s = hsv_img.at<Vec3b>(y, x)[1];
			v = hsv_img.at<Vec3b>(y, x)[2];
			if (min[0] <= h && max[0] > h && min[1] <= s && max[1] > s&& min[2] <= v && max[2] > v) {
				dst_img.at<Vec3b>(y, x) = { 255 , 255, 255 };
				count += 1;
			}
		}
	}
	return count;
}

Mat fruit_color(Mat src_img, Scalar color) {
	Mat dst_img = Mat::zeros(src_img.size(), CV_8UC3);
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			int b = src_img.at<Vec3b>(y, x)[0];
			int g = src_img.at<Vec3b>(y, x)[1];
			int r = src_img.at<Vec3b>(y, x)[2];
			if (r==255&&g==255&&b==255) {
				dst_img.at<Vec3b>(y, x)[0] = color[0];
				dst_img.at<Vec3b>(y, x)[1] = color[1];
				dst_img.at<Vec3b>(y, x)[2] = color[2];
			}
		}
	}
	return dst_img;
}

void main() {
	Mat src_img = imread("D:/study/디영처/7주차/fruits.png", 1);
	imshow("src_img", src_img);

	Mat hsv = MyBgr2Hsv(src_img);
	imshow("hsv_img", hsv);

	int fruit_color_num=0; //image에서 가장 많은 색을 차지하는 찾기 위한 변수
	int cn1, cn2, cn3, cn4, cn5, cn6;

	Mat dst_img_r = Mat::zeros(src_img.size(), CV_8UC3);
	cn1=inRange(hsv, Scalar(160, 20, 20), Scalar(20, 230, 230), dst_img_r); //hsv Red 영역
	if (cn1 > fruit_color_num) fruit_color_num = cn1;

	Mat dst_img_y = Mat::zeros(src_img.size(), CV_8UC3);
	cn2 = inRange(hsv, Scalar(20, 20, 20), Scalar(40, 230, 230), dst_img_y);//hsv Yellow 영역
	if (cn2 > fruit_color_num) fruit_color_num = cn2;

	Mat dst_img_g = Mat::zeros(src_img.size(), CV_8UC3);
	cn3 = inRange(hsv, Scalar(40, 20, 20), Scalar(80, 230, 230), dst_img_g);//hsv Green 영역
	if (cn3 > fruit_color_num) fruit_color_num = cn3;

	Mat dst_img_c = Mat::zeros(src_img.size(), CV_8UC3);
	cn4 = inRange(hsv, Scalar(80, 20, 20), Scalar(100, 230, 230), dst_img_c);//hsv Cyan 영역
	if (cn4 > fruit_color_num) fruit_color_num = cn4;

	Mat dst_img_b = Mat::zeros(src_img.size(), CV_8UC3);
	cn5 = inRange(hsv, Scalar(100, 20, 20), Scalar(140, 230, 230), dst_img_b);//hsv Blue 영역
	if (cn5 > fruit_color_num) fruit_color_num = cn5;

	Mat dst_img_m = Mat::zeros(src_img.size(), CV_8UC3);
	cn6 = inRange(hsv, Scalar(140, 20, 20), Scalar(160, 230, 230), dst_img_m);//hsv Magenta 영역
	if (cn6 > fruit_color_num) fruit_color_num = cn6;

	Mat dst_img_final = Mat::zeros(src_img.size(), CV_8UC3);

	if (fruit_color_num == cn1) {
		dst_img_final=fruit_color(dst_img_r, Scalar(0, 0, 255));
		printf("Fruit's Color: Red");
		imshow("dst_img_r", dst_img_r);
		imshow("dst_img_final", dst_img_final);
	}
	else if (fruit_color_num == cn2) {
		dst_img_final = fruit_color(dst_img_y, Scalar(0, 255, 255));
		printf("Fruit's Color: Yellow");
		imshow("dst_img_y", dst_img_y);
		imshow("dst_img_final", dst_img_final);
	}
	else if (fruit_color_num == cn3) {
		dst_img_final = fruit_color(dst_img_g, Scalar(0, 255, 0));
		printf("Fruit's Color: Green");
		imshow("dst_img_g", dst_img_g);
		imshow("dst_img_final", dst_img_final);
	}
	else if (fruit_color_num == cn4) {
		dst_img_final = fruit_color(dst_img_c, Scalar(255, 255, 0));
		printf("Fruit's Color: Cyan");
		imshow("dst_img_c", dst_img_c);
		imshow("dst_img_final", dst_img_final);
	}
	else if (fruit_color_num == cn5) {
		dst_img_final = fruit_color(dst_img_b, Scalar(255, 0, 0));
		printf("Fruit's Color: Blue");
		imshow("dst_img_b", dst_img_b);
		imshow("dst_img_final", dst_img_final);
	}
	else if (fruit_color_num == cn6) {
		dst_img_final = fruit_color(dst_img_m, Scalar(255, 0, 255));
		printf("Fruit's Color: Magenta");
		imshow("dst_img_m", dst_img_m);
		imshow("dst_img_final", dst_img_final);
	}

	waitKey(0);
	destroyAllWindows();
}
```

## 1-2) 실험결과

### [fruits.png에 대한 결과]

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/db525ab5-f894-4eac-9d78-56b0e30ab155)

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/60f6bdf5-4558-403a-b8ed-ccb073eac8ce)

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/7ebbf8c4-1c3f-404e-b84b-92c93a2135c4)

### [수박 사진에 대한 결과]

fruits.png는 다양한 과일들이 나와서 비슷한 색의 과일들도 함께 추출되었습니다. 따라서 단일의 과일사진이 들어가면 결과가 잘 나오는지 확인해보았고, 그 사진을 수박 사진으로 사용하였습니다.

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/83f16b80-313e-47f4-ae94-01af23ad6a8d)

![Untitled 4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/5b8a513d-2310-4afd-914e-733cf293cb60)

![Untitled 5](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ee6e4220-6976-4336-abe7-d4c956253ce4)

### [포도 사진에 대한 결과]

![Untitled 6](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/c68f0dc3-c139-46b7-a0a3-5106fa8f895c)

![Untitled 7](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8bbde456-073e-4f7e-933a-a343f6fe5f55)

![Untitled 8](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/a37e9f6f-359d-4f50-b095-1c56b3fdea0f)

## 1-3)구현 과정

- **`Mat MyBgr2Hsv(Mat src_img) :`**  MyBgr2Hsv함수를 사용하여 RGB영상을 HSV 영상으로 변환합니다.

![Untitled](%E1%84%83%E1%85%B5%E1%84%8C%E1%85%B5%E1%84%90%E1%85%A5%E1%86%AF%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%89%E1%85%A1%E1%86%BC%E1%84%8E%E1%85%A5%E1%84%85%E1%85%B5%E1%84%89%E1%85%A5%E1%86%AF%E1%84%80%E1%85%A8%20%E1%84%89%E1%85%B5%E1%86%AF%E1%84%89%E1%85%B3%E1%86%B8%20%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%206fee2e08332e43d09134e1a37009bb6d/Untitled%209.png)

HSV의 범위는 0-180으로 설정하였고, 색상은 Red, Green, Blue, Cyan, Magenta, Yellow로 설정하였습니다. 채도와 명도는 각각 0-255의 범위를 갖도록 하였습니다. 해당 코딩은 강의노트를 참고하여 하였으며, RGB를 HSV로 변환하는 일반적인 알고리즘입니다.

- `**int inRange(Mat hsv_img, Scalar min, Scalar max, Mat dst_img):**`  inRange함수는 색을 나타내는 범위를 min, max로 주어줍니다. 예를 들어 Red로 전환하는 pixel의 범위는 Scalar(160, 20, 20)에서 Scalar(20, 230, 230)까지 입니다. 즉 전체 180중에서 160-180, 0-20까지의 색상을 red라고 나타내는 것입니다. 채도와 명도는 20-230의 범위로 모두 동일하게 주었습니다. 이때 색상별로 서로 다른 범위가 주어지는데, 색상이라고 판단하는 pixel의 마스크를 생성하기 위해 해당 픽셀을 255,255,255로 바꿔줍니다.
이 함수를 통해 각 색상의 픽셀이 몇개인지 유지하는 count를 반환하고, 이를 통해 image의 과일이 무슨 색인지 알 수 있도록 함수를 작성하였습니다.
- **`Mat fruit_color(Mat src_img, Scalar color):`**  fruit_color함수는 가장 픽셀수가 많은 색상의 마스크를 입력받고, 그 mask에 따라 해당하는 색상을 대입해주는 함수입니다. 이를 통해 최종적으로 image의 과일의 색상을 해당 색상으로 주어줍니다.
- **`main 함수:`** 
main함수에서 거의 모든 동작을 실행합니다. 위에서 언급한 것처럼 색상을 Red, Green, Blue, Cyan, Magenta, Yellow로 주어줬습니다. HSV 영상의 색상 스펙트럼을 육안으로 확인하고 경험적으로 범위를 주어줬습니다.

```cpp
Mat dst_img_y = Mat::zeros(src_img.size(), CV_8UC3);
cn2 = inRange(hsv, Scalar(20, 20, 20), Scalar(40, 230, 230), dst_img_y);//hsv Yellow 영역
if (cn2 > fruit_color_num) fruit_color_num = cn2;
```

- 예시로 위는 yellow의 범위입니다. 색상 0-180의 범위 중 20정도의 범위를 주어주었고, 여기서 Primary Color는 40의 범위, Second Primary Color는 20의 범위로 설정해주었습니다.

## 1-4)결과 분석

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/60f6bdf5-4558-403a-b8ed-ccb073eac8ce)

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/7ebbf8c4-1c3f-404e-b84b-92c93a2135c4)

fruit.png를 통해 실습한 결과는 다소 아쉬운 결과가 나왔던 것 같습니다. 그 원인을 생각해보았을 때, 우선 과일의 수가 많고, 비슷한 색상의 과일로 인해 같은 과일만을 정확하게 segmentation하는 것이 어려웠습니다. 또한 색상의 범위를 second primary color까지만 주다보니 다소 부정확했습니다. 육안으로 보았을 때, 과일의 색은 연두색, 주황색, 보라색 등 코딩에 포함되지 않은 색상이 많습니다. 따라서 Second primary color라는 다소 포괄적인 범위로 인해 완벽한 결과를 얻을 수 없었습니다. 하지만 HSV를 더욱 세분화하여 범위를 주어준다면 더 정확한 결과를 얻을 수 있을 것이라고 생각할 수 있었습니다. 또한 과일이 여러개라서 비교적 많은 노란색 계열의 과일을 인식하는 것을 볼 수 있었는데, 하나의 과일이 있는 image를 입력으로 주었을 때, 정확하게 영역을 추출하는지 실험해보고 싶었습니다. 

따라서 수박과 포도 사진으로 실험을 진행하였고, 강의에서 주어진 여러개의 과일 사진보다 직관적으로 영역이 추출되는 것을 확인할 수 있었습니다. 이를 통해 코드가 정상적으로 동작하는 것을 확인하였습니다. 

# 2. beach.jpg에 대해 군집 간 평균 색으로 segmentation을 수행하는 k-means clustering 수행 (OpenCV 사용, 군집 수인 k에 따른 결과 확인 및 분석)

## 2-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

Mat CvKMeans(Mat src_img, int k) {

	Mat samples(src_img.rows * src_img.cols, src_img.channels(), CV_32F);
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					samples.at<float>(y + x * src_img.rows, z) =
						(float)src_img.at<Vec3b>(y, x)[z];
				}
			}
			else {
				samples.at<float>(y + x * src_img.rows) =
					(float)src_img.at<uchar>(y, x);
			}
		}
	}

	//<OpenCV k-means 수행>
	Mat labels;
	Mat centers;
	int attempts = 5;
	kmeans(samples, k, labels,
		TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
		attempts, KMEANS_PP_CENTERS,
		centers);

	//<1차원 벡터 -> 2차원 영상>
	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * src_img.rows, 0);
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					dst_img.at<Vec3b>(y, x)[z] =
						(uchar)centers.at<float>(cluster_idx, z);
						//군집판별 결과에 따라 각 군집의 중앙값으로 결과 생성
				}
			}
			else {
				dst_img.at<uchar>(y, x) =
					(uchar)centers.at<float>(cluster_idx, 0);
			}
		}
	}

	return dst_img;
}

void main() {
	Mat src_img = imread("D:/study/디영처/7주차/beach.jpg", 1);
	imshow("src_img", src_img);
	Mat dst_img1,dst_img2,dst_img3,dst_img4;

	dst_img1 = CvKMeans(src_img, 2);
	imshow("K=2", dst_img1);

	dst_img2 = CvKMeans(src_img, 3);
	imshow("K=3", dst_img2);

	dst_img3 = CvKMeans(src_img, 4);
	imshow("K=4", dst_img3);

	dst_img4 = CvKMeans(src_img, 4);
	imshow("K=5", dst_img4);

	waitKey(0);
	destroyAllWindows();
}
```

## 2-2) 실험 결과

### [원본 사진]

![Untitled 10](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/913b71e3-5a1f-4a45-8332-88999c4ade0b)

### [K=2]

![Untitled 11](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/0c4259fb-72f0-4c3d-898b-36d85a900d23)

### [K=3]

![Untitled 12](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/29037f37-d836-40d8-90d0-03b1e32bb097)

### [K=4]

![Untitled 13](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/9875cab2-c6db-44ab-8232-a22bd50eb930)

### [K=5]

![Untitled 14](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4ef36d9e-80d3-4cbf-a44c-bf76a899055f)

## 2-3) 구현 과정

- **`Mat CvKMeans(Mat src_img, int k):`**  CvKMeans함수는 Opencv에서 제공하는 kmeans()함수를 통해 clustering을 할 수 있는 함수입니다. k개의 clustering center를 정하고, 이를 통해 clustering이 진행됩니다. 이를 통해 k개의 비슷한 pixel을 가진 군집이 모이게 되고, 적은 수의 색상으로 영상을 표현하여, segmentation이 수행됩니다. 과제 2번에서는 위 함수 하나만으로 K Mean Clustering을 진행하게 됩니다.

## 2-4) 결과 분석

과제 2번은 Opencv kmean을 활용하여 K-Mean Clustering을 수행하는 코드를 작성하였습니다. 이를 beach.jpg 이미지에 적용하여 결과를 확인하였습니다. 결과는 K의 값을 변형시켜서 확인해보고자 하였습니다. 실험을 진행하기 전에 결과를 예상해보았을 때, K값이 늘어날 때, 점점 더 image가 정교하게 표현될 것이라고 예상했습니다. 그리고 그 예상 결과는 맞았습니다.

![Untitled 11](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/0c4259fb-72f0-4c3d-898b-36d85a900d23)

![Untitled 14](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4ef36d9e-80d3-4cbf-a44c-bf76a899055f)

왼쪽은 K=2의 값을 통해 K-Mean Clustering을 진행한 image이고, 오른 쪽은 K=5를 사용하여 진행한 image입니다. 육안으로도 확인 할 수 있듯이 더 정교하게 표현된 것을 볼 수 있습니다. 이는 K값이 클수록 다양한 화소값을 사용하여 segmentation을 수행하기 때문입니다.

위 결과를 통해 알 수 있듯이 과제 2번에 대한 코드가 정상적으로 동작하는 것을 볼 수 있습니다.

# 3. 임의의 과일 사진에 대해 K-means clustering로 segmentation 수행 후, 과일 영역 컬러 추출 수행 (1번 과제 결과와 비교)

## 3-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

Mat MyBgr2Hsv(Mat src_img) {
	double b, g, r, h, s, v;
	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			b = (double)src_img.at<Vec3b>(y, x)[0];
			g = (double)src_img.at<Vec3b>(y, x)[1];
			r = (double)src_img.at<Vec3b>(y, x)[2];

			double my_max = max(max(r, g), b);
			double my_min = min(min(r, g), b);

			v = my_max;

			s = (v != 0) ? (my_max - my_min) / my_max : 0;
			if (my_max == r) h = 30 * ((g - b) / (my_max - my_min));
			else if (my_max == g) h = 30 * (2 + (b - r) / (my_max - my_min));
			else if (my_max == b) h = 30 * (4 + (r - g) / (my_max - my_min));

			if (h < 0) h = h + 180;

			s = s * 255;

			dst_img.at<Vec3b>(y, x)[0] = (uchar)h;
			dst_img.at<Vec3b>(y, x)[1] = (uchar)s;
			dst_img.at<Vec3b>(y, x)[2] = (uchar)v;

		}
	}
	return dst_img;
}
int inRange(Mat hsv_img, Scalar min, Scalar max, Mat dst_img) {

	int count = 0;
	double h,s,v;
	for (int y = 0; y < hsv_img.rows; y++) {
		for (int x = 0; x < hsv_img.cols; x++) {
			h = hsv_img.at<Vec3b>(y, x)[0];
			s = hsv_img.at<Vec3b>(y, x)[1];
			v = hsv_img.at<Vec3b>(y, x)[2];
			if (min[0] <= h && max[0] > h && min[1] <= s && max[1] > s&& min[2] <= v && max[2] > v) {
				dst_img.at<Vec3b>(y, x) = { 255 , 255, 255 };
				count += 1;
			}
		}
	}
	return count;
}

Mat fruit_color(Mat src_img, Scalar color) {
	Mat dst_img = Mat::zeros(src_img.size(), CV_8UC3);
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			int b = src_img.at<Vec3b>(y, x)[0];
			int g = src_img.at<Vec3b>(y, x)[1];
			int r = src_img.at<Vec3b>(y, x)[2];
			if (r == 255 && g == 255 && b == 255) {
				dst_img.at<Vec3b>(y, x)[0] = color[0];
				dst_img.at<Vec3b>(y, x)[1] = color[1];
				dst_img.at<Vec3b>(y, x)[2] = color[2];
			}
		}
	}
	return dst_img;
}

Mat CvKMeans(Mat src_img, int k) {

	Mat samples(src_img.rows * src_img.cols, src_img.channels(), CV_32F);
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					samples.at<float>(y + x * src_img.rows, z) =
						(float)src_img.at<Vec3b>(y, x)[z];
				}
			}
			else {
				samples.at<float>(y + x * src_img.rows) =
					(float)src_img.at<uchar>(y, x);
			}
		}
	}

	//<OpenCV k-means 수행>
	Mat labels;
	Mat centers;
	int attempts = 5;
	kmeans(samples, k, labels,
		TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
		attempts, KMEANS_PP_CENTERS,
		centers);

	//<1차원 벡터 -> 2차원 영상>
	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * src_img.rows, 0);
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					dst_img.at<Vec3b>(y, x)[z] =
						(uchar)centers.at<float>(cluster_idx, z);
					//군집판별 결과에 따라 각 군집의 중앙값으로 결과 생성
				}
			}
			else {
				dst_img.at<uchar>(y, x) =
					(uchar)centers.at<float>(cluster_idx, 0);
			}
		}
	}

	return dst_img;
}

void main() {
	Mat src_img = imread("D:/study/디영처/7주차/fruits.png", 1);
	imshow("src_img", src_img);

	Mat dst_img = CvKMeans(src_img, 6);
	imshow("K=6", dst_img);

	Mat hsv = MyBgr2Hsv(dst_img);
	imshow("hsv_img", hsv);

	int fruit_color_num = 0; //image에서 가장 많은 색을 차지하는 찾기 위한 변수
	int cn1, cn2, cn3, cn4, cn5, cn6;

	Mat dst_img_r = Mat::zeros(src_img.size(), CV_8UC3);
	cn1 = inRange(hsv, Scalar(160, 20, 20), Scalar(20, 230, 230), dst_img_r); //hsv Red 영역
	if (cn1 > fruit_color_num) fruit_color_num = cn1;

	Mat dst_img_y = Mat::zeros(src_img.size(), CV_8UC3);
	cn2 = inRange(hsv, Scalar(20, 20, 20), Scalar(40, 230, 230), dst_img_y);//hsv Yellow 영역
	if (cn2 > fruit_color_num) fruit_color_num = cn2;

	Mat dst_img_g = Mat::zeros(src_img.size(), CV_8UC3);
	cn3 = inRange(hsv, Scalar(40, 20, 20), Scalar(80, 230, 230), dst_img_g);//hsv Green 영역
	if (cn3 > fruit_color_num) fruit_color_num = cn3;

	Mat dst_img_c = Mat::zeros(src_img.size(), CV_8UC3);
	cn4 = inRange(hsv, Scalar(80, 20, 20), Scalar(100, 230, 230), dst_img_c);//hsv Cyan 영역
	if (cn4 > fruit_color_num) fruit_color_num = cn4;

	Mat dst_img_b = Mat::zeros(src_img.size(), CV_8UC3);
	cn5 = inRange(hsv, Scalar(100, 20, 20), Scalar(140, 230, 230), dst_img_b);//hsv Blue 영역
	if (cn5 > fruit_color_num) fruit_color_num = cn5;

	Mat dst_img_m = Mat::zeros(src_img.size(), CV_8UC3);
	cn6 = inRange(hsv, Scalar(140, 20, 20), Scalar(160, 230, 230), dst_img_m);//hsv Magenta 영역
	if (cn6 > fruit_color_num) fruit_color_num = cn6;

	Mat dst_img_final = Mat::zeros(src_img.size(), CV_8UC3);

	if (fruit_color_num == cn1) {
		dst_img_final = fruit_color(dst_img_r, Scalar(0, 0, 255));
		printf("Fruit's Color: Red");
		imshow("dst_img_r", dst_img_r);
		imshow("dst_img_final", dst_img_final);
	}
	else if (fruit_color_num == cn2) {
		dst_img_final = fruit_color(dst_img_y, Scalar(0, 255, 255));
		printf("Fruit's Color: Yellow");
		imshow("dst_img_y", dst_img_y);
		imshow("dst_img_final", dst_img_final);
	}
	else if (fruit_color_num == cn3) {
		dst_img_final = fruit_color(dst_img_g, Scalar(0, 255, 0));
		printf("Fruit's Color: Green");
		imshow("dst_img_g", dst_img_g);
		imshow("dst_img_final", dst_img_final);
	}
	else if (fruit_color_num == cn4) {
		dst_img_final = fruit_color(dst_img_c, Scalar(255, 255, 0));
		printf("Fruit's Color: Cyan");
		imshow("dst_img_c", dst_img_c);
		imshow("dst_img_final", dst_img_final);
	}
	else if (fruit_color_num == cn5) {
		dst_img_final = fruit_color(dst_img_b, Scalar(255, 0, 0));
		printf("Fruit's Color: Blue");
		imshow("dst_img_b", dst_img_b);
		imshow("dst_img_final", dst_img_final);
	}
	else if (fruit_color_num == cn6) {
		dst_img_final = fruit_color(dst_img_m, Scalar(255, 0, 255));
		printf("Fruit's Color: Magenta");
		imshow("dst_img_m", dst_img_m);
		imshow("dst_img_final", dst_img_final);
	}

	waitKey(0);
	destroyAllWindows();

}
```

## 3-2) 실험 결과

### [fruits.png에 대한 결과]

![Untitled 15](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/39f2ca41-97ba-4873-8277-d19ef5fe42e8)

![Untitled 16](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/90f70dae-25fb-4bc7-9621-46c193eaee2f)

![Untitled 17](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8b357937-32b6-449e-a5b1-0775cbd72a37)

### [수박 사진에 대한 결과]

![Untitled 18](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/73aa01a9-a5c0-4543-8396-652cf1d16e91)

![Untitled 19](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/ee5ca840-0d2a-4c87-ba9c-1061ac7e2d46)

![Untitled 20](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/09fbf348-b2bb-4bf1-aa3c-a46a004462fe)

### [포도 사진에 대한 결과]

![Untitled 21](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8ed4b4c3-ea0e-4ed8-9841-079548239a07)

![Untitled 22](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4124f82b-4c3f-453c-8831-733d613e3052)

![Untitled 23](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/23fe861a-a528-42a5-8acf-c11499290679)

## 3-3) 구현 과정

해당 과제는 따로 코드를 작성한 것이 없습니다. K-Mean Clustering을 통해 segmentation을 수행하는 것은 과제2번의 코드를 사용하였고, 과일 영역의 컬러를 추출하여 나타내는 것은 과제 1번의 코드를 사용하였습니다. 따라서 과제 3번은 분석을 위주로 과제를 진행하였습니다.

## 3-4) 결과 분석

### [fruits.png에 대한 실험 결과 비교]

![Untitled 24](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/501ad266-2c8f-40a3-ad06-32b99b64b9c7)

![Untitled 25](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/79456d94-562c-445d-bcd7-3abbaa6b068e)

**[과제 1번의 결과]**

(HSV변환을 통한 컬러 추출)

![Untitled 26](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/e3d75d45-b443-4d8a-9902-fc3cce38ea01)

**[과제 2번의 결과]**

(K-Mean Clustering을 한 후 HSV변환을 통한 컬러 추출)

- 위 실험은 여러가지 과일이 있는 image를 실험적으로 진행한 결과입니다. 사실 이 image에 대해서는 K-Mean clustering을 통해 Segmentation을 진행한 후 컬러를 추출한 결과가 더 부정확하다고 할 수 있습니다. 이유는 K-Mean Clustering을 통해 image를 표현하는 화소 값이 줄어들고, 이로 인해 비슷한 색상의 과일들이 더욱 비슷한 화소값을 가지게 됩니다. 따라서 더 많은 범위에서 컬러가 추출되어 부정확한 결과가 나옵니다. 하지만, 단일 과일에 대해서는 더욱 정확한 결과가 나올 것이라고 예상되고, 실제로 그렇게 결과가 나왔습니다.

### [수박 사진에 대한 실험 결과 비교]

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/412d76ca-b81a-4ed6-86a3-c5fc94a6afa6)

![Untitled 27](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/f2269ec8-e25c-48aa-b876-6b4220609705)

**[1번의 결과]**

(HSV변환을 통한 컬러 추출)

![Untitled 20](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/de7c8125-7ddf-4708-aebf-8bfbdabc4f79)

**[2번의 결과]**

(K-Mean Clustering을 한 후 HSV변환을 통한 컬러 추출)

### [포도 사진에 대한 실험 결과 비교]

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/324f76b2-0394-43e2-81ed-d41b8e46206d)

![Untitled 28](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/8bac5c3a-66f8-4f87-b5ee-4c1ad92ceeb8)

**[1번의 결과]**

(HSV변환을 통한 컬러 추출)

![Untitled 29](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/50d13208-5648-4dda-9be3-6c0e72cd51a2)

**[2번의 결과]**

(K-Mean Clustering을 한 후 HSV변환을 통한 컬러 추출)

- 위 수박과 포도 사진에 대한 결과를 비교해보았을 떄, K-Mean Clustering을 한 후 컬러를 추출하는 것이 더 정확한 결과를 내는 것을 확인할 수 있었습니다. HSV변환만을 통해 컬러를 추출하는 결과는 과일의 영역임에도 컬러가 추출되지 않는 부분이 있는 반면, K-Mean Clustering을 통해 Segmentation을 진행한 후 컬러를 추출한 결과는 더 정확하게 컬러를 추출해내는 것을 볼 수 있었습니다.

# [고찰]

이번 과제는 영상에서 컬러를 추출하고, K-Mean Clustering, 그리고 최종적으로 두 기법을 모두 활용하여 컬러를 추출하는 과제를 진행하였습니다. 기존에 RGB만을 사용했던 방법과 다르게, 영상을 전환하고 이를 이용하여 색상을 추출하는 과정이 흥미로웠습니다. 또한 예시사진에 나오는 여러가지 과일들이 있는 이미지를 실습에 사용해보았는데, 단일 과일이 있는 image와 어떤 차이가 있는지, 그리고 각 결과에 대해 어떤 파라미터를 넣었을 때, 더 좋은 결과를 낼 수 있을지를 생각해볼 수 있는 과제였습니다. 각 image별로 결과를 확인하고, 분석하면서 image의 색상, 채도, 명도 등에 대해 더욱 직관적으로 이해할 수 있는 과제였습니다.