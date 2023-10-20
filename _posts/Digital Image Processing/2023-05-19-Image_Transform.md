---
title: Image Transformation
date: 2023-05-19 00:00:00 +09:00
categories: [Digital Image Processing]
tags:
  [
    Computer Vision,
    Digital Image Processing,
    Image Transformation
  ]
pin: true
---

# 1. getRotationMatrix()과 동일한getMyRotationMatrix()함수를 직접 구현하고 두 결과가 동일한지 검증하라
❑ Scale 변화는 구현하지 않아도 됨
❑ 45도 변화 결과가 동일한지 비교하면 됨

## 1-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
using namespace cv;
using namespace std;

Mat getMyRotationMatrix(Point center, double angle, double scale) {
	double alpha = scale * cos(angle / 180 * CV_PI);
	double beta = scale * sin(angle / 180 * CV_PI);

	Mat matrix = (Mat_<double>(2, 3) <<
		alpha, beta, (1 - alpha) * center.x - beta * center.y,
		-beta, alpha, beta * center.x + (1 - alpha) * center.y);

	return matrix;
}

Mat cvRotation(Mat src) {
	Mat dst, matrix;

	Point center = Point(src.cols / 2, src.rows / 2);

	matrix = getRotationMatrix2D(center, 45.0, 1.0);

	warpAffine(src, dst, matrix, src.size());

	return dst;
}

Mat myRotation(Mat src) {
	Mat dst, matrix;

	Point center = Point(src.cols / 2, src.rows / 2);

	matrix = getMyRotationMatrix(center, 45.0, 1.0);

	warpAffine(src, dst, matrix, src.size());

	return dst;
}

void main() {
	Mat src = imread("D:/study/디영처/12주차/12주차 실습 자료/Lenna.png", 1);
	imshow("src", src);
	Mat myRot, cvRot;

	cvRot = cvRotation(src);
	myRot = myRotation(src);

	uchar* myData = myRot.data;
	uchar* cvData = cvRot.data;
	for (int i = 0; i < src.cols; i++) {
		for (int j = 0; j < src.rows; j++) {
			for (int k = 0; k < 3; k++) {
				if (myData[j * src.cols + i + k] != cvData[j * src.cols + i + k]) {
					cout << "두 이미지가 다릅니다." << endl;
					break;
				}
				if (i == src.cols - 1 && j == src.rows - 1 && k == 2)
					cout << "두 이미지가 같습니다." << endl;
			}
		}

	}
	imshow("getRotationMatrix", cvRot);
	imshow("getMyRotationMatrix", myRot);

	waitKey(0);
	destroyAllWindows();
}
```

## 1-2) 실험결과

### [원본 사진]

![Untitled](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/73dda88e-50f8-482f-a7ac-d4661735d256)

### [좌: getRotationMatrix()결과, 우: getMyRotationMatrix()결과]

![Untitled 1](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/cb6ca827-9cb1-43f1-a28b-f8a246d3a67b)

![Untitled 2](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/86858534-a412-433a-83ad-749767286643)

## 1-3)구현 과정

- `**Mat getMyRotationMatrix(Point center, double angle, double scale)` :**
위 getMyRotationMarix는 영상의 Rotation을 할 수 있도록 Matrix를 구현한 함수입니다. 함수를 구성할 때는 아래 강의노트의 식을 참고하였습니다. 따라서 입력 변수를 영상의 중심점, Rotation할 각도, scale 3가지로 받았고, 이는 기존의 Rotation Matrix와 동일한 연산을 진행합니다. scale은 1.0로 입력하였기 때문에 scale변화는 없습니다.

![Untitled 3](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/34b693ad-52ea-4786-b8df-1ddeaeeb2b0c)

- **`Mat cvRotation(Mat src)`** :cvRotation함수는 Open CV함수를 사용하여 Rotation을 진행하는 함수입니다. 실제로 로테이션을 image에 적용하는 함수이고, Open CV에 구현되어 있는 getRotationMatrix2D()함수를 사용하였습니다.
- **`Mat myRotation(Mat src)`** :myRotation함수는 위에서 정의한 getMyRotationMatrix를 통해 실제로 image에 Rotation을 적용하는 함수입니다.
- 위 RotationMatrix함수를 통해 image에 Rotation을 진행한 image는 육안상으로 보기에 동일했습니다. 하지만 완벽하게 일치하는지 확인하기 위해 for문을 통해 image 내의 모든 pixel들이 동일한지 비교하는 코드를 main함수에 작성하였고, 그 결과 동일하다는 것을 확인할 수 있었습니다.

## 1-4)결과 분석

1번은 getRotationMatrix()과 동일한 getMyRotationMatrix() 함수를 구현하는 과제였습니다. getRotationMatrix는 Open CV에 구현되어 있는 함수로, Rotation 연산을 할 수 있는 Matrix를 center, angle, scale에 따라서 만들어내는 함수입니다. 이때 getRotationMatrix와 동일한 함수를 구현하기 위해서는 해당 함수 내에서 구현되는 연산을 그대로 진행해야 합니다. 따라서 같은 Matrix를 만들어 낼 수 있도록 함수 코드를 작성하였습니다. Matrix에 대한 정보는 강의 노트에 있는 식을 바탕으로 만들었습니다.

그 결과 우선 Rotation된 image를 육안으로 보았을 때에는 동일한 것으로 보였습니다. 하지만, 육안보다 더 정확하게 두 image가 같은지 확인하기 위해 pixel level로 두 image를 비교할 수 있도록 코딩하였고, 하나의 pixel이라도 같지 않다면 오류가 발생하도록 하였습니다. 하지만, 모든 pixel이 같다는 결과를 확인하였고, 결과적으로 getMyRotationMatrix()함수가 getRotationMatrix()함수와 같은 동작을 한다는 것을 확인할 수 있었습니다.

# 2. card_per.png영상을getPerspectiveTransform() 함수를 이용해 카드의 면이 시선정면을 향하도록 정렬시켜라.

## **[HarrisCorner Detector를 사용하여 네 꼭지점을 자동으로탐색하도록만들었습니다.]**

## 2-1) 코드

```cpp
#include <iostream>
#include<sstream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
using namespace cv;
using namespace std;

void cvHarrisCorner(Mat img, Point2f srcQuad[4]) {
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	threshold(gray, gray, 10, 255, THRESH_BINARY);

	Mat harr;
	cornerHarris(gray, harr, 2, 3, 0.03, BORDER_DEFAULT);
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);

	Point2f search_srcQuad[4];
	int num = 0;
	int thresh = 165;
	Mat result = img.clone();
	for (int y = 0; y < harr.rows; y += 1) {
		for (int x = 0; x < harr.cols; x += 1) {
			if ((int)harr.at<float>(y, x) > thresh)
			{
				circle(result, Point(x, y), 7, Scalar(255, 255, 0), -1, 4, 0);
				search_srcQuad[num] = Point2f(x, y);
				num++;
			}
		}
	}
	int num2 = 0;
	int num3 = 0;
	int t1[2];
	int t2[2];
	for (int i = 0; i < 4; i++) {
		if (search_srcQuad[i].y < img.cols / 2) {
			t1[num2] = i;
			num2++;
		}
		else {
			t2[num3] = i;
			num3++;
		}
	}
	if (search_srcQuad[t1[0]].x > search_srcQuad[t1[1]].x) {
		srcQuad[1] = search_srcQuad[t1[0]];
		srcQuad[0] = search_srcQuad[t1[1]];
	}
	else {
		srcQuad[1] = search_srcQuad[t1[1]];
		srcQuad[0] = search_srcQuad[t1[0]];
	}
	if (search_srcQuad[t2[0]].x > search_srcQuad[t2[1]].x) {
		srcQuad[3] = search_srcQuad[t2[0]];
		srcQuad[2] = search_srcQuad[t2[1]];
	}
	else {
		srcQuad[3] = search_srcQuad[t2[1]];
		srcQuad[2] = search_srcQuad[t2[0]];
	}
	imshow("Harris image", harr_abs);
	imshow("Target image", result);
}

Mat cvPerspective(Mat src) {
	Mat dst, matrix;

	Point2f srcQuad[4];
	cvHarrisCorner(src, srcQuad);

	Point2f dstQuad[4];
	dstQuad[0] = Point2f(40.f,120.f);
	dstQuad[1] = Point2f(src.cols - 40.f, 120.f);
	dstQuad[2] = Point2f(40.f, 360.f);
	dstQuad[3] = Point2f(src.cols - 40.f, 360.f);

	matrix = getPerspectiveTransform(srcQuad, dstQuad);
	warpPerspective(src, dst, matrix, src.size());

	return dst;
}

void main() {
	Mat src = imread("D:/study/디영처/12주차/12주차 실습 자료/card_per.png", 1);
	Mat dst;
	imshow("src", src);

	dst = cvPerspective(src);

	imshow("result", dst);

	waitKey(0);
	destroyAllWindows();
}
```

## 2-2) 실험 결과

### [Harris Corner Detector결과]

![Untitled 4](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/2ca57267-e1de-4ed3-8b59-1da88eb054af)

### [최종 Perspective결과]

![Untitled 5](https://github.com/gihuni99/gihuni99.github.io/assets/90080065/4b72579f-26b1-42a2-a8fd-97bdbbb6e2da)

## 2-3) 구현 과정

- **`void cvHarrisCorner(Mat img, Point2f srcQuad[4])`** :
cvHarrisCorner함수에서 corner detect를 하는 부분은 threshold와 같은 파라미터 값들을 제외하고 지난주 실습에서와 모두 동일합니다. 즉 카드의 모서리 4부분을 detect합니다. 하지만, 이 꼭지점들을 자동으로 찾은 이후에 각 좌표값을 Perspective를 위해 순서에 맞도록 저장할 필요가 있습니다. srcQuad[]에 각 좌표를 순서대로 저장을 해야 하는데, HarrisCorner Detector는 순서대로 Corner를 Detect하지는 않습니다. 따라서 img.cols/2보다 작은 값을 y값으로 갖는 corner는 위쪽 2개의 corner, 큰 값을 갖는 corner는 아래쪽 corner인 점을 사용하였고, 위쪽 corner 두 값 중 x값이 더 큰 corner가 srcQuad[1], 작은 corner가 srcQuad[0]에 저장하는 방식으로 아래쪽 corner도 srcQuad[]저장하였습니다.
- **`Mat cvPerspective(Mat src)`** : 실제로 Perspective를 적용하는 함수입니다. 여기서 원본사진에서의 Corner좌표인 srcQuad[]값은 cvHarrisCorner함수를 통해 얻고, dstQuad[]값에는 카드가 기울어진 것이 아닌 원래의 형태를 가질 수 있도록 좌표를 지정해주었습니다. 그 후 srcQuad[]와 dstQuad[]를 getPerspectiveTransform()에 전달하여 perspective를 수행할 수 있는 matrix를 만들어 연산을 진행할 수 있도록 하였습니다.

## 2-4) 결과 분석

2번은 Perspective Transformation을 적용하여 카드 이미지의 시점을 변경하는 과제였습니다. 우선 이미지에서 카드의 Corner를 Detecting하는 과정은 지난 주 실습에서 사용했던 HarrisCorner Detector를 사용하여 자동으로 네 꼭지점을 찾을 수 있도록 코딩하였습니다. 하지만 getPerspectiveTransform함수를 사용하여 시점을 변경하기 위해서는 image의 perspective transformation 이전과 이후의 네 꼭지점 정보를 순서에 맞도록 입력해주어야 합니다. 하지만 지난 주 과제에서 사용했던 HarrisCorner Detector는 Corner정보를 취득할 수 있지만, 꼭지점의 순서에 맞도록 취득하는 것은 불가능했습니다. 따라서 cvHarrisCorner함수에 각 꼭지점들을 순서에 맞도록 정렬할 필요가 있었습니다. 따라서 image의 중심을 기준으로 x,y좌표가 있다고 할 때, 1사분면의 점은 srcQuad[1], 2사분면의 점은 srcQuad[0], 3사분면의 점은 srcQuad[2], 4사분면의 점은 srcQuad[3]에 각각 넣을 수 있도록 코드를 추가하였습니다. 따라서 원본 이미지에서의 네 꼭지점을 정상적으로 Detect하였습니다. 이후, dstQuad[]는 각 꼭지점이 카드의 면이 시선 정면을 향할 수 있도록 각각 좌표를 지정해주었습니다.

이렇게 얻은 꼭지점 정보를 getPerspectiveTransform()에 전달하여 Perspective Transform연산을 할 수 있는 Matrix를 생성하였고, 결과적으로 카드의 면이 시선 정면을 향할 수 있도록 코드가 동작하였습니다. 그 결과는 위에서 확인할 수 있습니다.

# [고찰]

이번 실습는 이론 시간에 배운 2D transformation을 직접 코드를 구현하여 결과를 확인해볼 수 있는 실습이었습니다. Transformation Matrix를 통해 연산을 진행하는데, 이론적인 식을 통해 어떤 방식으로 2D image가 변환되는지는 이론 시간에 충분히 이해하였습니다. 하지만, 실질적으로 어떻게 동작을 하는지 조금은 이해되지 않았는데, Open CV에 구현된 함수 뿐만 아니라, 동일한 동작을 하는 함수를 작성하고, 코드를 동작시키면서 Transformation에 대해서 더 직관적으로 이해할 수 있었습니다.