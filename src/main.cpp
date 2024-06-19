#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <time.h>
#include <vector>

using namespace cv;
using namespace std;

enum SeamDirection { VERTICAL, HORIZONTAL };

bool demo;
bool debug;

Mat energyGen(Mat &image) {
  Mat image_blur, image_gray;
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  Mat grad, energy_image;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  GaussianBlur(image, image_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);
  cvtColor(image_blur, image_gray, COLOR_BGR2GRAY);

  Scharr(image_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
  Scharr(image_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);

  convertScaleAbs(grad_x, abs_grad_x);
  convertScaleAbs(grad_y, abs_grad_y);

  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

  grad.convertTo(energy_image, CV_64F, 1.0 / 255.0);

  if (demo) {
    namedWindow("Energy Image", WINDOW_AUTOSIZE);
    imshow("Energy Image", energy_image);
  }

  return energy_image;
}

Mat energyMapGen(Mat &energy_image, SeamDirection seam_direction) {
  int rowsize = energy_image.rows;
  int colsize = energy_image.cols;

  Mat enrgMap = Mat(rowsize, colsize, CV_64F, double(0));

  if (seam_direction == VERTICAL)
    energy_image.row(0).copyTo(enrgMap.row(0));
  else if (seam_direction == HORIZONTAL)
    energy_image.col(0).copyTo(enrgMap.col(0));

  if (seam_direction == VERTICAL) {
    for (int row = 1; row < rowsize; row++) {
      for (int col = 0; col < colsize; col++) {
        double a = (col > 0) ? enrgMap.at<double>(row - 1, col - 1) : DBL_MAX;
        double b = enrgMap.at<double>(row - 1, col);
        double c = (col < colsize - 1) ? enrgMap.at<double>(row - 1, col + 1)
                                       : DBL_MAX;

        enrgMap.at<double>(row, col) =
            energy_image.at<double>(row, col) + min(a, min(b, c));
      }
    }
  } else if (seam_direction == HORIZONTAL) {
    for (int col = 1; col < colsize; col++) {
      for (int row = 0; row < rowsize; row++) {
        double a = (row > 0) ? enrgMap.at<double>(row - 1, col - 1) : DBL_MAX;
        double b = enrgMap.at<double>(row, col - 1);
        double c = (row < rowsize - 1) ? enrgMap.at<double>(row + 1, col - 1)
                                       : DBL_MAX;

        enrgMap.at<double>(row, col) =
            energy_image.at<double>(row, col) + min(a, min(b, c));
      }
    }
  }

  if (demo) {
    Mat color_enrgMap;
    double Cmin, Cmax;
    minMaxLoc(enrgMap, &Cmin, &Cmax);
    float scale = 255.0 / (Cmax - Cmin);
    enrgMap.convertTo(color_enrgMap, CV_8UC1, scale);
    applyColorMap(color_enrgMap, color_enrgMap, COLORMAP_JET);

    namedWindow("Cumulative Energy Map", WINDOW_AUTOSIZE);
    imshow("Cumulative Energy Map", color_enrgMap);
  }

  return enrgMap;
}

vector<int> findOptimalSeam(Mat &enrgMap, SeamDirection seam_direction) {
  vector<int> path;

  int rowsize = enrgMap.rows;
  int colsize = enrgMap.cols;

  if (seam_direction == VERTICAL) {
    Mat row = enrgMap.row(rowsize - 1);
    Point min_pt;
    minMaxLoc(row, nullptr, nullptr, &min_pt, nullptr);

    path.resize(rowsize);
    int min_index = min_pt.x;
    path[rowsize - 1] = min_index;

    for (int i = rowsize - 2; i >= 0; i--) {
      double a =
          (min_index > 0) ? enrgMap.at<double>(i, min_index - 1) : DBL_MAX;
      double b = enrgMap.at<double>(i, min_index);
      double c = (min_index < colsize - 1)
                     ? enrgMap.at<double>(i, min_index + 1)
                     : DBL_MAX;

      if (min(a, b) > c)
        min_index += 1;
      else if (min(a, c) > b)
        min_index += 0;
      else if (min(b, c) > a)
        min_index -= 1;

      min_index = min(max(min_index, 0), colsize - 1);
      path[i] = min_index;
    }
  } else if (seam_direction == HORIZONTAL) {
    Mat col = enrgMap.col(colsize - 1);
    Point min_pt;
    minMaxLoc(col, nullptr, nullptr, &min_pt, nullptr);

    path.resize(colsize);
    int min_index = min_pt.y;
    path[colsize - 1] = min_index;

    for (int i = colsize - 2; i >= 0; i--) {
      double a =
          (min_index > 0) ? enrgMap.at<double>(min_index - 1, i) : DBL_MAX;
      double b = enrgMap.at<double>(min_index, i);
      double c = (min_index < rowsize - 1)
                     ? enrgMap.at<double>(min_index + 1, i)
                     : DBL_MAX;

      if (min(a, b) > c)
        min_index += 1;
      else if (min(a, c) > b)
        min_index += 0;
      else if (min(b, c) > a)
        min_index -= 1;

      min_index = min(max(min_index, 0), rowsize - 1);
      path[i] = min_index;
    }
  }

  return path;
}

void reduce(Mat &image, vector<int> path, SeamDirection seam_direction) {
  int rowsize = image.rows;
  int colsize = image.cols;

  Mat dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));

  if (seam_direction == VERTICAL) {
    for (int i = 0; i < rowsize; i++) {
      Mat new_row;
      Mat lower = image.rowRange(i, i + 1).colRange(0, path[i]);
      Mat upper = image.rowRange(i, i + 1).colRange(path[i] + 1, colsize);

      if (!lower.empty() && !upper.empty()) {
        hconcat(lower, upper, new_row);
        hconcat(new_row, dummy, new_row);
      } else {
        if (lower.empty()) {
          hconcat(upper, dummy, new_row);
        } else if (upper.empty()) {
          hconcat(lower, dummy, new_row);
        }
      }
      new_row.copyTo(image.row(i));
    }
    image = image.colRange(0, colsize - 1);
  } else if (seam_direction == HORIZONTAL) {
    for (int i = 0; i < colsize; i++) {
      Mat new_col;
      Mat lower = image.colRange(i, i + 1).rowRange(0, path[i]);
      Mat upper = image.colRange(i, i + 1).rowRange(path[i] + 1, rowsize);

      if (!lower.empty() && !upper.empty()) {
        vconcat(lower, upper, new_col);
        vconcat(new_col, dummy, new_col);
      } else {
        if (lower.empty()) {
          vconcat(upper, dummy, new_col);
        } else if (upper.empty()) {
          vconcat(lower, dummy, new_col);
        }
      }
      new_col.copyTo(image.col(i));
    }
    image = image.rowRange(0, rowsize - 1);
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    cout << "Usage: ./seam_carving <image_path> <num_iterations> <direction "
            "(H/V)> [demo (optional)] [debug (optional)]"
         << endl;
    return -1;
  }

  string image_path = argv[1];
  int iterations = stoi(argv[2]);
  string direction = argv[3];
  debug = (argc > 5 && string(argv[5]) == "debug");
  demo = (argc > 4 && string(argv[4]) == "demo");

  SeamDirection seam_direction;
  if (direction == "V")
    seam_direction = VERTICAL;
  else if (direction == "H")
    seam_direction = HORIZONTAL;
  else {
    cout << "Invalid direction. Use 'H' for horizontal or 'V' for vertical."
         << endl;
    return -1;
  }

  Mat image = imread(image_path);
  Mat orgImage = image.clone();
  if (!image.data) {
    cout << "Could not open or find the image." << endl;
    return -1;
  }

  if (seam_direction == VERTICAL && iterations >= image.cols) {
    cout << "Number of iterations is greater than or equal to the number of "
            "columns in the image."
         << endl;
    return -1;
  } else if (seam_direction == HORIZONTAL && iterations >= image.rows) {
    cout << "Number of iterations is greater than or equal to the number of "
            "rows in the image."
         << endl;
    return -1;
  }

  if (debug) {
    cout << "Starting seam carving with " << iterations
         << " iterations in direction "
         << (seam_direction == VERTICAL ? "VERTICAL" : "HORIZONTAL") << endl;
  }

  for (int i = 0; i < iterations; i++) {
    if (debug)
      cout << "Iteration " << i + 1 << " started." << endl;

    Mat energy_image = energyGen(image);
    Mat enrgMap = energyMapGen(energy_image, seam_direction);
    vector<int> path = findOptimalSeam(enrgMap, seam_direction);

    if (debug) {
      cout << "Energy image and energy map generated." << endl;
      cout << "Optimal seam found: ";
      for (int val : path) {
        cout << val << " ";
      }
      cout << endl;
    }

    reduce(image, path, seam_direction);

    if (debug) {
      cout << "Iteration " << i + 1 << " completed." << endl;
    }
  }

  imwrite("output.jpg", image);
  cout << "Seam carving completed. Result saved as 'output.jpg'." << endl;

  if (demo) {
    namedWindow("Output Image", WINDOW_AUTOSIZE);
    imshow("Output Image", image);
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", orgImage);
    waitKey(0);
  }

  return 0;
}
