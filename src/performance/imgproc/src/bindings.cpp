#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <string>

namespace py = pybind11;

//Ejemplo de función que procesa una imagen
cv::Mat process_image(py::array_t<uint8_t> input) {
    // Tu código de procesamiento aquí
    py::buffer_info buf = input.request();

    cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC3, (uint8_t*)buf.ptr);

    // Procesar imagen con OpenCV
    cv::Mat result;
    cv::cvtColor(img, result, cv::COLOR_BGR2GRAY);

    return result;
}

void hello_world(const std::string msg) {
    printf("%s", msg.c_str());
}



PYBIND11_MODULE(imgproc, m) {
    m.doc() = "Módulo de procesamiento de imágenes con C++ y OpenCV";

    m.def("hello_world", &hello_world,
          "Procesa una imagen usando OpenCV",
          py::arg("msg"));
}
