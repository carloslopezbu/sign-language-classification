#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // <--- IMPORTANTE: Necesario para convertir std::string
#include <opencv2/core/types.hpp>
#include "header.hpp"

namespace py = pybind11;

PYBIND11_MODULE(videoproc, m) {
    m.doc() = "Módulo de procesamiento de imágenes con C++ y OpenCV";

    // 1. Binding de la clase cv::Rect
    // Esto permite crear objetos videoproc.Rect(x, y, w, h) en Python
    py::class_<cv::Rect>(m, "Rect")
        .def(py::init<>()) // Constructor vacío
        .def(py::init<int, int, int, int>()) // Constructor con parámetros
        .def("area", &cv::Rect::area)
        .def_readwrite("x", &cv::Rect::x)
        .def_readwrite("y", &cv::Rect::y)
        .def_readwrite("width", &cv::Rect::width)
        .def_readwrite("height", &cv::Rect::height) // Corregido typo: heigth -> height
        .def("__repr__", [](const cv::Rect &r) {
            return "<videoproc.Rect (x=" + std::to_string(r.x) +
                   ", y=" + std::to_string(r.y) +
                   ", w=" + std::to_string(r.width) +
                   ", h=" + std::to_string(r.height) + ")>";
        });



    m.def("align_sign_text_sequences_v2",
              &align_sign_text_sequences,
              "Extrae texto del porcentaje inferior de la pantalla",
              py::arg("video_path"),
              py::arg("text_detector_path"),
              py::arg("text_recognitor_path"),
              py::arg("roi_ratio") // <--- CAMBIO AQUÍ: Ahora espera un float
        );
}
