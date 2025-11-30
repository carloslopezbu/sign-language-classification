#ifndef HEADER
#define HEADER
#include <opencv2/core/types.hpp>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>



void align_sign_text_sequences(
    const std::string &video_path,
    const std::string &text_detector_path,
    const std::string &text_recognitor_path,
    float roi_ratio // NUEVO: Porcentaje de pantalla (0.0 a 1.0)
);
#endif
