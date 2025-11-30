#include "header.hpp"
#include <filesystem>
#include <fstream> // Necesario para leer el vocabulario
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <algorithm>

// Función para leer el archivo de vocabulario línea a línea
std::vector<std::string> read_vocabulary(const std::string& filename) {
    std::vector<std::string> vocabulary;
    std::ifstream file(filename);
    if (!file.is_open()) {
        spdlog::error("No se pudo abrir el vocabulario: {}", filename);
        return vocabulary;
    }
    std::string str;
    while (std::getline(file, str)) {
        vocabulary.push_back(str);
    }
    return vocabulary;
}

// Levenshtein para filtrar duplicados (igual que antes)
int levenshtein_distance(const std::string &s1, const std::string &s2) {
    const std::size_t m = s1.size();
    const std::size_t n = s2.size();
    if (m == 0) return n;
    if (n == 0) return m;
    std::vector<std::vector<int>> matrix(m + 1, std::vector<int>(n + 1));
    for (std::size_t i = 0; i <= m; ++i) matrix[i][0] = i;
    for (std::size_t j = 0; j <= n; ++j) matrix[0][j] = j;
    for (std::size_t i = 1; i <= m; ++i) {
        for (std::size_t j = 1; j <= n; ++j) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            matrix[i][j] = std::min({ matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost });
        }
    }
    return matrix[m][n];
}

double calculate_similarity(const std::string &s1, const std::string &s2) {
    if (s1.empty() && s2.empty()) return 1.0;
    int max_len = std::max(s1.length(), s2.length());
    if (max_len == 0) return 0.0;
    return 1.0 - (static_cast<double>(levenshtein_distance(s1, s2)) / max_len);
}

// --- FUNCIÓN PRINCIPAL MODIFICADA ---
void align_sign_text_sequences(
    const std::string &video_path,
    const std::string &text_detector_path, // Modelo EAST
    const std::string &text_recognitor_path, // AHORA ES EL MODELO .ONNX (DenseNet)
    float roi_ratio
) {
    auto console = spdlog::get("console");
    if (!console) console = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(console);
    spdlog::set_level(spdlog::level::info);

    // Asumimos que el archivo de vocabulario está en la misma carpeta que el modelo onnx
    // o pásalo como argumento extra si prefieres. Aquí lo deducimos:
    std::filesystem::path model_path(text_recognitor_path);
    std::string vocab_path = model_path.parent_path() / "alphabet_94.txt"; // Asegúrate de tener este archivo

    if (!std::filesystem::exists(video_path)) { spdlog::error("Video no existe"); return; }
    if (!std::filesystem::exists(text_detector_path)) { spdlog::error("Detector no existe"); return; }
    if (!std::filesystem::exists(text_recognitor_path)) { spdlog::error("Reconocedor (ONNX) no existe"); return; }
    if (!std::filesystem::exists(vocab_path)) { spdlog::error("Vocabulario no encontrado: {}", vocab_path); return; }

    // 1. Cargar Detector (EAST)
    cv::dnn::TextDetectionModel_EAST detector(text_detector_path);
    detector.setConfidenceThreshold(0.5);
    detector.setNMSThreshold(0.4);
    detector.setInputParams(1.0, cv::Size(320, 320), cv::Scalar(123.68, 116.78, 103.94), true);

    // 2. Cargar Reconocedor (DenseNet / CRNN) - ¡Adiós Tesseract!
    cv::dnn::TextRecognitionModel recognizer(text_recognitor_path);

    // Configuración para decodificación CTC
    recognizer.setDecodeType("CTC-greedy");

    // Cargar y establecer vocabulario
    std::vector<std::string> vocabulary = read_vocabulary(vocab_path);
    recognizer.setVocabulary(vocabulary);

    // IMPORTANTE: Parámetros específicos para DenseNet/CRNN
    // Escala 1/127.5, tamaño (100, 32), media (127.5) -> Normalización a [-1, 1]
    // Debes convertir a gris primero.
    recognizer.setInputParams(1.0/127.5, cv::Size(100, 32), cv::Scalar(127.5), true);

    cv::VideoCapture cap(video_path);
    cv::Mat curr, curr_roi_mat;
    int frame_idx = 0;
    std::string last_valid_text = "";
    int frames_since_change = 0;

    spdlog::info("Iniciando con Deep Learning OCR...");

    while (cap.read(curr)) {
        if (curr.empty()) break;

        // ROI Inferior
        int h = curr.rows;
        int roi_h = static_cast<int>(h * roi_ratio);
        cv::Rect dynamic_roi(0, h - roi_h, curr.cols, roi_h);
        curr_roi_mat = curr(dynamic_roi);

        // Detectar
        std::vector<std::vector<cv::Point>> det_results;
        detector.detect(curr_roi_mat, det_results);

        if (!det_results.empty()) {
            cv::Rect composed_bb;
            for(const auto& c : det_results) composed_bb |= cv::boundingRect(c);

            // Padding
            composed_bb.x = std::max(0, composed_bb.x - 10);
            composed_bb.y = std::max(0, composed_bb.y - 5);
            composed_bb.width = std::min(curr_roi_mat.cols - composed_bb.x, composed_bb.width + 20);
            composed_bb.height = std::min(curr_roi_mat.rows - composed_bb.y, composed_bb.height + 10);

            if (composed_bb.area() > 0) {
                cv::Mat text_patch = curr_roi_mat(composed_bb);

                // Deep Learning OCR funciona mejor en Escala de Grises
                cv::Mat gray_patch;
                if (text_patch.channels() == 3) {
                    cv::cvtColor(text_patch, gray_patch, cv::COLOR_BGR2GRAY);
                } else {
                    gray_patch = text_patch;
                }

                // --- INFERENCIA OCR (Aquí ocurre la magia) ---
                std::string output_text;
                try {
                    output_text = recognizer.recognize(gray_patch);
                } catch (const cv::Exception& e) {
                    spdlog::warn("Error en inferencia OCR: {}", e.what());
                }

                if (!output_text.empty()) {
                    if (calculate_similarity(last_valid_text, output_text) < 0.70) {
                        spdlog::info("Frame {}: '{}'", frame_idx, output_text);
                        last_valid_text = output_text;
                        frames_since_change = 0;
                    } else {
                        frames_since_change++;
                    }
                }
            }
        } else {
            if (frames_since_change > 10) last_valid_text = "";
            frames_since_change++;
        }
        frame_idx++;
    }
}
