#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdio>
#include <algorithm>
#include "whisper.h"

namespace fs = std::filesystem;

// Декодирование .ogg/.opus/любого аудио напрямую в вектор float (16 кГц mono) через ffmpeg
bool load_audio_via_ffmpeg(const std::string& filepath, std::vector<float>& pcmf32) {
    std::string cmd = "ffmpeg -nostdin -threads 0 -i \"" + filepath + "\" -f f32le -ac 1 -ar 16000 - 2>nul";
    
    FILE* pipe = _popen(cmd.c_str(), "rb");
    if (!pipe) return false;

    pcmf32.clear();
    char buffer[4096];
    size_t bytes_read;

    while ((bytes_read = fread(buffer, 1, sizeof(buffer), pipe)) > 0) {
        float* samples = reinterpret_cast<float*>(buffer);
        size_t count = bytes_read / sizeof(float);
        pcmf32.insert(pcmf32.end(), samples, samples + count);
    }

    _pclose(pipe);
    return !pcmf32.empty();
}

int main() {
    const std::string model_path = "ggml-small.bin";
    const fs::path input_dir = R"(C:\Users\MONHTEPO\Downloads\Telegram Desktop\ChatExport_2026-10-09 (1)\voice_messages)";
    const fs::path output_file = input_dir / "transcription_result.txt";

    if (!fs::exists(input_dir)) {
        std::cerr << "Папка не найдена: " << input_dir << "\n";
        return 1;
    }

    // 1. Инициализация модели Whisper с поддержкой CUDA
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true; // Задействует RTX 2060 (требуется сборка с GGML_CUDA=1)

    std::cout << "Загрузка модели " << model_path << "...\n";
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Ошибка загрузки файла модели " << model_path << "\n";
        return 1;
    }

    // 2. Сбор всех файлов голосовых сообщений
    std::vector<fs::directory_entry> files;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".ogg" || ext == ".oga" || ext == ".opus" || ext == ".mp3" || ext == ".wav") {
                files.push_back(entry);
            }
        }
    }

    // Сортировка по времени изменения (хронологический порядок)
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
        return fs::last_write_time(a) < fs::last_write_time(b);
    });

    std::cout << "Найдено файлов для обработки: " << files.size() << "\n\n";

    // 3. Настройка параметров распознавания
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = "ru";
    wparams.n_threads = 4;
    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;

    std::ofstream out(output_file, std::ios::app);
    std::vector<float> pcmf32;

    size_t counter = 0;
    for (const auto& file : files) {
        counter++;
        std::string filename = file.path().filename().string();
        std::cout << "[" << counter << "/" << files.size() << "] " << filename << std::flush;

        if (!load_audio_via_ffmpeg(file.path().string(), pcmf32)) {
            out << "[" << filename << "]\n[Ошибка чтения аудио / пустой файл]\n\n";
            std::cout << " -> Ошибка\n";
            continue;
        }

        if (whisper_full(ctx, wparams, pcmf32.data(), static_cast<int>(pcmf32.size())) != 0) {
            out << "[" << filename << "]\n[Ошибка распознавания]\n\n";
            std::cout << " -> Сбой инференса\n";
            continue;
        }

        std::string text = "";
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            text += whisper_full_get_segment_text(ctx, i);
        }

        if (text.empty()) {
            text = "[Тишина или неразборчиво]";
        }

        out << "[" << filename << "]\n" << text << "\n\n";
        out.flush();
        std::cout << " -> OK\n";
    }

    whisper_free(ctx);
    std::cout << "\nГотово! Текст сохранен в: " << output_file << "\n";
    return 0;
}