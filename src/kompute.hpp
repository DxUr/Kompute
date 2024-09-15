#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <EGL/egl.h>
#include <GL/glew.h>
}

#define GL_CHECK_ERROR()                                                  \
    do {                                                                  \
        GLenum err = glGetError();                                        \
        if (err != GL_NO_ERROR) {                                         \
            throw std::runtime_error("GL error: " + std::to_string(err)); \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

struct Buff {
    GLuint ssbo;
    size_t size = 0;
};

template <typename T>
struct StorageBuff : public Buff {
    StorageBuff() {
        glGenBuffers(1, &ssbo);
    }

    ~StorageBuff() {
        glDeleteBuffers(1, &ssbo);
    }

    StorageBuff(const StorageBuff&) = delete;

    void set_data(const std::span<const T> data, int usage = GL_STATIC_COPY) {
        this->size = data.size_bytes();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, size, data.data(), usage);
        GL_CHECK_ERROR();
    }

    void set_size(size_t size, int usage = GL_STATIC_COPY) {
        this->size = size;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, size, nullptr, GL_STATIC_COPY);
        GL_CHECK_ERROR();
    }

    std::vector<T> get_data() {
        std::vector<T> data(size / sizeof(T));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size, data.data());
        GL_CHECK_ERROR();
        return data;
    }
};

struct KomputeKernel {
    unsigned int shader;
    GLuint program;

    KomputeKernel(const std::string& src);
    KomputeKernel(const std::filesystem::path& path);
    ~KomputeKernel();
    KomputeKernel(const KomputeKernel&) = delete;

private:
    void compile(const char** src);
};

struct Uniform {
    std::string name;
    std::variant<float, int, std::vector<float>, std::vector<int>> val;
};

class Kompute {
    std::mutex mtx;
    int dvr_fd = -1;
    struct gbm_device* gbm = nullptr;
    EGLDisplay egl_display = EGL_NO_DISPLAY;
    EGLContext egl_context = EGL_NO_CONTEXT;
    EGLConfig egl_config;
    EGLint num_configs;

public:
    Kompute(std::string dev);
    void dispatch(
        KomputeKernel& kernel, const std::vector<Uniform> uniforms,
        const std::vector<std::shared_ptr<Buff>> buffers, int x, int y = 1, int z = 1
    );
    ~Kompute();
};