#include "kompute.hpp"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <EGL/egl.h>
#include <GL/gl.h>
#include <GL/glew.h>
#include <fcntl.h>
#include <gbm.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
}

KomputeKernel::KomputeKernel(const std::string& src) {
    shader = glCreateShader(GL_COMPUTE_SHADER);
    auto csrc = src.c_str();
    compile(&csrc);
}

KomputeKernel::KomputeKernel(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::stringstream ss;
    ss << file.rdbuf();
    std::string src = ss.str();
    auto csrc = src.c_str();
    compile(&csrc);
}

void KomputeKernel::compile(const char** src) {
    shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, src, nullptr);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetShaderInfoLog(shader, 512, NULL, buffer);
        std::cerr << buffer << std::endl;
        throw std::runtime_error("Compute shader compilation failed");
    }

    program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetProgramInfoLog(program, 512, NULL, buffer);
        std::cerr << buffer << std::endl;
        throw std::runtime_error("Shader program linking failed");
    }
}

KomputeKernel::~KomputeKernel() {
    glDeleteShader(shader);
    glDeleteProgram(program);
}

Kompute::Kompute(std::string dev) {
    dvr_fd = open(dev.c_str(), O_RDWR | O_CLOEXEC);
    if (dvr_fd < 0) {
        throw std::runtime_error("Cannot open " + dev);
    }

    gbm = gbm_create_device(dvr_fd);
    if (!gbm) {
        throw std::runtime_error("Cannot create GBM device");
    }

    egl_display = eglGetDisplay(gbm);
    if (egl_display == EGL_NO_DISPLAY) {
        throw std::runtime_error("Failed to get EGL display");
    }

    if (!eglInitialize(egl_display, NULL, NULL)) {
        throw std::runtime_error("Failed to initialize EGL");
    }

    EGLint configAttribs[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE };
    if (!eglChooseConfig(egl_display, configAttribs, &egl_config, 1, &num_configs) ||
        num_configs == 0)
    {
        throw std::runtime_error("Failed to choose EGL config");
    }

    eglBindAPI(EGL_OPENGL_API);
    egl_context = eglCreateContext(egl_display, egl_config, EGL_NO_CONTEXT, NULL);
    if (egl_context == EGL_NO_CONTEXT) {
        EGLint err = eglGetError();
        throw std::runtime_error("Failed to create EGL context, error: " + std::to_string(err));
    }

    if (!eglMakeCurrent(egl_display, NULL, NULL, egl_context)) {
        throw std::runtime_error("Failed to make EGL context current");
    }

    if (glewInit() != GLEW_OK) {
        throw std::runtime_error("Failed to initialize GLEW");
    }
};

Kompute::~Kompute() {
    eglDestroyContext(egl_display, egl_context);
    eglTerminate(egl_display);
    gbm_device_destroy(gbm);
    close(dvr_fd);
};

void Kompute::dispatch(
    KomputeKernel& kernel, const std::vector<Uniform> uniforms,
    const std::vector<std::shared_ptr<Buff>> buffers, int x, int y, int z
) {
    std::lock_guard l(mtx);
    int idx = 0;
    for (auto& buffer : buffers) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, idx++, buffer->ssbo);
    }
    glUseProgram(kernel.program);

    for (auto& uniform : uniforms) {
        if (std::holds_alternative<float>(uniform.val)) {
            glUniform1f(
                glGetUniformLocation(kernel.program, uniform.name.c_str()),
                std::get<float>(uniform.val)
            );
        } else if (std::holds_alternative<int>(uniform.val)) {
            glUniform1i(
                glGetUniformLocation(kernel.program, uniform.name.c_str()),
                std::get<int>(uniform.val)
            );
        } else if (std::holds_alternative<std::vector<float>>(uniform.val)) {
            auto& val = std::get<std::vector<float>>(uniform.val);
            switch (val.size()) {
                case 1:
                    glUniform1f(glGetUniformLocation(kernel.program, uniform.name.c_str()), val[0]);
                    break;
                case 2:
                    glUniform2f(
                        glGetUniformLocation(kernel.program, uniform.name.c_str()), val[0], val[1]
                    );
                    break;
                case 3:
                    glUniform3f(
                        glGetUniformLocation(kernel.program, uniform.name.c_str()), val[0], val[1],
                        val[2]
                    );
                    break;
                case 4:
                    glUniform4f(
                        glGetUniformLocation(kernel.program, uniform.name.c_str()), val[0], val[1],
                        val[2], val[3]
                    );
                    break;
                default:
                    throw std::runtime_error("Unsupported uniform vector size");
            }
        } else if (std::holds_alternative<std::vector<int>>(uniform.val)) {
            auto& val = std::get<std::vector<int>>(uniform.val);
            switch (val.size()) {
                case 1:
                    glUniform1i(glGetUniformLocation(kernel.program, uniform.name.c_str()), val[0]);
                    break;
                case 2:
                    glUniform2i(
                        glGetUniformLocation(kernel.program, uniform.name.c_str()), val[0], val[1]
                    );
                    break;
                case 3:
                    glUniform3i(
                        glGetUniformLocation(kernel.program, uniform.name.c_str()), val[0], val[1],
                        val[2]
                    );
                    break;
                case 4:
                    glUniform4i(
                        glGetUniformLocation(kernel.program, uniform.name.c_str()), val[0], val[1],
                        val[2], val[3]
                    );
                    break;
                default:
                    throw std::runtime_error("Unsupported uniform vector size");
            }
        } else {
            throw std::runtime_error("Unsupported uniform type");
        }
    }

    glDispatchCompute(x, y, z);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    GL_CHECK_ERROR();
}
