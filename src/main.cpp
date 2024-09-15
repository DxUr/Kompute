#include "kompute.hpp"

#include <opencv2/core/hal/interface.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <span>
#include <sstream>
#include <vector>

extern "C" {
#include <libswscale/swscale.h>
#include <libavcodec/codec.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/codec_par.h>
#include <libavcodec/packet.h>
#include <libavformat/avformat.h>
}

// Convert AVFrame (YUV) to OpenCV Mat (BGR)
cv::Mat avframe_to_cvmat(AVFrame* frame, SwsContext* sws_ctx) {
    int width = frame->width;
    int height = frame->height;
    cv::Mat bgr_image(cv::Size(width, height), CV_8UC3);  // 8-bit unsigned, 3 channels (BGR)

    // Define the destination buffer and linesize for BGR image
    uint8_t* dest[4] = { bgr_image.data, nullptr, nullptr, nullptr };
    int dest_linesize[4] = { static_cast<int>(bgr_image.step[0]), 0, 0, 0 };

    // Perform the color conversion (YUV to BGR)
    sws_scale(sws_ctx,
              frame->data, frame->linesize, 0, height,  // Source: YUV planes
              dest, dest_linesize);                     // Destination: BGR

    bgr_image.convertTo(bgr_image, CV_32FC3, 1.0 / 255.0);
    return bgr_image;
}

int main() {
    // read RTCP using ffmpeg
        const char* rtsp_url = "test0.mp4";
    AVFormatContext* fmt_ctx = avformat_alloc_context();
    if (avformat_open_input(&fmt_ctx, rtsp_url, 0, 0) < 0) {
        fprintf(stderr, "can't open %s\n", rtsp_url);
        exit(1);
    }

    if (avformat_find_stream_info(fmt_ctx, 0) < 0) {
        fprintf(stderr, "can't find stream info\n");
        exit(1);
    }

    int video_stream_index = -1;
    AVCodecParameters* codecpar = nullptr;

    // Find the video stream
    for (int i = 0; i < fmt_ctx->nb_streams; i++) {
        AVStream* stream = fmt_ctx->streams[i];
        if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            codecpar = stream->codecpar;
            break;
        }
    }

    if (video_stream_index == -1) {
        fprintf(stderr, "can't find a video stream\n");
        exit(1);
    }

    // Find the decoder for the video stream
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "failed to find codec\n");
        exit(1);
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        fprintf(stderr, "failed to allocate codec context\n");
        exit(1);
    }

    // Copy codec parameters from input stream to codec context
    if (avcodec_parameters_to_context(codec_ctx, codecpar) < 0) {
        fprintf(stderr, "failed to copy codec parameters\n");
        exit(1);
    }

    // Open the codec
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        fprintf(stderr, "failed to open codec\n");
        exit(1);
    }


    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    cv::Mat img;
    cv::Mat old_img;

    Kompute k("/dev/dri/renderD128");

    KomputeKernel kernel(std::filesystem::path("kernel.glsl"));

    auto in1 = std::make_shared<StorageBuff<float>>();
    auto in2 = std::make_shared<StorageBuff<float>>();
    auto out = std::make_shared<StorageBuff<float>>();



    while (av_read_frame(fmt_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_index) {
            // Send the packet to the decoder
            if (avcodec_send_packet(codec_ctx, packet) < 0) {
                fprintf(stderr, "error sending packet to decoder\n");
                continue;
            }

            // Receive all available frames from the decoder
            while (avcodec_receive_frame(codec_ctx, frame) >= 0) {
                // Process the decoded frame (e.g., display or save it)
                printf("Decoded frame, resolution: %dx%d\n", frame->width, frame->height);

                SwsContext* sws_ctx = sws_getContext(
                    frame->width, frame->height, (AVPixelFormat)frame->format,  // Source format (YUV)
                    frame->width, frame->height, AV_PIX_FMT_BGR24,              // Destination format (BGR)
                    SWS_BILINEAR, nullptr, nullptr, nullptr);
                
                if (!sws_ctx) {
                    fprintf(stderr, "Error: Could not initialize the sws context.\n");
                    return -1;
                }

                old_img = img;
                img = avframe_to_cvmat(frame, sws_ctx);
                if (old_img.empty() || img.empty()) {
                    continue;
                }

                in1->set_data({ img.ptr<float>(), img.total() * img.channels() }, GL_STATIC_READ);
                in2->set_data({ old_img.ptr<float>(), old_img.total() * old_img.channels() }, GL_STATIC_READ);
                out->set_size(img.total() * sizeof(float), GL_STATIC_DRAW);

                std::vector<int> dims = { img.cols, img.rows, img.channels() };

                // calc kernel time
                k.dispatch(
                    kernel,
                    {
                        { "dims", dims },
                    },
                    { in1, in2, out }, img.cols, img.rows
                );
                
                auto tp = std::chrono::system_clock::now();
                auto res = out->get_data();
                
                auto now = std::chrono::system_clock::now();
                std::cout << "Kernel time: " << (std::chrono::duration_cast<std::chrono::milliseconds>(now - tp)).count() << " ms" << std::endl;
                tp = now;
                
                cv::Mat res_img(img.rows, img.cols, CV_32FC1, res.data());
                // Convert the image to grayscale
                res_img.convertTo(res_img, CV_8UC1);

                // Find contours in the binary mask
                std::vector<std::vector<cv::Point>> contours;
                std::vector<cv::Vec4i> hierarchy;
                cv::findContours(res_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                // Draw bounding boxes around the detected contours
                cv::Mat outputImage = img.clone();
                for (const auto& contour : contours) {
                    cv::Rect boundingBox = cv::boundingRect(contour);
                    cv::rectangle(outputImage, boundingBox, cv::Scalar(0, 255, 0), 2); // Draw bounding box in green
                }

                now = std::chrono::system_clock::now();
                std::cout << "Find contours time: " << (std::chrono::duration_cast<std::chrono::milliseconds>(now - tp)).count() << " ms" << std::endl;
                // Show the result
                cv::imshow("Bounding Boxes", outputImage);
                cv::waitKey(1);

            }
        }

        av_packet_unref(packet); // Clean up the packet for the next iteration
    }

    // Clean up
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);

    av_packet_free(&packet);
    avformat_close_input(&fmt_ctx);

    return 0;
}
