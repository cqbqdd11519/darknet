#ifndef __CL__vFPGA__
#define __CL__vFPGA__

#include <CL/cl.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdarg>
#include <unistd.h>
#include <chrono>

#include "image.h"

void printError(cl_int error);
void _checkError(int line,
                                                                 const char *file,
                                                                 cl_int error,
                 const char *msg,
                 ...); // does not return
#define checkError(status, ...) _checkError(__LINE__, __FILE__, status, __VA_ARGS__)

unsigned char *loadBinaryFile(const char *file_name, size_t *size);

bool fileExists(const char *file_name);

void init_cl();
void run_cl(std::string path);
void exportOutput(float*);

class CLvFPGA{
  public:
    CLvFPGA();
    image runResize(image img);

  private:
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_mem input;
    cl_mem output;

    float * output_host;
};

#endif
