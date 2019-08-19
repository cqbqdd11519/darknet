#include "cl_vfpga.hpp"

#define AOCX "resize.aocx"

#define OUT_W 320
#define OUT_H 320

#define BLOCK_W 32
#define BLOCK_H 32

#define MAX_W 2048
#define MAX_H 2048

CLvFPGA::CLvFPGA(): context(NULL), queue(NULL), kernel(NULL), input(NULL), output(NULL), output_host(NULL) {
  cl_int status;

  cl_uint n_platform;
  cl_platform_id platform;
  status = clGetPlatformIDs(0, NULL, &n_platform);
  checkError(status, "Platform IDs Fail");
  cl_platform_id platforms[n_platform];
  status = clGetPlatformIDs(n_platform, platforms, NULL);
  checkError(status, "Platform ID Fail");
  for(auto pid : platforms){
    size_t sz;
    char name[1024];
    status = clGetPlatformInfo(pid, CL_PLATFORM_NAME, 0, NULL, &sz);
    checkError(status, "Platform Info n Fail");
    status = clGetPlatformInfo(pid, CL_PLATFORM_NAME, sz, name, NULL);
    checkError(status, "Platform Info Fail");

    if(std::string(name) == std::string("Intel(R) FPGA SDK for OpenCL(TM)")){
        std::cout << "FPGA Platform Found" << std::endl;
        platform = pid;
        break;
    }
  }

  cl_uint n_device;
  cl_device_id device;
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &n_device);
  checkError(status, "Device IDs fail");
  cl_device_id devices[n_device];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, n_device, devices, NULL);
  checkError(status, "Device ID fail");

  n_device = 1;
  device = devices[1];

  context = clCreateContext(NULL, n_device, &device, NULL, NULL, &status);
  checkError(status, "Context fail");

  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Queue fail");

  cl_program program;
  cl_int bin_status;
  unsigned char* bin;
  size_t bin_sz;
  if(!fileExists(AOCX)) {
    printf("AOCX file '%s' does not exist.\n", AOCX);
    checkError(CL_INVALID_PROGRAM, "Failed to load binary file");
  }
  bin = loadBinaryFile(AOCX, &bin_sz);
  program = clCreateProgramWithBinary(context, n_device, &device, &bin_sz, (const unsigned char**) &bin, &bin_status, &status);
  checkError(status, "Program Error");
  checkError(bin_status, "Bin Error");

  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);

  kernel = clCreateKernel(program, "resize", &status);
  checkError(status, "Kernel fail");

  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, OUT_H * OUT_W * 3 * sizeof(float), NULL, &status);
  checkError(status, "Output Buffer fail");

  input = clCreateBuffer(context, CL_MEM_READ_ONLY, MAX_W * MAX_H * 3 * sizeof(float), NULL, &status);
  checkError(status, "Input Buffer fail");

  output_host = (float*)aligned_alloc(64, OUT_H * OUT_W * 3 * sizeof(float));
}

image CLvFPGA::runResize(image img){
  cl_int status;
  cl_event ev, ev_k;

  auto start = std::chrono::system_clock::now();

  int width = img.w;
  int height = img.h;

  float rat_w = (float)OUT_W / (float)width;
  float rat_h = (float)OUT_H / (float)height;

  float ratio = rat_w;
  int out_w = OUT_W;
  int out_h = (int)((float)height* ratio);

  if(rat_h < rat_w){
    ratio = rat_h;

    out_w = (int)((float)width * ratio);
    out_h = OUT_H;
  }

  int dx = (OUT_W - out_w)/2;
  int dy = (OUT_H - out_h)/2;

  int offset_in = width * height;
  int offset_out = OUT_W * OUT_H;

  float* input_host = (float*) img.data;

  input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 3 * sizeof(float), input_host, &status);
  checkError(status, "Input Buffer fail");

  status  = clSetKernelArg(kernel, 0, sizeof(int), &width);
  status |= clSetKernelArg(kernel, 1, sizeof(int), &height);
  status |= clSetKernelArg(kernel, 2, sizeof(int), &out_w);
  status |= clSetKernelArg(kernel, 3, sizeof(int), &out_h);
  status |= clSetKernelArg(kernel, 4, sizeof(int), &offset_in);
  status |= clSetKernelArg(kernel, 5, sizeof(int), &offset_out);
  status |= clSetKernelArg(kernel, 6, sizeof(int), &dx);
  status |= clSetKernelArg(kernel, 7, sizeof(int), &dy);
  status |= clSetKernelArg(kernel, 8, sizeof(float), &ratio);
  status |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &input);
  status |= clSetKernelArg(kernel, 10, sizeof(cl_mem), &output);
  checkError(status, "Set arg error");

  const size_t global[2] = {OUT_W, OUT_H};
  const size_t local[2] = {BLOCK_W, BLOCK_H};
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &ev_k);
  checkError(status, "Kernel launch error");

  clWaitForEvents(1, &ev_k);

  status = clEnqueueReadBuffer(queue, output, CL_FALSE, 0, OUT_H*OUT_W*3*sizeof(float), output_host, 0, NULL, &ev);
  checkError(status, "Read error");

  clWaitForEvents(1, &ev);

  auto end = std::chrono::system_clock::now();
  auto usec = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Resize takes " << ((float)usec.count()/1000.0f) << "msec" << std::endl;

  clReleaseEvent(ev);
  clReleaseEvent(ev_k);

  image img_out = make_empty_image(OUT_W, OUT_H, 3);
  img_out.data = output_host;
  
  return img_out;
}
