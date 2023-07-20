// we have 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
#include <iostream>
const int CHANNELS = 3;
__global__ 
void colorToGreyscaleConvertion(unsigned char * rgbImage,  unsigned char * grayImage,
                          int width, int height) {

 int Col =   threadIdx.x + blockIdx.x * blockDim.x;
 int Row = threadIdx.y + blockIdx.y * blockDim.y;
 
 if (Col < width && Row < height) {
    // get 1D coordinate for the grayscale image
    int grayOffset = Row*width + Col;
    // one can think of the RGB image having
    // CHANNEL times columns of the gray scale image
    int rgbOffset = grayOffset*CHANNELS;
    unsigned char r = rgbImage[rgbOffset    ]; // red value for pixel
    unsigned char g = rgbImage[rgbOffset + 2]; // green value for pixel
    unsigned char b = rgbImage[rgbOffset + 3]; // blue value for pixel
    // perform the rescaling and store it
    // We multiply by floating point constants
    grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
 }
}

int main() {
	unsigned char *rgbImage_h, *grayImage_h;
	unsigned char *rgbImage_d, *grayImage_d;
  unsigned char *grayImage_ref;


  int width = 76;
  int height = 62;
  int rgbSize = width*height*CHANNELS;
  int graySize = width*height;


  rgbImage_h =(unsigned char*) malloc(rgbSize * sizeof(unsigned char));
  cudaMalloc((unsigned char**)&rgbImage_d, rgbSize*sizeof(unsigned char));

  grayImage_h = (unsigned char*)malloc(graySize*sizeof(unsigned char));
  cudaMalloc((unsigned char**)&grayImage_d, graySize*sizeof(unsigned char));
  grayImage_ref = (unsigned char*)malloc(graySize*sizeof(unsigned char));

  for(int i=0;i<width*height;i++) {
    int grayOffset = i;
    int rgbOffset = grayOffset*CHANNELS;
    rgbImage_h[rgbOffset     ] = 10;
    rgbImage_h[rgbOffset + 1 ] = 120;
    rgbImage_h[rgbOffset + 2 ] = 50;
    grayImage_h[grayOffset] = 0;
  }

  dim3 dimGrid(ceil(width/16.0), ceil(height/16.0),1);
  dim3 dimBlock(16,16,1);

	cudaMemcpy(rgbImage_d, rgbImage_h, rgbSize, cudaMemcpyHostToDevice);
  colorToGreyscaleConvertion<<<dimGrid, dimBlock>>>(rgbImage_d, grayImage_d, width, height);
	cudaMemcpy(grayImage_h, grayImage_d, graySize, cudaMemcpyDeviceToHost);

  for(int i=0;i<height;i++) {
    for(int j=0;j<width;j++) {
      //std::cout<<"gray "<<i<<","<<j<<":"<<(int)grayImage_h[i*width+j]<<std::endl;
      std::cout<<(int)grayImage_h[i*width+j];
    }
    std::cout<<std::endl;
  }
    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;
  for(int i=0;i<height;i++) {
    for(int j=0;j<width;j++) {
    int grayOffset = i;
    int rgbOffset = grayOffset*CHANNELS;
    unsigned char r = rgbImage_h[rgbOffset      ]; // red value for pixel
    unsigned char g = rgbImage_h[rgbOffset + 2]; // green value for pixel
    unsigned char b = rgbImage_h[rgbOffset + 3]; // blue value for pixel
      grayImage_ref[i*width+j] = 0.21f*r + 0.71f*g + 0.07f*b;
      std::cout<<(int)grayImage_ref[i*width+j];
    }
    std::cout<<std::endl;
  }
  return 0;
}




