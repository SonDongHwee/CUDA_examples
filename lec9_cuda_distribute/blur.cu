// we have 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
#include <iostream>
const int BLUR_SIZE = 3;
__global__ 
void blurKernel(unsigned char * in, unsigned char * out, int w, int h) {
  int Col  = blockIdx.x * blockDim.x + threadIdx.x;
  int Row  = blockIdx.y * blockDim.y + threadIdx.y;
  if (Col < w && Row < h) {
    int pixVal = 0;
    int pixels = 0;
      // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
      for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
        for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
          int curRow = Row + blurRow;
          int curCol = Col + blurCol;
          // Verify we have a valid image pixel
          if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
            pixVal += in[curRow * w + curCol];
            pixels++; // Keep track of number of pixels in the avg
          }
        }
      }
      // Write our new pixel value out
      out[Row * w + Col] = (unsigned char)(pixVal / pixels);
  }
}


int main() {
  unsigned char *grayImage_h;
  unsigned char *grayImage_d;
  unsigned char *grayImageOut_h;
  unsigned char *grayImageOut_d;


  int width = 76;
  int height = 62;
  int graySize = width*height;



  grayImage_h = (unsigned char*)malloc(graySize*sizeof(unsigned char));
  grayImageOut_h = (unsigned char*)malloc(graySize*sizeof(unsigned char));
  cudaMalloc((unsigned char**)&grayImage_d, graySize*sizeof(unsigned char));
  cudaMalloc((unsigned char**)&grayImageOut_d, graySize*sizeof(unsigned char));

  for(int i=0;i<width*height;i++) {
    int grayOffset = i;
    grayImage_h[grayOffset] = (i%3) * 10 + 10;
    grayImageOut_h[grayOffset] = 0;
  }

  dim3 dimGrid(ceil(width/16.0), ceil(height/16.0),1);
  dim3 dimBlock(16,16,1);

  cudaMemcpy(grayImage_d, grayImage_h, graySize, cudaMemcpyHostToDevice);
  blurKernel<<<dimGrid, dimBlock>>>(grayImage_d, grayImageOut_d, width, height);
  cudaMemcpy(grayImageOut_h, grayImageOut_d, graySize, cudaMemcpyDeviceToHost);

  for(int i=0;i<height;i++) {
    for(int j=0;j<width;j++) {
      //std::cout<<"gray "<<i<<","<<j<<":"<<(int)grayImage_h[i*width+j]<<std::endl;
      std::cout<<(int)grayImageOut_h[i*width+j]<<" ";
    }
    std::cout<<std::endl;
  }
  std::cout<<std::endl;
  std::cout<<std::endl;
  std::cout<<std::endl;
  return 0;
}




