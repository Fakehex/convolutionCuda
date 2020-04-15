#include <opencv2/opencv.hpp>
#include <vector>
#include <omp.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < cols && j < rows ) {
    g[ j * cols + i ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) / 1024;
  }
}

__global__ void convolution_global_memory_gray(unsigned char *N,float *M,unsigned char* g,std::size_t cols, std::size_t rows,std::size_t mask_size){
  //magical numbers
  int paddingSize = ( mask_size-1 )/2;
  unsigned int paddedH = cols + 2 * paddingSize;
  unsigned int paddedW = rows + 2 * paddingSize;

  int i = blockIdx.x * blockDim.x + threadIdx.x + paddingSize;
  int j = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;

  if( (j >= paddingSize) && (j < paddedW-paddingSize) && (i >= paddingSize) && (i<paddedH-paddingSize)) {
    unsigned int oPixelPos = (j - paddingSize ) * cols + (i -paddingSize);
    for(int k = -paddingSize; k <= paddingSize; k++){
      for(int l = -paddingSize; l<=paddingSize; l++){
        unsigned int iPixelPos = (j+l)*cols+(i+k);
        unsigned int coefPos = (k + paddingSize) * mask_size + (l+ paddingSize);
        g[oPixelPos] += N[iPixelPos] * M[coefPos];
      }
    }
  }
}

static void simple_blur(std::vector< float >  &M_h, int mask_size){
  for(int i = 0; i< mask_size; i++){
    for(int j = 0; j< mask_size; j++){
      M_h[i+j*mask_size] = 1.0/(mask_size*mask_size);
    }
  }
}
static void left_sobel_maskSize3(std::vector< float >  &M_h){
  unsigned int mask_size = 3;
  for(int i = 0; i< mask_size; i++){
    for(int j = 0; j< mask_size; j++){
      if(i==1){
        M_h[i+j*mask_size] = 0;
      }else{
        if(i==0){
          if(j==1){
            M_h[i+j*mask_size] = 2;
          }else{
            M_h[i+j*mask_size] = 1;
          }
        }
        if(i==2){
          if(j==1){
            M_h[i+j*mask_size] = -2.0;
          }else{
            M_h[i+j*mask_size] = -1.0;

          }
        }

      }
    }
  }
}
int main()
{

  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  std::vector< unsigned char > g( rows * cols );
  std::vector< unsigned char > g2( rows * cols );
  unsigned char * rgb_d;
  unsigned char * g_d;
  unsigned char * g2_d;
  unsigned char * out_d;

  //convolution
  unsigned int mask_size = 9;
  std::vector< float > M_h( mask_size * mask_size * sizeof(float) );
  //simple blur
  simple_blur(M_h,mask_size);
  //left_Sobel seulement avec mask 3
  //left_sobel_maskSize3(M_h);


  std::size_t paddingSize = (mask_size-1)/2;
  std::size_t paddedW = rows + 2 * paddingSize;
  std::size_t paddedH = cols + 2 * paddingSize;
  std::vector< unsigned char > data_pad( paddedH * paddedW );


  //fin
  HANDLE_ERROR(cudaMalloc( &rgb_d, 3 * rows * cols ));
  HANDLE_ERROR(cudaMalloc( &g_d, rows * cols ));
  HANDLE_ERROR(cudaMalloc( &g2_d, rows * cols ));
  HANDLE_ERROR(cudaMalloc( &out_d, rows * cols ));
  HANDLE_ERROR(cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice ));
  dim3 t( 32, 32 );
  dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );



  grayscale<<< b, t >>>( rgb_d, g_d, cols, rows );

  //sobel<<< b, t >>>( g_d, out_d, cols, rows );


  HANDLE_ERROR(cudaMemcpy( g.data(), g_d, rows * cols, cudaMemcpyDeviceToHost ));
  //cv::imwrite( "out.jpg", m_out );




  for(int i=0; i < paddedW ; i++){
    for(int j=0; j < paddedH ; j++){
      if((i<=paddingSize && j<=paddingSize)|| (i>=paddedW-paddingSize & j>=paddedH-paddingSize)){
        data_pad[i+j*paddedW] = 255;
      }else{
        data_pad[i+j*paddedW] = g[i+j*paddedW];
      }
    }
  }
  float * M_d;
  unsigned char * data_d;
  HANDLE_ERROR(cudaMalloc( &M_d, mask_size * mask_size * sizeof(float)));
  HANDLE_ERROR(cudaMalloc( &data_d, paddedH * paddedW ));


  HANDLE_ERROR(cudaMemcpy(data_d, data_pad.data(), paddedW * paddedH, cudaMemcpyHostToDevice ));
  HANDLE_ERROR(cudaMemcpy(M_d, M_h.data(),mask_size * mask_size*sizeof(float),cudaMemcpyHostToDevice));

  dim3 t2( 32, 32 );
  dim3 b2( ( paddedH - 1) / t.x + 1 , ( paddedW - 1 ) / t.y + 1 );

  convolution_global_memory_gray<<< b2, t2 >>>( data_d,M_d, g2_d, cols, rows,mask_size );


  HANDLE_ERROR(cudaMemcpy( g2.data(), g2_d, rows * cols, cudaMemcpyDeviceToHost ));

  cv::Mat m_out( rows, cols, CV_8UC1, g2.data() );
  cv::imwrite( "out.jpg", m_out );

  cudaDeviceSynchronize();
/*  auto err = cudaGetLastError();
  if( err != cudaSuccess )
  {
    std::cout << cudaGetErrorString( err );
  }
  /*/
  cudaFree( rgb_d);
  cudaFree( g_d);
  return 0;
}
