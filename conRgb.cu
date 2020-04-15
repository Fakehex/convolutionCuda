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

__global__ void convolution_rgb(unsigned char *N,float *M,unsigned char* g,std::size_t cols, std::size_t rows,std::size_t mask_size){

  int paddingSize = (( mask_size-1 )/2)*3;
  unsigned int paddedH = cols + 2 * paddingSize;
  unsigned int paddedW = rows*3 + 2 * paddingSize;

  int i = blockIdx.x * blockDim.x + threadIdx.x + paddingSize;
  int j = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;

  if( (j >= paddingSize) && (j < paddedW-paddingSize) && (i >= paddingSize) && (i<paddedH-paddingSize)) {
    unsigned int oPixelPos = (j - paddingSize ) * rows + (i -paddingSize);
    g[oPixelPos] = N[oPixelPos];
    for(int k = -paddingSize; k <= paddingSize; k++){
      for(int l = -paddingSize; l<=paddingSize; l++){
        unsigned int iPixelPos = (j+l)*paddedW+(i+k);
        unsigned int filtrePos = (k + paddingSize) * mask_size + (l+ paddingSize);

          //g[oPixelPos] += N[iPixelPos] * M[filtrePos];

      }
    }
  }
}

static void simple_blur_rgb(std::vector< float >  &M_h, int mask_size){
  for(int i = 0; i< mask_size; i++){
    for(int j = 0; j< mask_size*3; j++){ // on multiplie par 3 pour avoir un filtre coherent avec le rgb
      M_h[i+j*mask_size] = 1.0/(mask_size*mask_size);
    }
  }
}

int main()
{

  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  std::vector< unsigned char > g( rows * cols * 3);
  unsigned char * g_d;

  //init convolution, creation filtre et image avec padding
  unsigned int mask_size = 3;
  std::vector< float > M_h( mask_size * mask_size * sizeof(float) );
  //simple blur
  simple_blur_rgb(M_h,mask_size);

  std::size_t paddingSize = ((mask_size-1)/2) * 3;
  std::size_t paddedW = rows + 2 * paddingSize;
  std::size_t paddedH = cols + 2 * paddingSize;
  std::vector< unsigned char > data_pad( paddedH * paddedW * 3 );

  // on crée l'image avec le padding
  // a améliorer plus tard, 2 bande noir sur l'image,
  for(int i=0; i < paddedW*3 ; i++){
    for(int j=0; j < paddedH ; j++){
        if((i<=paddingSize && j<=paddingSize)|| (i>=paddedW-paddingSize & j>=paddedH-paddingSize) || (i>rows*3) || j >cols){
          data_pad[i*paddedH+j] = 0;
        }else{
          data_pad[i*paddedH+j] = rgb[i*cols+j];
        }
    }
  }

  //fin init convolution

  HANDLE_ERROR(cudaMalloc( &g_d, rows * cols * 3));

  float * M_d;
  unsigned char * data_d;
  HANDLE_ERROR(cudaMalloc( &M_d,3 * mask_size * mask_size * sizeof(float)));
  HANDLE_ERROR(cudaMalloc( &data_d, paddedH * paddedW * 3));


  HANDLE_ERROR(cudaMemcpy(data_d, data_pad.data(), paddedW * paddedH * 3, cudaMemcpyHostToDevice ));
  HANDLE_ERROR(cudaMemcpy(M_d, M_h.data(),3* mask_size * mask_size*sizeof(float),cudaMemcpyHostToDevice));

  dim3 t( 32, 32 );
  dim3 b( ( cols - 1) / t.x + 1 , ( rows*3 - 1 ) / t.y + 1 );
  convolution_rgb<<< b, t >>>( data_d,M_d, g_d, cols, rows,mask_size );


  HANDLE_ERROR(cudaMemcpy( g.data(), g_d, rows * cols * 3, cudaMemcpyDeviceToHost ));

  cv::Mat m_out( rows, cols, CV_8UC3, g.data() );
  cv::imwrite( "out.jpg", m_out );

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if( err != cudaSuccess )
  {
    std::cout << cudaGetErrorString( err );
  }

  cudaFree( g_d);
  return 0;
}
