#include <opencv2/opencv.hpp>
#include <vector>
#include <omp.h>


const unsigned int MaxFiltreSize = 79;
__device__ __constant__ float filtre_d[MaxFiltreSize * MaxFiltreSize];

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

__global__ void convolution_rgb(unsigned char *paddedImage, unsigned char* g,std::size_t cols, std::size_t rows,std::size_t mask_size){

  int paddingSize = (( mask_size-1 )/2)*3;
  unsigned int paddedH = cols + 2 * paddingSize;
  unsigned int paddedW = rows*3 + 2 * paddingSize;

  int i = blockIdx.x * blockDim.x + threadIdx.x + paddingSize ;
  int j = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;
  if( (i >= paddingSize) && (i < paddedW-paddingSize) && (j >= paddingSize) && (j<paddedH-paddingSize)) {
    unsigned int oPixelPos = (i - paddingSize ) * cols + (j -paddingSize);
    g[oPixelPos] = 0;
    unsigned int iterationK = 0;
    for(int k = -paddingSize; k <= paddingSize; k=k+3){
      unsigned int iterationL = 0;
      for(int l = -paddingSize; l<=paddingSize; l=l+3){
        unsigned int iPixelPos = (i+k)*paddedH+(j+l);
        unsigned int filtrePos = iterationK*mask_size + iterationL;

        g[oPixelPos] += paddedImage[iPixelPos] * filtre_d[filtrePos];
        iterationL++;
      }
      iterationK++;
    }
  }
}

static void simple_blur_rgb(std::vector< float >  &M_h, int mask_size){
  for(int i = 0; i< mask_size; i++){
    for(int j = 0; j< mask_size; j++){ // on multiplie par 3 pour avoir un filtre coherent avec le rgb
      M_h[i+j*mask_size] = 1.0/(mask_size*mask_size);
    }
  }
}
//filtre de mask_size=3 attention dans le main a son utilisation
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
  //auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;

  //init convolution, creation filtre et image avec padding
  unsigned int mask_size = 3;

  //DEMANDE DE FILTRE
  int ask = 0;
  while(ask!=1 || ask!=2){

    std::cout << "Quel filtre voulez vous utiliser ? (1 ou 2)" << std::endl;
    std::cout << "1. simple blur" << std::endl;
    std::cout << "2. left sobel (avec la taille du masque = 3)" << std::endl;
    std::cin >> ask;
    if(std::cin.fail()){
      std::cout << "ERREUR" << std::endl;
      std::cin.clear();
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      ask = 0;
    }
    if(ask==1){
      std::cout << "quel taille du masque ? (3,5,9 conseillé)" << std::endl;
      std::cin >> mask_size;
      if(std::cin.fail()){
        std::cout << "ERREUR" << std::endl;
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        ask = 0;
        mask_size = 3;
      }
      break;
    }else{
      if(ask==2){
        break;
      }else{
        std::cout << "Erreur, Reessayez " << std::endl;
      }
    }

  }
  std::vector< float > filtre_h( mask_size * mask_size * sizeof(float) );
  if(ask==1){
    simple_blur_rgb(filtre_h,mask_size);
  }else{
    left_sobel_maskSize3(filtre_h);
  }


  std::size_t paddingSize = ((mask_size-1)/2)*3;
  std::vector< unsigned char > g( rows * cols * 3);
  copyMakeBorder(m_in,m_in,paddingSize,paddingSize,paddingSize,paddingSize,cv::BORDER_CONSTANT,0);
  auto data_pad = m_in.data;
  std::size_t paddedW = m_in.rows;
  std::size_t paddedH = m_in.cols;
  unsigned char * g_d;


  //fin init convolution
  unsigned char * data_d;
  unsigned int filterSizeByte = mask_size * mask_size * sizeof(float);

  dim3 t( 8, 8 );
  dim3 b( ( rows*3 - 1) / t.x + 1 , ( cols - 1 ) / t.y + 1 );

  HANDLE_ERROR(cudaMalloc( &g_d, rows * cols * 3));
  HANDLE_ERROR(cudaMemcpyToSymbol(filtre_d, filtre_h.data(),filterSizeByte,0,cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc( &data_d, paddedH * paddedW * 3));
  HANDLE_ERROR(cudaMemcpy(data_d, data_pad, paddedW * paddedH * 3, cudaMemcpyHostToDevice ));


  convolution_rgb<<< b, t>>>( data_d, g_d, cols, rows,mask_size );

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
  cudaFree(data_d);
  cudaFree(filtre_d);

  return 0;
}
