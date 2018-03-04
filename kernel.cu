
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define FILTER_HEIGHT 3
#define FILTER_WIDTH 3

#define IMAGE_HEIGHT 8000
#define IMAGE_WIDTH 10000

#define XY_TO_ARRAY(x, y, width) ((x * width) + y)
#define ERROR_CHECK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
    printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void convolutionWithCuda();


//__global__ void convolutionKernel(float * deviceImageToProcess, float * deviceImageProcessed, int imageHeight, int imageWidth,
//	                              float * deviceFilter, int filterHeight, int filterWidth)
__global__ void convolutionKernel(float * deviceFilter, int filterHeight, int filterWidth, 
	                              float * deviceImageToProcess, int imageHeight, int imageWidth,
								  float * deviceImageProcessed)
{
	__shared__ float imageShared[FILTER_HEIGHT + FILTER_HEIGHT - 1][FILTER_WIDTH + FILTER_WIDTH - 1];
	__shared__ float filterShared[FILTER_HEIGHT][FILTER_WIDTH];

	float pixelProcessed = 0.0f;

	int idThread = threadIdx.y * blockDim.x + threadIdx.x;
	//filterShared[threadIdx.x][threadIdx.y] = deviceFilter[idThread];

	//printf("ThreadIdx.x: %d - ThreadIdx.y: %d, filterValue: %f\n", threadIdx.x, threadIdx.y, deviceFilter[idThread]);
	
	//printf("BlockIdx.x: %d - BlockIdx.y: %d - ThreadIdx.x: %d - ThreadIdx.y: %d, idThread: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, idThread);

	filterShared[threadIdx.y][threadIdx.x] = deviceFilter[idThread];

	//printf("BlockIdx.x: %d - BlockIdx.y: %d - ThreadIdx.x: %d - ThreadIdx.y: %d, idThread: %d, filterValue: %f\n", 
	//	    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, idThread, filterShared[threadIdx.y][threadIdx.x]);


	int posicionInicial = blockIdx.y * blockDim.y * gridDim.x * blockDim.x + threadIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

	//threadIdx.y * gridDim.x * blockDim.x + threadIdx.x
	//threadIdx.y * gridDim.x * blockDim.x + (threadIdx.x + filterWidth - 1)
	//(threadIdx.y + filterHeight - 1) * gridDim.x * blockDim.x + (threadIdx.x + filterWidth - 1)
	//(threadIdx.y + filterHeight - 1) * gridDim.x * blockDim.x + threadIdx.x

	int filaInferior = (filterHeight - 1) * gridDim.x * blockDim.x;

	imageShared[threadIdx.y][threadIdx.x] = deviceImageToProcess[posicionInicial];
	imageShared[threadIdx.y][threadIdx.x + filterWidth - 1] = deviceImageToProcess[posicionInicial + filterWidth - 1];
	imageShared[threadIdx.y + filterHeight - 1][threadIdx.x + filterWidth - 1] = deviceImageToProcess[posicionInicial + filaInferior + filterWidth - 1 ];
	imageShared[threadIdx.y + filterHeight - 1][threadIdx.x] = deviceImageToProcess[posicionInicial + filaInferior];

	/*if (blockIdx.x == 1 && blockIdx.y == 1 && threadIdx.x == 0 && threadIdx.y == 0)
		printf("BlockIdx.x: %d - BlockIdx.y: %d - ThreadIdx.x: %d - ThreadIdx.y: %d, idThread: %d, filterValue: %f, valorImagen: %f, %f, %f, %f\n",
			blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, idThread, filterShared[threadIdx.y][threadIdx.x], 
			imageShared[threadIdx.y][threadIdx.x], imageShared[threadIdx.y][threadIdx.x + filterWidth - 1],
			imageShared[threadIdx.y + filterHeight - 1][threadIdx.x + filterWidth - 1], imageShared[threadIdx.y + filterHeight - 1][threadIdx.x]);

*/

	__syncthreads();


	// Convolution

	for (int i = 0; i < FILTER_WIDTH; ++i)
		for (int j = 0; j < FILTER_HEIGHT; ++j)
			pixelProcessed += filterShared[i][j] * imageShared[i + threadIdx.y][j + threadIdx.x];

	deviceImageProcessed[posicionInicial + (FILTER_WIDTH / 2) + ((FILTER_HEIGHT / 2) * gridDim.x * blockDim.x)] = pixelProcessed;

	//int i = threadIdx.x;

	//printf("c: %d\n", a);
}



int main()
{
    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }*/


	convolutionWithCuda();

    return 0;
}


void convolutionWithCuda()
{

	char * imageFilename = 0;
	float * deviceImage = 0;
	float * deviceFilter = 0;
	float * deviceImageToProcess = 0;
	float * deviceImageProcessed = 0;
	float * hostImage = 0;
	float * hostFilter = 0;
	float * hostImageProcessed = 0;
	float * hostImageToProcess = 0;

	float * finalImage = 0;

	int imageWidth = 0;
	int imageHeight = 0;
	int filterWidth = 0;
	int filterHeight = 0;
	int imageProcessedWidth = 0;
	int imageProcessedHeight = 0;
	int block_size = 0;

	dim3 block;
	dim3 grid;

	printf("Ejecucion comenzada\n");

	/*if (argc == 2) {
	imageFilename = strdup(argv[1]);

	} else {
	fprintf(stderr, "Error: Wrong input parameter numbers.\n");
	fprintf(stderr, "Usage:\n"
	"$> ./lab2.1-matrixmul <8, 128, 512, 3072, 4096>\n"
	"Examples:\n"
	"      $> ./lab2.1-matrixmul 128\n"
	);
	*/

	//hostFilter
	hostFilter = (float *)malloc(FILTER_HEIGHT * FILTER_WIDTH * sizeof(float));

	for (int i = 0; i < FILTER_HEIGHT * FILTER_WIDTH; ++i)
		hostFilter[i] = i;

	// deviceFilter

	cudaMalloc((void **)&deviceFilter, FILTER_HEIGHT * FILTER_WIDTH * sizeof(float));


	cudaMemcpy(deviceFilter, hostFilter, FILTER_HEIGHT * FILTER_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

	// Obtener el tamaño de la imagen
	imageWidth = IMAGE_WIDTH;
	imageHeight = IMAGE_HEIGHT;

	filterHeight = FILTER_HEIGHT;
	filterWidth = FILTER_WIDTH;


	// obtener memoria para las imagenes

	// Imagen original
	hostImage = (float *)malloc(imageWidth * imageHeight * sizeof(float));
	for (int i = 0; i < imageWidth * imageHeight; ++i)
		hostImage[i] = i;

	// Reservar memoria para la imagen resltante
	finalImage = (float *)malloc(imageWidth * imageHeight * sizeof(float));

	// Imagen procesada. Tiene que ser mas grande para que durante el procesado no haya problemas de indices
	// Calcular el numero de filas necesarias
	int filasNecesarias = imageHeight + (2 * (filterHeight / 2));
	if (filasNecesarias % filterHeight)
		filasNecesarias += filterHeight - (filasNecesarias % filterHeight);

	// Calcular el numero de columnas necesarias
	int columnasNecesarias = imageWidth + (2 * (filterWidth / 2));
	if (columnasNecesarias % filterWidth)
		columnasNecesarias += filterWidth - (columnasNecesarias % filterWidth);

	//reservar memoria
	hostImageToProcess = (float *)malloc(filasNecesarias * columnasNecesarias * sizeof(float));
	hostImageProcessed = (float *)malloc(filasNecesarias * columnasNecesarias * sizeof(float));


	memset(hostImageToProcess, 0, filasNecesarias * columnasNecesarias * sizeof(float));

	for (int i = 0; i < imageHeight; ++i)
		for (int j = 0; j < imageWidth; ++j)
			hostImageToProcess[XY_TO_ARRAY((i + (filterHeight / 2)), (j + (filterWidth / 2)), columnasNecesarias)] = hostImage[XY_TO_ARRAY(i, j, imageWidth)];


	///////////////////////////////////////////
	/*
	for (int i = 0; i < filasNecesarias; ++i)
	{
		for (int j = 0; j < columnasNecesarias; ++j)
		{
			printf("%f ", hostImageToProcess[XY_TO_ARRAY(i, j, columnasNecesarias)]);
		}
		printf("\n");
	}
	*/
	///////////////////////////////////////////


	//Cargar la imagen a GPU

	// La imagen sobre la que aplicar el filtro se crea
	// dos filas y dos columnas mas grandes en el device y se inicializan a 0
	// Esto permite evitar problemas con los indices

	/*imageProcessedWidth = imageWidth + (2 * (filterWidth / 2));
	imageProcessedHeight = imageHeight + (2 * (filterHeight / 2));
	*/

	//Reservar memoria en la tarjeta para cargar la imagen
	//cudaMalloc((void **)&deviceImage, imageProcessedWidth * imageProcessedHeight * sizeof(float));
	//cudaMemset(deviceImage, 0, imageProcessedWidth * imageProcessedHeight * sizeof(float));

	// Copiar la imagen a la tarjeta
	//memcpy(deviceImage + imageProcessedWidth, hostImage, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);


	cudaMalloc((void **)&deviceImageToProcess, filasNecesarias * columnasNecesarias * sizeof(float));
	cudaMemcpy(deviceImageToProcess, hostImageToProcess, filasNecesarias * columnasNecesarias * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&deviceImageProcessed, filasNecesarias * columnasNecesarias * sizeof(float));
	cudaMemset(deviceImageProcessed, 0, filasNecesarias * columnasNecesarias * sizeof(float));

	//cudaMemcpy(deviceImageToProcess, hostImageToProcess, filasNecesarias * columnasNecesarias * sizeof(float), cudaMemcpyDeviceToHost);

	//Reservar memoria en la tarjeta para guardar la imagen procesada
	//cudaMalloc((void **)&deviceImageProcessed, filasNecesarias * columnasNecesarias * sizeof(float));

	//cudaMemset(deviceImageProcessed, 0, imageWidth * imageHeight * sizeof(float));


	// Hacer los tamaños de bloque de tareas en funcion del tamaño de la matriz del filtro


	grid.x = columnasNecesarias / FILTER_WIDTH;
	grid.y = filasNecesarias / FILTER_HEIGHT;

	block.x = FILTER_WIDTH;
	block.y = FILTER_HEIGHT;









	//convolutionKernel << <1, 4 >> >(deviceImageToProcess, deviceImageProcessed, imageHeight, imageWidth, deviceFilter, filterHeight, filterWidth);
	
	
	ERROR_CHECK

	convolutionKernel <<<grid, block>>>(deviceFilter, FILTER_HEIGHT, FILTER_WIDTH, deviceImageToProcess, filasNecesarias, columnasNecesarias, deviceImageProcessed);

	ERROR_CHECK

		cudaThreadSynchronize();


	cudaMemcpy(hostImageProcessed, deviceImageProcessed, filasNecesarias * columnasNecesarias * sizeof(float), cudaMemcpyDeviceToHost);

	///////////////////////////////////////////
	/*
	printf("\n\n\n");

	for (int i = 0; i < filasNecesarias; ++i)
	{
		for (int j = 0; j < columnasNecesarias; ++j)
		{
			printf("%f ", hostImageProcessed[XY_TO_ARRAY(i, j, columnasNecesarias)]);
		}
		printf("\n");
	}


	printf("\n\n\n");
	*/
	///////////////////////////////////////////

	// Extraer Imagen


	for (int i = 0; i < imageHeight; ++i)
	{
		for (int j = 0; j < imageWidth; ++j)
		{


			finalImage[XY_TO_ARRAY(i, j, imageWidth)] = hostImageProcessed[XY_TO_ARRAY((i + (FILTER_HEIGHT / 2)), (j + (FILTER_WIDTH / 2)), columnasNecesarias)];
			//printf("%f ", finalImage[XY_TO_ARRAY(i, j, imageWidth)]);
		}
		//printf("\n");
	}

	cudaFree(deviceImage);
	cudaFree(deviceImageToProcess);
	cudaFree(deviceImageProcessed);

	free(hostImage);
	free(hostImageToProcess);
	free(hostImageProcessed);
	free(finalImage);

		printf("Ejecucion finalizada\n");
}


