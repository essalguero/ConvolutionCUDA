// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes, project
#include <cutil.h>

// include, images library
#include "CImg.h"



#define MIN_VALUE_BIG (1.0 / 36.0)
#define MID_VALUE_BIG (2.0 / 36.0)
#define MAX_VALUE_BIG (4.0 / 36.0)


#define MIN_VALUE_SMALL (1.0 / 9.0)
#define MAX_VALUE_SMALL (4.0 / 9.0)

#define FILTER_7x7 MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MID_VALUE_BIG, MID_VALUE_BIG, MID_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, 	MIN_VALUE_BIG, MIN_VALUE_BIG, MID_VALUE_BIG, MAX_VALUE_BIG, MID_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MID_VALUE_BIG, MID_VALUE_BIG, MID_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG, MIN_VALUE_BIG

#define FILTER_5x5 MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MAX_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL

#define FILTER_3x3 MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MAX_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL, MIN_VALUE_SMALL



#define USED_FILTER FILTER_7x7
#define FILTER_HEIGHT 7
#define FILTER_WIDTH FILTER_HEIGHT



#define ANCHO_GATOS 323
#define ALTO_GATOS 156
#define IMAGE_GATOS "gatos.jpeg"

#define ANCHO_CIUDAD 768
#define ALTO_CIUDAD 576
#define IMAGE_CIUDAD "ciudad.jpeg"

#define ANCHO_MOUNTAIN 6000
#define ALTO_MOUNTAIN 4000
#define IMAGE_MOUNTAIN "mountain.jpg"

#define ANCHO_EDIFICIO 3264
#define ALTO_EDIFICIO 2448
#define IMAGE_EDIFICIO "edificio.jpg"

#define ANCHO_GITHUB 225
#define ALTO_GITHUB 225
#define IMAGE_GITHUB "GitHubImage.png"

#define IMAGE_HEIGHT ALTO_EDIFICIO
#define IMAGE_WIDTH ANCHO_EDIFICIO
#define IMAGE_NAME IMAGE_EDIFICIO



#define XY_TO_ARRAY(x, y, width) ((x * width) + y)
#define ERROR_CHECK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
    printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}


using namespace cimg_library;

float * convolutionWithCuda(float * originalImage, float * originalFilter);
float * convolutionCPU(float * originalImage, float * originalFilter);

// Shows in the console the value of all the pixels in the image
void displayImage(float * image, int width, int height)
{
	
	for(int i = 0; i < height; ++i)
	{
		for(int j = 0; j < height; ++j)
		{
			printf("%f ", image[XY_TO_ARRAY(i, j, width)]);
		}
		printf("\n");
	}	
}


__global__ void convolutionKernel(float * deviceFilter, int filterHeight, int filterWidth, 
	                              float * deviceImageToProcess, int imageHeight, int imageWidth, int totalImageHeight, int totalImageWidth,
								  float * deviceImageProcessed)
{
	__shared__ float imageShared[FILTER_HEIGHT + FILTER_HEIGHT - 1][FILTER_WIDTH + FILTER_WIDTH - 1];
	__shared__ float filterShared[FILTER_HEIGHT][FILTER_WIDTH];

	float pixelProcessed = 0.0f;

	int idThread = threadIdx.y * blockDim.x + threadIdx.x;
	

	// Copy the filter to shared memory
	filterShared[threadIdx.y][threadIdx.x] = deviceFilter[idThread];

	//Calculate the position where the thread should start copying the image to process to shared memory
	int rowSize = totalImageWidth;
	int offset = rowSize * (filterHeight / 2) + (filterWidth / 2);
	int posicionInicial = ((blockIdx.y * blockDim.y * rowSize) + ((threadIdx.y - (FILTER_HEIGHT / 2)) * rowSize) + (blockIdx.x * blockDim.x) + (threadIdx.x - (FILTER_WIDTH / 2))) + offset;

	
	//Calculate the offset to the lower row this thread should copy image data
	int  filaInferior = (filterHeight - 1) * totalImageWidth;

	//Copy the needed values to the image in shared memory. Each thread is in charge of copying 4 values to the final image
	imageShared[threadIdx.y][threadIdx.x] = deviceImageToProcess[posicionInicial];
	imageShared[threadIdx.y][threadIdx.x + filterWidth - 1] = deviceImageToProcess[posicionInicial + filterWidth - 1];
	imageShared[threadIdx.y + filterHeight - 1][threadIdx.x + filterWidth - 1] = deviceImageToProcess[posicionInicial + filaInferior + filterWidth - 1 ];
	imageShared[threadIdx.y + filterHeight - 1][threadIdx.x] = deviceImageToProcess[posicionInicial + filaInferior];


	//All the threads should be synchronized after the data copy. When they get synchronized, the copy of data to shared memory has finished
	__syncthreads();


	// Convolution process
	for (int i = 0; i < FILTER_WIDTH; ++i)
		for (int j = 0; j < FILTER_HEIGHT; ++j)
			pixelProcessed += filterShared[i][j] * imageShared[i + threadIdx.y][j + threadIdx.x];


	// Copy the data to the final image	
	deviceImageProcessed[posicionInicial + offset] = pixelProcessed;
}	
//__global__ void convolutionKernel


int main()
{

	clock_t startTime;
	clock_t endTime;

    
	// Cargar la imagen and display it on a windws
	CImg<float> image(IMAGE_NAME);
	CImgDisplay original_disp(image, "Original Image", 0);



	float * originalImage = (float *)malloc(image.height() * image.width() * sizeof(float));



	float originalFilter [] = { USED_FILTER };

	float * cpuResult;
	float * gpuResult;

	for (int i = 0; i < image.height(); ++i)
	{
		for (int j = 0; j < image.width(); ++j)
		{
			originalImage[XY_TO_ARRAY(i, j, image.width())] = image[XY_TO_ARRAY(i, j, image.width())];
		}
	}



	// Measure the execution time for CPU algorithm
	startTime = clock();
	cpuResult = convolutionCPU(originalImage, originalFilter);
	endTime = clock();



	CImg<float> imageProcessedCPU(cpuResult, image.width(), image.height());

	CImgDisplay cpu_disp(imageProcessedCPU, "Image Processed on CPU", 0);

	// Save the image processed by the CPU as a BMP image
	imageProcessedCPU.save("ImageCPU.bmp");


	printf("Tiempo en ejecutar la convolucion en CPU: %f seconds\n", float(endTime - startTime) / CLOCKS_PER_SEC);


	// Measure the execution time for GPU algorithm
	startTime = clock();
	gpuResult = convolutionWithCuda(originalImage, originalFilter);
	endTime = clock();


	CImg<float> imageProcessedGPU(gpuResult, image.width(), image.height());

	// Save the image processed by the GPU as a BMP image
	imageProcessedGPU.save("ImageGPU.bmp");

	printf("Tiempo en ejecutar la convolucion en GPU: %f seconds\n", float(endTime - startTime) / CLOCKS_PER_SEC);

	// Show the image
	CImgDisplay gpu_disp(imageProcessedGPU, "Image Processed on GPU", 0);

	// Añadir un bucle para esperar a que el usuario cierre todas las ventanas
	while (!original_disp.is_closed() || !cpu_disp.is_closed() || !gpu_disp.is_closed()) {
		original_disp.wait();
	}


	// liberar los arrays conteniendo las imagenes
	free(originalImage);
	free(gpuResult);
	free(cpuResult);

	return 0;
}
// int main()



float * convolutionWithCuda(float * originalImage, float * originalFilter)
{
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

	dim3 block;
	dim3 grid;

	printf("Ejecucion comenzada\n");

	
	//hostFilter
	hostFilter = (float *)malloc(FILTER_HEIGHT * FILTER_WIDTH * sizeof(float));

	for (int i = 0; i < FILTER_HEIGHT * FILTER_WIDTH; ++i)
	{
		hostFilter[i] = originalFilter[i];
	}

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
	{
		hostImage[i] = originalImage[i];
	}

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
	hostImageToProcess = (float *)calloc(filasNecesarias * columnasNecesarias, sizeof(float));
	hostImageProcessed = (float *)calloc(filasNecesarias * columnasNecesarias, sizeof(float));


	//memset(hostImageToProcess, 0, filasNecesarias * columnasNecesarias * sizeof(float));

	for (int i = 0; i < imageHeight; ++i)
		for (int j = 0; j < imageWidth; ++j)
			hostImageToProcess[XY_TO_ARRAY((i + (filterHeight / 2)), (j + (filterWidth / 2)), columnasNecesarias)] = hostImage[XY_TO_ARRAY(i, j, imageWidth)];


	//displayImage(hostImageToProcess, columnasNecesarias, filasNecesarias);


	//Cargar la imagen a GPU

	//Reservar memoria en la tarjeta para cargar la imagen
	cudaMalloc((void **)&deviceImageToProcess, filasNecesarias * columnasNecesarias * sizeof(float));
	cudaMemcpy(deviceImageToProcess, hostImageToProcess, filasNecesarias * columnasNecesarias * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&deviceImageProcessed, filasNecesarias * columnasNecesarias * sizeof(float));
	cudaMemset(deviceImageProcessed, 0, filasNecesarias * columnasNecesarias * sizeof(float));


	// Hacer los tamaños de bloque de tareas en funcion del tamaño de la matriz del filtro
	grid.x = (imageWidth / FILTER_WIDTH) + 1;
	grid.y = (imageHeight / FILTER_HEIGHT) + 1;

	block.x = FILTER_WIDTH;
	block.y = FILTER_HEIGHT;


	ERROR_CHECK

	convolutionKernel <<<grid, block>>>(deviceFilter, FILTER_HEIGHT, FILTER_WIDTH, deviceImageToProcess, imageHeight, imageWidth, filasNecesarias, columnasNecesarias, deviceImageProcessed);

	ERROR_CHECK

	cudaThreadSynchronize();


	cudaMemcpy(hostImageProcessed, deviceImageProcessed, filasNecesarias * columnasNecesarias * sizeof(float), cudaMemcpyDeviceToHost);



	// Extraer Imagen
	for (int i = 0; i < imageHeight; ++i)
	{
		for (int j = 0; j < imageWidth; ++j)
		{


			finalImage[XY_TO_ARRAY(i, j, imageWidth)] = hostImageProcessed[XY_TO_ARRAY((i + (FILTER_HEIGHT / 2)), (j + (FILTER_WIDTH / 2)), columnasNecesarias)];
		}
	}


	// release memory allocated
	cudaFree(deviceImage);
	cudaFree(deviceImageToProcess);
	cudaFree(deviceImageProcessed);

	free(hostImage);
	free(hostImageToProcess);
	free(hostImageProcessed);

	printf("Ejecucion finalizada\n");

	return finalImage;
}
// float * convolutionWithCuda(float * originalImage, float * originalFilter)



float * convolutionCPU(float * originalImage, float * originalFilter)
{
	float * hostImage = 0;
	float * hostFilter = 0;
	float * hostImageProcessed = 0;
	float * hostImageToProcess = 0;

	float * finalImage = 0;

	int imageWidth = 0;
	int imageHeight = 0;
	int filterWidth = 0;
	int filterHeight = 0;


	// Obtener el tamaño de la imagen
	imageWidth = IMAGE_WIDTH;
	imageHeight = IMAGE_HEIGHT;

	filterHeight = FILTER_HEIGHT;
	filterWidth = FILTER_WIDTH;

	
	printf("Convolution on CPU started\n");

	//Get memory for hostFilter
	hostFilter = (float *)malloc(FILTER_HEIGHT * FILTER_WIDTH * sizeof(float));

	if (hostFilter == 0)
	{
		printf("Error getting memory hostFilter CPU\n");
		exit(1);
	}

	for (int i = 0; i < FILTER_HEIGHT * FILTER_WIDTH; ++i)
	{
		hostFilter[i] = originalFilter[i];
	}


	// obtener memoria para las imagenes

	// Imagen original
	hostImage = (float *)malloc(imageWidth * imageHeight * sizeof(float));


	if (hostImage == 0)
	{
		printf("Error getting memory hostImage CPU\n");
		exit(1);
	}

	for (int i = 0; i < imageWidth * imageHeight; ++i)
	{
		hostImage[i] = originalImage[i];
	}

	// Reservar memoria para la imagen resultante
	finalImage = (float *)malloc(imageWidth * imageHeight * sizeof(float));

	if (finalImage == 0)
	{
		printf("Error getting memory finalImage CPU\n");
		exit(1);
	}

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

	if (hostImageToProcess == 0)
	{
		printf("Error getting memory hostImageToProcess CPU\n");
		exit(1);
	}

	hostImageProcessed = (float *)malloc(filasNecesarias * columnasNecesarias * sizeof(float));

	if (hostImageProcessed == 0)
	{
		printf("Error getting memory hostImageProcessed CPU\n");
		exit(1);
	}

	// Set all the pixels in the image to 0
	memset(hostImageToProcess, 0, filasNecesarias * columnasNecesarias * sizeof(float));

	//Copy the image to process to a bigger array. This avoids problems when processing the image edges with the convolution matrix
	for (int i = 0; i < imageHeight; ++i)
		for (int j = 0; j < imageWidth; ++j)
			hostImageToProcess[XY_TO_ARRAY((i + (filterHeight / 2)), (j + (filterWidth / 2)), columnasNecesarias)] = hostImage[XY_TO_ARRAY(i, j, imageWidth)];

	// For testing purposes
	//displayImage(hostImage, imageWidth, imageHeight);

	// Convolution process
	// Loop through all the cells in the image but skipping the added rows and columns
	for(int i = filterHeight / 2; i < imageHeight + (filterHeight / 2); ++i)
	{
		for(int j = filterWidth / 2; j < imageWidth + (filterWidth / 2); ++j)
		{
			float result = 0;
			
			// Loop through all the cells in the filter
			for (int y = 0; y < filterHeight; y++)
			{
				for(int x = 0; x < filterWidth; x++)
				{
					result += hostFilter[XY_TO_ARRAY(y, x, filterWidth)] * 
						hostImageToProcess[XY_TO_ARRAY((i - ((filterHeight / 2) - y)), (j - ((filterWidth / 2) - x)), columnasNecesarias)];
				}
			}
			
			hostImageProcessed[XY_TO_ARRAY(i, j, columnasNecesarias)] = result;
			finalImage[XY_TO_ARRAY((i - (filterHeight / 2)), (j - (filterWidth / 2)), imageWidth)] = result;
		}
	}
	


	// For testing pusposes
	//displayImage(finalImage, imageWidth, imageHeight);
	

	//release the memory allocated
	free(hostFilter);
	free(hostImage);
	free(hostImageToProcess);
	free(hostImageProcessed);

	printf("Convolution on CPU finished\n");

	// return the final image
	return finalImage;

}
// float * convolutionCPU(float * originalImage, float * originalFilter)
