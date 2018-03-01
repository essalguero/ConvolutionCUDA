#include <stdlib.h>
#include <stdio.h>
#include <string.h>

__global__ void kernel(float * deviceImage, float * deviceImageProcessed, float * deviceFilter, int filterHeight, int filterWidth, int imageHeight, int imageWidth)
{
    __shared__ float filter[filterHeigth][filterWidth];
    __shared__ float image[filterHeigth + 2][filterWidth + 2];

    //Copiar la matriz del filtro a memoria compartida
    

    //Copiar la parte de la imagen correspondiente a memoria compartida
}

int main (int argc, char ** argv)
{

    char *imageFilename = NULL;
    float * deviceImage = NULL, * deviceFilter = NULL, * deviceImageProcessed = NULL;
    float * hostImage = NULL, * hostFilter = NULL, * hostImageProcessed = NULL;
    int imageWidth = 0, imageHeight = 0, filterWidth = 0, filterHeith = 0, imageProcessedWidth = 0, imageProcessedHeight = 0;
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


    // Obtener el tamaño de la imagen
    imageWidth = 10;
    imageHeight = 10;



    hostImageProcessed = (float *)malloc(imageWidth * imageHeight * sizeof(float));

    hostImage = (float *)malloc(imageWidth * imageHeight * sizeof(float));

   

    //Cargar la imagen a GPU

    // La imagen sobre la que aplicar el filtro se crea
    // dos filas y dos columnas mas grandes en el device y se inicializan a 0
    // Esto permite evitar problemas con los indices

    imageProcessedWidth = imageWidth + 2;
    imageProcessedHeight = imageHeight + 2;

    //Reservar memoria en la tarjeta para cargar la imagen
    cudaMalloc(deviceImage, imageProcessedWidth * imageProcessedHeight * sizeof(float));
    cudaMemset(deviceImage, 0, imageProcessedWidth * imageProcessedHeight * sizeof(float));

    // Copiar la imagen a la tarjeta
    memcpy(deviceImage + imageProcessedWidth, hostImage, imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);

    //Reservar memoria en la tarjeta para guardar la imagen procesada
    cudaMalloc(deviceImageProcessed, imageWidth * imageHeight * sizeof(float));

    //cudaMemset(deviceImageProcessed, 0, imageWidth * imageHeight * sizeof(float));

    
    // Hacer los tamaños de bloque de tareas en funcion del tamaño de la matriz del filtro
    


    //llamada al kerner
    kernel<<<grid, block>>> (deviceImage, deviceImageProcessed, deviceFilter, filterHeight, filterWidth, imageHeight, imageWidth);


    
    

    printf("Ejecucion finalizada\n");
    printf("Liberando memoria...\n");

    free(hostImageProcessed);
    free(hostImage);

    cudaFree(deviceImage);
    cudaFree(deviceImageProcessed);

    exit(1);
}
