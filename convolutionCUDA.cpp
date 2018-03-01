#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main (int argc, char ** argv)
{

    char *imageFilename = NULL;
    float * deviceImage = NULL, * deviceFilter = NULL, * deviceImageProcessed = NULL;
    float * hostImage = NULL, * hostFilter = NULL, * hostImageProcessed = NULL;
    int imageWidth = 0, imageHeight = 0, filterWidth = 0, filterHeith = 0, imageProcessedWidth = 0, imageProcessedHeight = 0;
    int block_size = 0;

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

    // La imagen sobre la que aplicar el filtro, para evitar problemas con los indices, se crea
    // dos filas y dos columnas mas grandes en el device y se inicializan a 0
    imageProcessedWidth = imageWidth + 2;
    imageProcessedHeight = imageHeight + 2;


    hostImageProcessed = (float *)malloc(imageProcessedWidth * imageProcessedHeight * sizeof(float));

    hostImage = (float *)malloc(imageProcessedWidth * imageProcessedHeight * sizeof(float));

    memset(hostImageProcessed, 0, imageProcessedWidth * imageProcessedHeight * sizeof(float));


    //Cargar la imagen

    //Reservar memoria en la tarjeta para cargar la imagen
    cudaMalloc(deviceImageProcessed, imageProcessedWidth * imageProcessedHeight * sizeof(float));
    
    // Copiar la imagen a la tarjeta
    memcpy(deviceImage + imageProcessed Height, imageWidth * imageHeight * sizeof(float));

    //Reservar memoria en la tarjeta para guardar la imagen procesada
    cudaMalloc(deviceImageProcessed, imageProcessedWidth * imageProcessedHeight * sizeof(float));

    printf("Ejecucion finalizada\n");
    printf("Liberando memoria...\n");

    free(hostImageProcessed);

    exit(1);
}
