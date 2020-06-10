#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "main.h"
#include "helper_functions.h"


int main()
{
	 twoD_t meta_data0 = {
                           .r = 32,
                           .c = 32,
                           .channel = 3,
                           .data = buffer0,
                           .bias = NULL
                       };

	 twoD_t meta_data1 = {
                           .r = 16,
                           .c = 16,
                           .channel = 32,
                           .data = buffer1,
                           .bias = NULL
                       };

	 twoD_t meta_data2 = {
                           .r = 8,
                           .c = 8,
                           .channel = 16,
                           .data = buffer0,
                           .bias = NULL
                       };

	 twoD_t meta_data3 = {
                           .r = 4,
                           .c = 4,
                           .channel = 32,
                           .data = buffer1,
                           .bias = NULL
                       };

	 twoD_t meta_data4 = {
                           .r = 10,
                           .c = 1,
                           .channel = 1,
                           .data = buffer0,
                           .bias = NULL
                       };


	 memcpy(buffer0, test, sizeof(test));

	 printf("---Network starts---\n");
	 conv2D(&meta_data0, &w_0_weight_2d, &meta_data1, &reLU, 2, 2, 2, 2);
	 conv2D(&meta_data1, &w_3_weight_2d, &meta_data2, &reLU, 2, 2, 2, 2);
	 conv2D(&meta_data2, &w_6_weight_2d, &meta_data3, &reLU, 2, 2, 2, 2);
	 dot(&meta_data3, &w_10_weight_2d, &meta_data4, NULL);

	 print_twoD(&meta_data4, 0);
	 printf("PREDICTION: %d\n", get_class(&meta_data4));
	 return 0;
}