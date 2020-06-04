#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "main.h"

#define DEBUG 1U

void print_twoD(twoD_t *tmp, uint32_t channel)
{
#if (DEBUG)
	printf("Size: %uX%uX%u\n", tmp->channel, tmp->r, tmp->c);
	for (uint32_t i = 0; i < tmp->r; i++)
	{
		for (uint32_t j = 0; j < tmp->c; j++)
		{
			printf("%3.5f ", (tmp->data[channel * tmp->c * tmp->r + i * tmp->c + j]));
		}
		printf("\n");
	}
#endif
}

// Choose stride=c=r to reuse existing buffer again.
void maxPooling(twoD_t *inout, uint32_t r, uint32_t c, uint32_t stride)
{
	data_t max = 0;

	for (uint32_t i = 0; i < inout->r;  i+=stride)
	{
		for (uint32_t j = 0; j < inout->c;  j+=stride)
		{
			max = 0;
			for (uint32_t ii = 0; ii < r;  ii++)
			{
				for (uint32_t jj = 0; jj < c;  jj++)
				{
					if (inout->data[(i+ii) * inout->c + (j+jj)] > max)
					{
						max = inout->data[(i+ii) * inout->c + (j+jj)];
					}
				}
			}
			inout->data[i/stride * inout->r/stride + j/stride] = max;
		}
	}
}

void dot(twoD_t *input, const twoD_t *weights, twoD_t *output, data_t (*activation)(data_t))
{
	if (weights->r != input->r * input->c * input->channel)
	{
		printf("size mismatch\n");
	}

	output->r = weights->c;
	output->c = 1;

	data_t sum;

	for (uint32_t i = 0; i < weights->c;  i++)
	{
		sum = 0;
		for (uint32_t j = 0; j < weights->r;  j++)
		{
			sum += input->data[j] * weights->data[i * weights->r + j];
		}

		if (weights->bias)
		{
			sum += weights->bias[i];
		}

		if (activation)
		{
			sum = activation(sum);
		}
        
        	output->data[i] = sum;
	}
}

data_t reLU(data_t a)
{
	return a > 0 ? a : 0;
}

// only works when stride = r = c
void conv2D(twoD_t *input, const twoD_t *kernel, twoD_t *output, data_t (*activation)(data_t), uint32_t stride, uint32_t r, uint32_t c, uint32_t padding)
{
	uint32_t aux_r = input->r - kernel->r + 1 + 2 * padding;
	uint32_t aux_c = input->c - kernel->c + 1 + 2 * padding;
	data_t aux_sum;
	data_t max_pool;

	output->r = aux_r / stride;
	output->c = aux_c / stride;

	for (uint32_t out_ch = 0; out_ch < output->channel; out_ch++)
	{
		for (uint32_t i = 0; i < aux_r;  i+=stride)
		{
			for (uint32_t j = 0; j < aux_c;  j+=stride)
			{
				max_pool = 0; /* it is safe because post ReLU values are being compared */
				for (uint32_t i_pool = 0; i_pool < r;  i_pool++)
				{
					for (uint32_t j_pool = 0; j_pool < c;  j_pool++)
					{
						aux_sum = 0;
						for (uint32_t ii = 0; ii < kernel->r;  ii++)
						{
							for (uint32_t jj = 0; jj < kernel->c;  jj++)
							{
								for (uint32_t in_ch = 0; in_ch < input->channel;  in_ch++)
								{
									data_t in_data;
									uint32_t sub_i = i + i_pool + ii;
									uint32_t sub_j = j + j_pool + jj;
									if ((sub_i < padding) || (sub_j < padding) || (sub_i >= (aux_r+padding)) || (sub_j >= (aux_c+padding)))
									{
										in_data  =  0.0;
									}
									else
									{

										in_data = input->data[in_ch * input->r * input->c + (i+i_pool+ii-padding) * input->c + (j+j_pool+jj-padding)];
									}
									aux_sum += in_data * kernel->data[out_ch * kernel->in_channel * kernel->r * kernel->c + in_ch * kernel->r * kernel->c + ii * kernel->c + jj];
								}
							}
						}
						if (kernel->bias)
						{
							aux_sum += kernel->bias[out_ch];
						}

						if (activation)
						{
							aux_sum = activation(aux_sum);
						}

						if (aux_sum > max_pool)
						{
							max_pool = aux_sum;
						}
					}
				}
				output->data[out_ch * output->r * output->c  + i/stride * output->c + j/stride] = max_pool;
			}
		}
	}
}

uint32_t get_class(const twoD_t *output)
{
	uint32_t max_inx = 0;
	for (uint32_t i = 0; i < output->r; i++)
	{
		if (output->data[i] > output->data[max_inx])
		{
			max_inx = i;
		}
	}
	return max_inx;
}

int main() 
{
	twoD_t input, output, output2, output3, output4, output5;

	// Maximum memory usage is 2*max(3*32*32, 32*16*16, 16*8*8, 32*4*4, 10)*sizeof(data_t), ping pong buffer is being used.
	// Maximum memory usage is 2*max(3072,    8192,     1024,   512,    10)*sizeof(data_t), ping pong buffer is being used.
	
	data_t buffer1[3072];
	data_t buffer2[8192];

	input.r = input.c = 32;
	input.data = buffer1;
	input.bias = NULL;

	output.r = output.c = (32-5+1+4)/2;
	output.data = buffer2; // Must be set here
	output.bias = NULL;

	output2.r = output2.c = (16-5+1+4)/2;
	output2.data = buffer1;
	output2.bias = NULL;

	output3.r = output3.c = (8-5+1+4)/2;
	output3.data = buffer2;
	output3.bias = NULL;

	output4.r = 4*4*32;
	output4.c = 1;
	output4.data = buffer1;
	output4.bias = NULL;

	memcpy(buffer1, test, sizeof(test));

	printf("---Network starts---\n");

	input.channel = 3; /* 3 channel input 32x32 matrix */
	output.channel = w_0_weight_2d.channel; /* 32 input channels */
	conv2D(&input, &w_0_weight_2d, &output, &reLU, 2, 2, 2, 2);
	print_twoD(&output, 1);

	output2.channel = w_3_weight_2d.channel; /* 32 input channels are maped to 16 channels */
	conv2D(&output, &w_3_weight_2d, &output2, &reLU, 2, 2, 2, 2);
	// print_twoD(&output2, 0);

	output3.channel = w_6_weight_2d.channel; /* 16 input channels are maped to 32 channels */
	conv2D(&output2, &w_6_weight_2d, &output3, &reLU, 2, 2, 2, 2);
	// print_twoD(&output3, 0);

	output4.channel = w_10_weight_2d.channel;
	dot(&output3, &w_10_weight_2d, &output4, NULL);

	print_twoD(&output4, 0);

	printf("prediction: %d\n", get_class(&output4));

	return 0;
}
