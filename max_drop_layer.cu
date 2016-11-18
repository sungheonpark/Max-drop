#include <algorithm>
#include <cfloat>
#include <vector>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include "caffe/layers/max_drop_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


	template <typename Dtype>
	__global__ void MaxDropForward(const int nthreads, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w, const unsigned int prob,
		const unsigned int* const offIndex, const Dtype* const bottom_data, Dtype *const mask) {

		CUDA_KERNEL_LOOP(index, nthreads){
			if (offIndex[index] < prob){
				//find max and off
				Dtype maxVal = Dtype(0);
				int w_offset = 0, h_offset = 0;
				int maxIndex = index*height*width;
				for (int h = 0; h < height; h++){
					for (int w = 0; w < width; w++){
						int index_in = (index*height + h)*width + w;
						if (bottom_data[index_in]>maxVal){
							maxVal = bottom_data[index_in];
							maxIndex = index_in;
							w_offset = w;
							h_offset = h;
						}
					}
				}

				//Recommand odd number of kernel size
				if (kernel_h>1 || kernel_w>1){
					int half_h = kernel_h / 2;
					int half_w = kernel_w / 2;
					for (int hh = -half_h; hh < half_h+1; hh++){
						for (int ww = -half_w; ww < half_w+1; ww++){
							//check boundary
							int curh = hh + h_offset;
							int curw = ww + w_offset;
							if (curh < height && curh >= 0 && curw < width && curw >= 0){
								int index_in = (index*height + curh)*width + curw;
								mask[index_in] = Dtype(0);
							}
						}
					}
				}
				else{
					mask[maxIndex] = Dtype(0);
				}
			}

		}
	}


	template <typename Dtype>
	__global__ void MaxDropForward_dim2(const int nthreads, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w, const int prob,
		const unsigned int* const offIndex, const Dtype* const bottom_data, Dtype* const mask){
		CUDA_KERNEL_LOOP(index, nthreads){
			if (offIndex[index] < prob){
				int n = index / (width*height);
				int wh = index % (width*height);
				int h = wh / height;
				int w = wh % height;
				//find max through channel and off
				Dtype maxVal = Dtype(0);
				int c_offset = 0;
				int maxIndex = (n*channels*height + h)*width + w;
				for (int c = 0; c < channels; c++){
					int index_in = ((n*channels + c)*height + h)*width + w;
					if (bottom_data[index_in]>maxVal){
						maxVal = bottom_data[index_in];
						maxIndex = index_in;
						c_offset = c;
					}
				}
				//Recommand odd number of kernel size
				if (kernel_h > 1 || kernel_w > 1){
					int half_h = kernel_h / 2;
					int half_w = kernel_w / 2;
					for (int hh = -half_h; hh < half_h + 1; hh++){
						for (int ww = -half_w; ww < half_w + 1; ww++){
							//check boundary
							int curh = hh + h;
							int curw = ww + w;
							if (curh < height && curh >= 0 && curw < width && curw >= 0){
								int index_in = ((n*channels + c_offset)*height + curh)*width + curw;
								mask[index_in] = Dtype(0);
							}
						}
					}
				}
				else{
					mask[maxIndex] = Dtype(0);
				}
			}
		}
	}

	template <typename Dtype>
	void MaxDropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const int num = bottom[0]->num();
		const int channels = bottom[0]->channels();
		const int width = bottom[0]->width();
		const int height = bottom[0]->height();
		const int count = bottom[0]->count();
		if (this->phase_ == TRAIN) {
			Dtype* mask_data = mask_.mutable_gpu_data();
			caffe_gpu_set(count, Dtype(1), mask_data);
			// Create random numbers
			unsigned int* offIndex =
				static_cast<unsigned int*>(offIndex_.mutable_gpu_data());
			if (dim_ == 1){
				// default : find max in feature map, selecting feature map in random
				//caffe_gpu_rng_bernoulli not implemented
				caffe_gpu_rng_uniform(num*channels, offIndex);
				//calc max for each channel map, TODO : fast max algorithm?
				MaxDropForward<Dtype> << <CAFFE_GET_BLOCKS(num*channels), CAFFE_CUDA_NUM_THREADS >> >(
					num*channels, channels, height, width, kernel_h_, kernel_w_, prob_int_, offIndex, bottom_data, mask_data);
		
			}
			else{
				// dim_2 : find max across channel, selecting spatial in random
				//caffe_gpu_rng_bernoulli not implemented
				caffe_gpu_rng_uniform(num*width*height, offIndex);
				//calc max for across channel
				MaxDropForward_dim2<Dtype> << <CAFFE_GET_BLOCKS(num*width*height), CAFFE_CUDA_NUM_THREADS >> >(
					num*width*height, channels, height, width, kernel_h_, kernel_w_, prob_int_, offIndex, bottom_data, mask_data);
			}
			caffe_gpu_mul(count, bottom_data, mask_.gpu_data(), top_data);
			caffe_gpu_scal(count, scale_, top_data);
			CUDA_POST_KERNEL_CHECK;
	
			
		}
		else{
			caffe_copy(count, bottom_data, top_data);
		}
	}
	

	template <typename Dtype>
	void MaxDropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) {
			return;
		}
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		if (this->phase_ == TRAIN) {
			// Create random numbers
			caffe_copy(top[0]->count(), top_diff, bottom_diff);
			caffe_gpu_mul(bottom[0]->count(), top_diff, mask_.gpu_data(), bottom_diff);
			caffe_gpu_scal(bottom[0]->count(), scale_, bottom_diff);
			CUDA_POST_KERNEL_CHECK;
		}
		else{
			//pass data
			caffe_copy(top[0]->count(), top_diff, bottom_diff);
		}
	}


	INSTANTIATE_LAYER_GPU_FUNCS(MaxDropLayer);


}  // namespace caffe
