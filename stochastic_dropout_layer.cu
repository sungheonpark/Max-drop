#include <vector>

#include "caffe/layers/stochastic_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype>
void StochasticDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
	  //LOG(INFO) << "layer drop : " << prob_layer_uint_ << "curProb : " << curProb_;
	  if (prob_layer_uint_){
		  if (curProb_ > 0.7) curProb_ = 0.7;
		  scale_ = 1. / (1. - curProb_);
		  if (scale_ > 1.5) scale_ = 1.5;
		  //scale_ = 1;
		  uint_thres_ = static_cast<unsigned int>(UINT_MAX * curProb_);

		  unsigned int* mask =
			  static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
		  caffe_gpu_rng_uniform(count, mask);

		  // set thresholds
		  // NOLINT_NEXT_LINE(whitespace/operators)
		  DropoutForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			  count, bottom_data, mask, uint_thres_, scale_, top_data);
		  CUDA_POST_KERNEL_CHECK;
	  }
	  else{
		  //no dropout performed
		  caffe_copy(count, bottom_data, top_data);
	  }
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
void StochasticDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
	      const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
		if (prob_layer_uint_){
			const unsigned int* mask =
				static_cast<const unsigned int*>(rand_vec_.gpu_data());
			const int count = bottom[0]->count();
			// NOLINT_NEXT_LINE(whitespace/operators)
			DropoutBackward<Dtype> << <CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS >> >(
				count, top_diff, mask, uint_thres_, scale_, bottom_diff);
			CUDA_POST_KERNEL_CHECK;
		}
		else{
			//no dropout
			caffe_copy(top[0]->count(), top_diff, bottom_diff);
		}
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(StochasticDropoutLayer);

}  // namespace caffe
