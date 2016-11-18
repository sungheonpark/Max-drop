#include <vector>

#include "caffe/layers/spatial_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void SpatialDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		NeuronLayer<Dtype>::LayerSetUp(bottom, top);
		threshold_ = this->layer_param_.spatial_dropout_param().prob();
		DCHECK(threshold_ > 0.);
		DCHECK(threshold_ < 1.);
		scale_ = 1. / (1. - threshold_);
		uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
	}

	template <typename Dtype>
	void SpatialDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		NeuronLayer<Dtype>::Reshape(bottom, top);
		// Set up the cache for random number generation, num x channel x 1 x 1 size
		rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
	}

	template <typename Dtype>
	void SpatialDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		unsigned int* mask = rand_vec_.mutable_cpu_data();
		const int num = bottom[0]->num();
		const int channels = bottom[0]->channels();
		const int width = bottom[0]->width();
		const int height = bottom[0]->height();
		if (this->phase_ == TRAIN) {
			// Create random numbers
			caffe_rng_bernoulli(num*channels, 1. - threshold_, mask);
			for (int n = 0; n < num; ++n) {
				for (int c = 0; c < channels; ++c) {
					int offIndex = n*channels + c;
					//off the entire wxh filter
					for (int h = 0; h < height; h++){
						for (int w = 0; w < width; w++){
							int index = (offIndex * height + h) * width + w;
							top_data[index] = bottom_data[index] * mask[offIndex] * scale_;
						}
					}
				}
			}
		}
		else {
			caffe_copy(bottom[0]->count(), bottom_data, top_data);
		}
	}

	template <typename Dtype>
	void SpatialDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			if (this->phase_ == TRAIN) {
				const unsigned int* mask = rand_vec_.cpu_data();
				const int num = bottom[0]->num();
				const int channels = bottom[0]->channels();
				const int width = bottom[0]->width();
				const int height = bottom[0]->height();
				for (int n = 0; n < num; ++n) {
					for (int c = 0; c < channels; ++c) {
						int offIndex = n*channels + c;
						//off the entire wxh filter
						for (int h = 0; h < height; h++){
							for (int w = 0; w < width; w++){
								int index = (offIndex * height + h) * width + w;
								bottom_diff[index] = top_diff[index] * mask[offIndex] * scale_;
							}
						}
					}
				}
			}
			else {
				caffe_copy(top[0]->count(), top_diff, bottom_diff);
			}
		}
	}


#ifdef CPU_ONLY
	STUB_GPU(DropoutLayer);
#endif

	INSTANTIATE_CLASS(SpatialDropoutLayer);
	REGISTER_LAYER_CLASS(SpatialDropout);

}  // namespace caffe
