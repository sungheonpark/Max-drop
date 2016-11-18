#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/max_drop_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	using std::min;
	using std::max;

	template <typename Dtype>
	void MaxDropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		MaxDropParameter max_drop_param = this->layer_param_.max_drop_param();

		//set params
		dim_ = max_drop_param.dim_option();
		kernel_h_ = max_drop_param.kernel_h();
		kernel_w_ = max_drop_param.kernel_w();
		prob_ = max_drop_param.prob();
		DCHECK(prob_ > 0.);
		DCHECK(prob_ < 1.);
		prob_int_ = static_cast<unsigned int>(UINT_MAX * prob_);
		const int width = bottom[0]->width();
		const int height = bottom[0]->height();
		if (dim_ == 1){
			scale_ = 1. / (1. - prob_*kernel_h_*kernel_w_ / (width*height));
		}
		else{
			scale_ = 1-prob_*(1 / bottom[0]->channels())*kernel_h_*kernel_w_ / (width*height);
		}
	}

	template <typename Dtype>
	void MaxDropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		top[0]->ReshapeLike(*bottom[0]);
		if (this->phase_ == TRAIN) {
			mask_.ReshapeLike(*bottom[0]);
			if (dim_ == 1){
				offIndex_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
			}
			else{
				offIndex_.Reshape(bottom[0]->num(),bottom[0]->width(), bottom[0]->height(), 1);
			}
		}
	}

	template <typename Dtype>
	void MaxDropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		LOG(FATAL) << "Only GPU implementation available for MaxDrop layer.";
	}

	template <typename Dtype>
	void MaxDropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		LOG(FATAL) << "Only GPU implementation available for MaxMaxDrop layer.";
	}


#ifdef CPU_ONLY
	STUB_GPU(MaxDropLayer);
#endif

	INSTANTIATE_CLASS(MaxDropLayer);
	REGISTER_LAYER_CLASS(MaxDrop);

}  // namespace caffe
