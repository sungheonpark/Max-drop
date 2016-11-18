

#include <vector>

#include "caffe/layers/stochastic_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StochasticDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
	  prob_layer_ = this->layer_param_.stochastic_dropout_param().prob_layer();
	  prob_mode_ = this->layer_param_.stochastic_dropout_param().type();
	  prob_activation_arg1_ = this->layer_param_.stochastic_dropout_param().prob_arg1();
	  prob_activation_arg2_ = this->layer_param_.stochastic_dropout_param().prob_arg2();

}

template <typename Dtype>
void StochasticDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  //
  caffe_rng_bernoulli(1, prob_layer_, &prob_layer_uint_);
  if (prob_mode_ == 0){
	  //mode : uniform distribution
	  caffe_rng_uniform(1, prob_activation_arg1_, prob_activation_arg2_, &curProb_);
  }
  else{
	  //mode : gaussian distribution
	  caffe_rng_gaussian(1, prob_activation_arg1_, prob_activation_arg2_, &curProb_);
	  if (curProb_ < 0){
		  if (prob_activation_arg1_ == 0) curProb_ = -curProb_;
		  else curProb_ = 0;
	  }
  }
  rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void StochasticDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	LOG(FATAL) << "Only GPU implementation available for stochastic dropout layer.";
}

template <typename Dtype>
void StochasticDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	LOG(FATAL) << "Only GPU implementation available for stochastic dropout layer.";
}


#ifdef CPU_ONLY
STUB_GPU(StochasticDropoutLayer);
#endif

INSTANTIATE_CLASS(StochasticDropoutLayer);
REGISTER_LAYER_CLASS(StochasticDropout);

}  // namespace caffe
