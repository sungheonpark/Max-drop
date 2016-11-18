/*************************************************
Caffe implementation of stochastic dropout proposed in
S. Park and N. Kwak, Analysis on the Dropout Effect in Convolutional Neural Networks, ACCV 2016
contact : sungheonpark@snu.ac.kr

**************************************************/

#ifndef CAFFE_STOCHASTIC_DROPOUT_LAYER_HPP_
#define CAFFE_STOCHASTIC_DROPOUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class StochasticDropoutLayer : public NeuronLayer<Dtype> {
 public:
	 explicit StochasticDropoutLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "StochasticDropout"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<unsigned int> rand_vec_;
  Dtype prob_activation_arg1_;
  Dtype prob_activation_arg2_;
  Dtype prob_layer_;
  unsigned int prob_layer_uint_;
  unsigned int prob_mode_;
  Dtype scale_;
  unsigned int uint_thres_;
  Dtype curProb_;
};

}  // namespace caffe

#endif  // CAFFE_STOCHASTIC_DROPOUT_LAYER_HPP_
