/********************************
*   Caffe implementation of Spatial Dropout, which is proposed in
*   Tompson et al., 'Efficient object localization using convolutional networks', CVPR 2015
*****************************/

#ifndef CAFFE_SPATIAL_DROPOUT_LAYER_HPP_
#define CAFFE_SPATIAL_DROPOUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {
	template <typename Dtype>
	class SpatialDropoutLayer : public NeuronLayer<Dtype> {
	public:
		explicit SpatialDropoutLayer(const LayerParameter& param)
			: NeuronLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "SpatialDropout"; }

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
		Dtype threshold_;
		Dtype scale_;
		unsigned int uint_thres_;
	};

}  // namespace caffe

#endif  // CAFFE_SPATIAL_DROPOUT_LAYER_HPP_
