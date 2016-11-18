/*************************************************
Caffe implementation of Max-drop layer proposed in
S. Park and N. Kwak, Analysis on the Dropout Effect in Convolutional Neural Networks, ACCV 2016
contact : sungheonpark@snu.ac.kr

**************************************************/

#ifndef CAFFE_MAX_DROP_LAYER_HPP_
#define CAFFE_MAX_DROP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	* @brief Supress the most activated neurons with probability p.
	*        
	*/
	template <typename Dtype>
	class MaxDropLayer : public Layer<Dtype> {
	public:
		explicit MaxDropLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "MaxDrop"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		//supressing kernel size (default : 1, 1)
		int kernel_h_, kernel_w_;
		//Max-drop option, 1: Feature-wise Max-drop, 2: Channel-wise Max-drop (default: 1)
		int dim_;
		//probability of suppression
		float prob_;
		//integer prob value (used for GPU forward)
		unsigned int prob_int_;
		//contains off or not for each index
		Blob<unsigned int> offIndex_;
		//weight blob
		Blob<Dtype> mask_;
		/// the scale for undropped inputs at train time
		Dtype scale_;
	};

}  // namespace caffe

#endif  // CAFFE_MAX_DROP_LAYER_HPP_
