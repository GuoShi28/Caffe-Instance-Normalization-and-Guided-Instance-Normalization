#ifndef CAFFE_INSTANCE_NORM_LAYER_HPP_
#define CAFFE_INSTANCE_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


	template <typename Dtype>
	class InstanceNormLayer : public Layer<Dtype> {
	public:
		explicit InstanceNormLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}

		virtual inline const char* type() const { return "InstanceNorm"; }
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int num_, ch_, h_, w_;
	int imgSize_, count_;
	float epsilon_;
	Blob<Dtype> SumMatrix_;
	Blob<Dtype> Mu_, Sigma_, Expand_Mu_, Expand_Sigma_, Sigma2_, Expand_Sigma2_;
	Blob<Dtype> XMinusMu_;
	Blob<Dtype> Top_Buffer_;
	};

}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
