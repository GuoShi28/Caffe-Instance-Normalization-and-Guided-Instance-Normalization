#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/instance_norm_layer.hpp"


namespace caffe {



	template <typename Dtype>
	void InstanceNormLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		h_ = bottom[0]->height();
		w_ = bottom[0]->width();
		ch_ = bottom[0]->channels();
		num_ = bottom[0]->num();
		imgSize_ = h_ * w_;
		count_ = imgSize_ * ch_ * num_;
		epsilon_ = 0.0001;
		top[0]->Reshape(num_, ch_, h_, w_);
		// define by GS
		SumMatrix_.Reshape(1, 1, 1, imgSize_);
		caffe_set(imgSize_, (Dtype)1, SumMatrix_.mutable_cpu_data());

		Mu_.Reshape(num_, ch_, 1, 1);
		Expand_Mu_.Reshape(num_, ch_, h_, w_);
		Sigma_.Reshape(num_, ch_, 1, 1);
		//Expand_Sigma_.Reshape(num_, ch_, h_, w_);
		Expand_Sigma_.ReshapeLike(*bottom[0]);
		Sigma2_.Reshape(num_, ch_, 1, 1);
		//Expand_Sigma2_.Reshape(num_, ch_, h_, w_);
		Expand_Sigma2_.ReshapeLike(*bottom[0]);
		XMinusMu_.Reshape(num_,ch_, h_, w_);
		Top_Buffer_.Reshape(num_, ch_, h_, w_);
	}


	template <typename Dtype>
	void InstanceNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* mutable_top_data = top[0]->mutable_cpu_data();
		const Dtype* top_data = top[0]->cpu_data();
		// define by GS
		const Dtype* weight_sum = SumMatrix_.cpu_data();

		Dtype* mutable_mu = Mu_.mutable_cpu_data();
		const Dtype* mu = Mu_.cpu_data();
		Dtype* mutable_expand_mu = Expand_Mu_.mutable_cpu_data();
		const Dtype* expand_mu = Expand_Mu_.cpu_data();

		Dtype* mutable_sigma = Sigma_.mutable_cpu_data();
		const Dtype* sigma = Sigma_.cpu_data();
		Dtype* mutable_expand_sigma = Expand_Sigma_.mutable_cpu_data();
		const Dtype* expand_sigma = Expand_Sigma_.cpu_data();

		Dtype* mutable_sigma2 = Sigma2_.mutable_cpu_data();
		const Dtype* sigma2 = Sigma2_.cpu_data();
		Dtype* mutable_expand_sigma2 = Expand_Sigma2_.mutable_cpu_data();
		const Dtype* expand_sigma2 = Expand_Sigma2_.cpu_data();

		Dtype* mutable_x_minus_mu = XMinusMu_.mutable_cpu_data();
		const Dtype* x_minus_mu = XMinusMu_.cpu_data();

		// top = bottom
		if (bottom[0] != top[0]) {
			caffe_copy(bottom[0]->count(), bottom_data, mutable_top_data);
		}

		// calculate mu 
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, 1, ch_*num_, imgSize_, (Dtype)(1. / imgSize_), weight_sum, bottom_data, (Dtype)0, mutable_mu); 
		// expand mu from num_ * ch_ to num_ * ch_ * w_ * h_     // y = x - mu(x)
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, ch_*num_, imgSize_, 1, (Dtype)(-1), mu, weight_sum, (Dtype)1, mutable_top_data);
		
		// calculate sigma, use expand_sigma2 as a buffer to restore 
		caffe_powx(count_, top_data, Dtype(2), mutable_expand_sigma2);
		// calculate sum of variance in the range of spatial
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, 1, ch_*num_, imgSize_, (Dtype)(1. / imgSize_), weight_sum, expand_sigma2, (Dtype)0, mutable_sigma2);
		
		// add epsilon to the var
		caffe_add_scalar(ch_*num_, Dtype(epsilon_), mutable_sigma2);
		caffe_powx(ch_*num_, sigma2, Dtype(0.5), mutable_sigma);

		// expand sigma from num_ * ch_ * 1 * 1 to num_ * ch_ * h_ * w_
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, ch_*num_, imgSize_, 1, (Dtype)1, sigma, weight_sum, (Dtype)0, mutable_expand_sigma);

		// y = (x - mu(x)) / (sqrt(var(X)+eps)
		caffe_div(count_, top_data, expand_sigma, mutable_top_data);

		// store for backpropogate
		caffe_copy(count_, top_data, Top_Buffer_.mutable_cpu_data());
	}

	template <typename Dtype>
	void InstanceNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* bottom_diff = bottom[0]->cpu_diff();
		Dtype* mutable_bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype* top_data = Top_Buffer_.cpu_data();
		const Dtype* weight_sum = SumMatrix_.cpu_data();
		// y = (x - mu(x)) / var(x)
		// dE(Y)/dX =
		//   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
		//     ./ sqrt(var(X) + eps)

		// ----- sum(dE/dY \cdot Y) -----
		caffe_mul(count_, top_data, top_diff, mutable_bottom_diff);
		// sum 
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, 1, ch_*num_, imgSize_, (Dtype)1, weight_sum, bottom_diff, (Dtype)0, Mu_.mutable_cpu_data());
		// extend
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, ch_*num_, imgSize_, 1, (Dtype)(1), Mu_.cpu_data(), weight_sum, (Dtype)0, mutable_bottom_diff);

		// ----- sum(dE/dY \cdot Y) \cdot Y -----
		caffe_mul(count_, top_data, bottom_diff, mutable_bottom_diff);

		// ----- sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y -----
		// sum 
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, 1, ch_*num_, imgSize_, (Dtype)1, weight_sum, top_diff, (Dtype)0, Mu_.mutable_cpu_data());
		// extend
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, ch_*num_, imgSize_, 1, (Dtype)(1), Mu_.cpu_data(), weight_sum, (Dtype)1, mutable_bottom_diff);

		// ----- dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y -----
		caffe_cpu_axpby(count_, Dtype(1), top_diff, Dtype(-1. / imgSize_), mutable_bottom_diff);

		// ----- {dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y} / sqrt(var(X) + eps) -----
		caffe_div(count_, bottom_diff, Expand_Sigma_.cpu_data(), mutable_bottom_diff);
	}



#ifdef CPU_ONLY
	STUB_GPU(BlurKernel);
#endif

	INSTANTIATE_CLASS(InstanceNormLayer);
	REGISTER_LAYER_CLASS(InstanceNorm);

}  // namespace caffe