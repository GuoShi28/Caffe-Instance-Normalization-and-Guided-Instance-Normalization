#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/guided_instance_norm_layer.hpp"


namespace caffe {



	template <typename Dtype>
	void GuidedInstanceNormLayer<Dtype>::Reshape(
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
		Expand_Sigma_.Reshape(num_, ch_, h_, w_);
		Sigma2_.Reshape(num_, ch_, 1, 1);
		Expand_Sigma2_.Reshape(num_, ch_, h_, w_);
		XMinusMu_.Reshape(num_, ch_, h_, w_);
		Top_Buffer_.Reshape(num_, ch_, h_, w_);
		Bottom_Buffer_.Reshape(num_, ch_, h_, w_);
		Temp_.Reshape(num_, ch_, h_, w_);
	}


	template <typename Dtype>
	void GuidedInstanceNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		// output 1: mu
		Dtype* mutable_top_data_mu = top[0]->mutable_cpu_data();
		const Dtype* top_data_mu = top[0]->cpu_data();
		// output 2: var
		Dtype* mutable_top_data_var = top[1]->mutable_cpu_data();
		const Dtype* top_data_var = top[1]->cpu_data();
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
		//if (bottom[0] != top[0]) {
		//	caffe_copy(bottom[0]->count(), bottom_data, Bottom_Buffer_.mutable_cpu_data());
		//}

		// calculate mu 
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, 1, ch_*num_, imgSize_, (Dtype)(1. / imgSize_), weight_sum, bottom_data, (Dtype)0, mutable_mu); 
		//expend mu from num_ * ch_ to num_ * ch_ * w_ * h_  ---> output 1
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, ch_*num_, imgSize_, 1, (Dtype)(1), mu, weight_sum, (Dtype)0, mutable_top_data_mu);
		// y = x - mu (x) --> store in Bottom_Buffer_
		caffe_sub(count_, bottom_data, top_data_mu, Bottom_Buffer_.mutable_cpu_data());
		
		// calculate sigma, use expand_sigma2 as a buffer to restore 
		caffe_powx(count_, Bottom_Buffer_.cpu_data(), Dtype(2), mutable_expand_sigma2);
		// calculate sum of variance in the range of spatial
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, 1, ch_*num_, imgSize_, (Dtype)(1. / imgSize_), weight_sum, expand_sigma2, (Dtype)0, mutable_sigma2);
		
		// add epsilon to the var
		caffe_add_scalar(ch_*num_, Dtype(epsilon_), mutable_sigma2);
		caffe_powx(ch_*num_, sigma2, Dtype(0.5), mutable_sigma);

		// expand sigma from num_ * ch_ * 1 * 1 to num_ * ch_ * h_ * w_   ---> output 2
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, ch_*num_, imgSize_, 1, (Dtype)1, sigma, weight_sum, (Dtype)0, mutable_top_data_var);

	}

	template <typename Dtype>
	void GuidedInstanceNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* bottom_diff = bottom[0]->cpu_diff();
		Dtype* mutable_bottom_diff = bottom[0]->mutable_cpu_diff();
		// output 1: mu
		const Dtype* top_mu_diff = top[0]->cpu_diff();
		// output 2: ver
		const Dtype* top_var_diff = top[1]->cpu_diff();
		const Dtype* top_data_var = Top_Buffer_.cpu_data();
		// cal sum buffer
		const Dtype* weight_sum = SumMatrix_.cpu_data();
		
		// d(E(y))/d(x) = d(E(y))/d(y) * d(y)/d(x) = top_mu_diff * d(mu(x)) / d(x) + top_var_diff * d(var(x)) / d(x)
		// ----- calculate the diff of mu (output 1) : top_mu_diff * d(mu(x)) / d(x) -----
		// top_mu_diff * d(mu(x)) / d(x) = mu(top_mu_diff)
		// sum 
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, 1, ch_*num_, imgSize_, (Dtype)1, weight_sum, top_mu_diff, (Dtype)0, Mu_.mutable_cpu_data());
		// extend
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, ch_*num_, imgSize_, 1, (Dtype)(1), Mu_.cpu_data(), weight_sum, (Dtype)0, mutable_bottom_diff);

		// ----- calculate the diff of var (output 2) : top_var_diff * d(var(x)) / d(x) -----
		// top_var_diff * d(var(x)) / d(x) = 2 * mu(top_var_diff /cdot (x - mu(x)))
		// top_var_diff /cdot (x - mu(x))
		caffe_mul(count_, top_var_diff, Bottom_Buffer_.cpu_data(), Temp_.mutable_cpu_data());
		// mu(top_var_diff /cdot (x - mu(x)))
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, 1, ch_*num_, imgSize_, (Dtype)1, weight_sum, Temp_.cpu_data(), (Dtype)0, Mu_.mutable_cpu_data());
		// extend
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, ch_*num_, imgSize_, 1, (Dtype)(1), Mu_.cpu_data(), weight_sum, (Dtype)0, Temp_.mutable_cpu_data());
		// mu(top_mu_diff) + 2 * mu(top_var_diff /cdot (x - mu(x)))
		caffe_axpy(count_, Dtype(2), Temp_.cpu_data(), mutable_bottom_diff);
		// bottom_diff / img_size_
		caffe_cpu_scale(count_, Dtype(1. / imgSize_), bottom_diff, mutable_bottom_diff);
	}



#ifdef CPU_ONLY
	STUB_GPU(BlurKernel);
#endif

	INSTANTIATE_CLASS(GuidedInstanceNormLayer);
	REGISTER_LAYER_CLASS(GuidedInstanceNorm);

}  // namespace caffe