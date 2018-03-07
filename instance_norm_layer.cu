#include <vector>

#include "caffe/layers/instance_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


	template <typename Dtype>
	void InstanceNormLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* mutable_top_data = top[0]->mutable_gpu_data();
		const Dtype* top_data = top[0]->gpu_data();
		// define by GS
		const Dtype* weight_sum = SumMatrix_.gpu_data();

		Dtype* mutable_mu = Mu_.mutable_gpu_data();
		const Dtype* mu = Mu_.gpu_data();
		Dtype* mutable_expand_mu = Expand_Mu_.mutable_gpu_data();
		const Dtype* expand_mu = Expand_Mu_.gpu_data();

		Dtype* mutable_sigma = Sigma_.mutable_gpu_data();
		const Dtype* sigma = Sigma_.gpu_data();
		Dtype* mutable_expand_sigma = Expand_Sigma_.mutable_gpu_data();
		const Dtype* expand_sigma = Expand_Sigma_.gpu_data();

		Dtype* mutable_sigma2 = Sigma2_.mutable_gpu_data();
		const Dtype* sigma2 = Sigma2_.gpu_data();
		Dtype* mutable_expand_sigma2 = Expand_Sigma2_.mutable_gpu_data();
		const Dtype* expand_sigma2 = Expand_Sigma2_.gpu_data();

		Dtype* mutable_x_minus_mu = XMinusMu_.mutable_gpu_data();
		const Dtype* x_minus_mu = XMinusMu_.gpu_data();

		// top = bottom
		//if (bottom[0] != top[0]) {
		caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
		//}

		// calculate mu 
		caffe_gpu_gemm(CblasNoTrans, CblasTrans, 1, ch_*num_, imgSize_, Dtype(1. / imgSize_), SumMatrix_.gpu_data(), bottom[0]->gpu_data(), Dtype(0), Mu_.mutable_gpu_data());
		// expand mu from num_ * ch_ to num_ * ch_ * w_ * h_     // y = x - mu(x)
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, ch_*num_, imgSize_, 1, Dtype(-1), Mu_.gpu_data(), SumMatrix_.gpu_data(), Dtype(1), top[0]->mutable_gpu_data());
		// calculate sigma, use expand_sigma2 as a buffer to restore 
		caffe_gpu_powx(count_, top[0]->gpu_data(), Dtype(2), mutable_expand_sigma2);
		// calculate sum of variance in the range of spatial
		caffe_gpu_gemm(CblasNoTrans, CblasTrans, 1, ch_*num_, imgSize_, (Dtype)(1. / imgSize_), SumMatrix_.gpu_data(), Expand_Sigma2_.gpu_data(), (Dtype)0, Sigma2_.mutable_gpu_data());
		// add epsilon to the var
		caffe_gpu_add_scalar(ch_*num_, Dtype(epsilon_), Sigma2_.mutable_gpu_data());
		caffe_gpu_powx(ch_*num_, Sigma2_.gpu_data(), Dtype(0.5), Sigma_.mutable_gpu_data());
		// expand sigma from num_ * ch_ * 1 * 1 to num_ * ch_ * h_ * w_
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, ch_*num_, imgSize_, 1, (Dtype)1, Sigma_.gpu_data(), SumMatrix_.gpu_data(), (Dtype)0, Expand_Sigma_.mutable_gpu_data());
		// y = (x - mu(x)) / (sqrt(var(X)+eps)
		caffe_gpu_div(count_, top[0]->gpu_data(), Expand_Sigma_.gpu_data(), top[0]->mutable_gpu_data());
		// store for backpropogate
		caffe_copy(count_, top[0]->gpu_data(), Top_Buffer_.mutable_gpu_data());

	}

	template <typename Dtype>
	void InstanceNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* bottom_diff = bottom[0]->gpu_diff();
		Dtype* mutable_bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* top_data = Top_Buffer_.gpu_data();
		const Dtype* weight_sum = SumMatrix_.gpu_data();
		// y = (x - mu(x)) / var(x)
		// dE(Y)/dX =
		//   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
		//     ./ sqrt(var(X) + eps)

		// ----- sum(dE/dY \cdot Y) -----
		caffe_gpu_mul(count_, top_data, top_diff, mutable_bottom_diff);
		// sum 
		caffe_gpu_gemm(CblasNoTrans, CblasTrans, 1, ch_*num_, imgSize_, (Dtype)1, weight_sum, bottom_diff, (Dtype)0, Mu_.mutable_gpu_data());
		// extend
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, ch_*num_, imgSize_, 1, (Dtype)(1), Mu_.gpu_data(), weight_sum, (Dtype)0, mutable_bottom_diff);

		// ----- sum(dE/dY \cdot Y) \cdot Y -----
		caffe_gpu_mul(count_, top_data, bottom_diff, mutable_bottom_diff);

		// ----- sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y -----
		// sum 
		caffe_gpu_gemm(CblasNoTrans, CblasTrans, 1, ch_*num_, imgSize_, (Dtype)1, weight_sum, top_diff, (Dtype)0, Mu_.mutable_gpu_data());
		// extend
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, ch_*num_, imgSize_, 1, (Dtype)(1), Mu_.gpu_data(), weight_sum, (Dtype)1, mutable_bottom_diff);

		// ----- dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y -----
		caffe_gpu_axpby(count_, Dtype(1), top_diff, Dtype(-1. / imgSize_), mutable_bottom_diff);

		// ----- {dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y} / sqrt(var(X) + eps) -----
		caffe_gpu_div(count_, bottom_diff, Expand_Sigma_.gpu_data(), mutable_bottom_diff);


	}

	INSTANTIATE_LAYER_GPU_FUNCS(InstanceNormLayer);

}  // namespace caffe
