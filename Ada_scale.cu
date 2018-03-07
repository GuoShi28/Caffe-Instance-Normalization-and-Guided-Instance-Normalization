#include <vector>

#include "caffe/layers/instance_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


	template <typename Dtype>
	void AdaScaleLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		// input data
		const Dtype* bottom_data = bottom[0]->gpu_data();  // BN data
		const Dtype* bottom_mu_data = bottom[1]->gpu_data(); // mu data from guided image
		const Dtype* bottom_var_data = bottom[2]->gpu_data(); // var data from guided image
		// output data
		const Dtype* top_data = top[0]->gpu_data();
		Dtype* mutable_top_data = top[0]->mutable_gpu_data();

		// ----- top_data = var_data * bottom_data + mu_data -----
		caffe_mul(bottom[0]->count(), bottom_data, bottom_var_data, mutable_top_data);
		caffe_add(bottom[0]->count(), bottom_mu_data, top_data, mutable_top_data);
	}

	template <typename Dtype>
	void AdaScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		// top diff: input
		const Dtype* top_diff = top[0]->gpu_diff();
		// bottom diff: output
		const Dtype* bottom_diff = bottom[0]->gpu_diff();
		Dtype* mutable_bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* bottom_mu_diff = bottom[1]->gpu_diff();
		Dtype* mutable_bottom_mu_diff = bottom[1]->mutable_gpu_diff();
		const Dtype* bottom_var_diff = bottom[2]->gpu_diff();
		Dtype* mutable_bottom_var_diff = bottom[2]->mutable_gpu_diff();
		// bottom data: input 
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* bottom_mu_data = bottom[1]->gpu_data();
		const Dtype* bottom_var_data = bottom[2]->gpu_data();

		// ----- data diff -----
		caffe_mul(bottom[0]->count(), top_diff, bottom_var_data, mutable_bottom_diff);
		// ----- mu diff -----
		caffe_copy(bottom[0]->count(), top_diff, mutable_bottom_mu_diff);
		// ----- var diff -----
		caffe_mul(bottom[0]->count(), top_diff, bottom_data, mutable_bottom_var_diff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(AdaScaleLayer);

}  // namespace caffe
