#include <torch/extension.h>
#include <sinkhorn.h>


void sinkhorn_log_kernel(torch::Tensor d_k, torch::Tensor add, torch::Tensor res, size_t N) {
    float *d_k_ptr = (float*)d_k.data_ptr();
    float *add_ptr = (float*)add.data_ptr();
    float *res_ptr = (float*)res.data_ptr();
    sinkhorn_log_glue(d_k_ptr, add_ptr, res_ptr, N);
}

void matrix_transpose_kernel(torch::Tensor d_t_k, torch::Tensor d_k, size_t N){
    float *d_t_k_ptr = (float*)d_t_k.data_ptr();
    float *d_k_ptr = (float*)d_k.data_ptr();
    matrix_transpose_glue(d_t_k_ptr, d_k_ptr, N);
    // return d_t_k;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sinkhorn_log_kernel", &sinkhorn_log_kernel, "");
    m.def("matrix_transpose_kernel", &matrix_transpose_kernel, "");
}