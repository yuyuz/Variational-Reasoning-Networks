#include "graph_inner_product.h"
#include "tensor/gpu_handle.h"
#include "tensor/gpu_unary_functor.h"

namespace gnn
{

template<typename Dtype>
__global__ void SetValKernel(Dtype *dst, Dtype *src, int* entity_idx, int* sample_idx, int cols, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        dst[sample_idx[i] * cols + entity_idx[i]] = src[i];
    }
}

template<typename Dtype>
void SetVal(DTensor<GPU, Dtype>& src, int* entity_idx, int* sample_idx, DTensor<GPU, Dtype>& dst)
{
    int thread_num = c_uCudaThreadNum;
	if (src.shape.Count() < thread_num)
		thread_num = src.shape.Count();
    int blocksPerGrid = (src.shape.Count() + thread_num - 1) / thread_num;

    SetValKernel <<< blocksPerGrid, thread_num, 0, cudaStreamPerThread >>>(dst.data->ptr, src.data->ptr, entity_idx, sample_idx, dst.cols(), src.shape.Count());
}

template void SetVal(DTensor<GPU, float>& src, int* entity_idx, int* sample_idx, DTensor<GPU, float>& dst);
template void SetVal(DTensor<GPU, double>& src, int* entity_idx, int* sample_idx, DTensor<GPU, double>& dst);

template<typename Dtype>
__global__ void BpErrorKernel(Dtype *dst, Dtype *src, int* entity_idx, int* sample_idx, int cols, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        dst[i] = src[sample_idx[i] * cols + entity_idx[i]];
    }
}

template<typename Dtype>
void BpError(DTensor<GPU, Dtype>& grad_out, int* entity_idx, int* sample_idx, DTensor<GPU, Dtype>& cur_grad)
{
    int thread_num = c_uCudaThreadNum;
	if (cur_grad.shape.Count() < thread_num)
		thread_num = cur_grad.shape.Count();
    int blocksPerGrid = (cur_grad.shape.Count() + thread_num - 1) / thread_num;

    BpErrorKernel <<< blocksPerGrid, thread_num, 0, cudaStreamPerThread >>>(cur_grad.data->ptr, grad_out.data->ptr, entity_idx, sample_idx, grad_out.cols(), cur_grad.shape.Count());
}

template void BpError(DTensor<GPU, float>& grad_out, int* entity_idx, int* sample_idx, DTensor<GPU, float>& cur_grad);
template void BpError(DTensor<GPU, double>& grad_out, int* entity_idx, int* sample_idx, DTensor<GPU, double>& cur_grad);

}