#ifndef GRAPH_INNER_PRODUCT_H
#define GRAPH_INNER_PRODUCT_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"
#include "util/fmt.h"

#ifdef USE_GPU
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

namespace gnn
{

template<typename Dtype>
void SetVal(DTensor<CPU, Dtype>& src, int* entity_idx, int* sample_idx, DTensor<CPU, Dtype>& dst);
template<typename Dtype>
void SetVal(DTensor<GPU, Dtype>& src, int* entity_idx, int* sample_idx, DTensor<GPU, Dtype>& dst);

template<typename Dtype>
void BpError(DTensor<CPU, Dtype>& grad_out, int* entity_idx, int* sample_idx, DTensor<CPU, Dtype>& cur_grad);
template<typename Dtype>
void BpError(DTensor<GPU, Dtype>& grad_out, int* entity_idx, int* sample_idx, DTensor<GPU, Dtype>& cur_grad);

template<typename mode, typename Dtype>
class GraphInnerProduct : public Factor
{
public:
	static std::string StrType()
	{
		return "GraphInnerProduct";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;

	OutType CreateOutVar()
	{
        auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< DTensorVar<mode, Dtype> >(out_name);
	}

	GraphInnerProduct(std::string _name, int _num_entities, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;

    int num_entities;
	DTensor<mode, Dtype> tmp_out;
#ifdef USE_GPU
	thrust::host_vector<int> entity_idx, sample_idx;
	thrust::device_vector<int> gpu_entity_idx, gpu_sample_idx;
#else
	std::vector<int> entity_idx, sample_idx;
#endif

	int* ptr_entity;
	int* ptr_sample;
};

}

#endif
