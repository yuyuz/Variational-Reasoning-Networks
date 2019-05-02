#include "graph_inner_product.h"
#include "var_sample.h"
#include "global.h"

namespace gnn
{

template<typename Dtype>
void SetVal(DTensor<CPU, Dtype>& src, int* entity_idx, int* sample_idx, DTensor<CPU, Dtype>& dst)
{
	for (size_t i = 0; i < src.shape.Count(); ++i)
	{
		auto row = sample_idx[i], col = entity_idx[i];
		dst.data->ptr[row * dst.cols() + col] = src.data->ptr[i];
	}
}

template<typename Dtype>
void BpError(DTensor<CPU, Dtype>& grad_out, int* entity_idx, int* sample_idx, DTensor<CPU, Dtype>& cur_grad)
{
	for (size_t i = 0; i < cur_grad.shape.Count(); ++i)
	{
		auto row = sample_idx[i], col = entity_idx[i];
		cur_grad.data->ptr[i] = grad_out.data->ptr[row * grad_out.cols() + col];
	}
}

template<typename mode, typename Dtype>
GraphInnerProduct<mode, Dtype>::GraphInnerProduct(std::string _name, int _num_entities, PropErr _properr) 
		: Factor(_name, _properr), num_entities(_num_entities)
{

}

template<typename mode, typename Dtype>
void GraphInnerProduct<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
									 Phase phase)
{
	ASSERT(operands.size() == 4, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 
		
	auto& q_embed = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& ans_embed = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;
	auto& node_info = *(dynamic_cast< VectorVar<Node*>* >(operands[2].get())->vec);
	auto& sample_nodecnt = *(dynamic_cast< VectorVar<int>* >(operands[3].get())->vec);	

	entity_idx.resize(node_info.size());
	sample_idx.resize(node_info.size());

	tmp_out.Reshape({node_info.size(), (size_t)1});
	size_t row_idx = 0;
	for (size_t i = 0; i < sample_nodecnt.size(); ++i)
	{
		auto row_cnt = sample_nodecnt[i];
		auto cur_q = q_embed.GetRowRef(i, 1);
		auto cur_ans = ans_embed.GetRowRef(row_idx, row_cnt);
		auto cur_out = tmp_out.GetRowRef(row_idx, row_cnt);

		cur_out.MM(cur_ans, cur_q, Trans::N, Trans::T, 1.0, 0.0);

		for (size_t j = row_idx; j < row_idx + row_cnt; ++j)
			sample_idx[j] = i;
		row_idx += row_cnt;
	}
	assert(row_idx == node_info.size());

	for (size_t i = 0; i < node_info.size(); ++i)
		entity_idx[i] = node_info[i]->idx;

#ifdef USE_GPU
	if (mode::type == MatMode::cpu)
	{
		ptr_entity = thrust::raw_pointer_cast(entity_idx.data());
		ptr_sample = thrust::raw_pointer_cast(sample_idx.data());
	} else {
		gpu_entity_idx = entity_idx;
		gpu_sample_idx = sample_idx;

		ptr_entity = thrust::raw_pointer_cast(gpu_entity_idx.data());
		ptr_sample = thrust::raw_pointer_cast(gpu_sample_idx.data());
	}
#else
	ptr_entity = entity_idx.data();
	ptr_sample = sample_idx.data();
#endif

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	output.Reshape({q_embed.rows(), (size_t)num_entities});
	output.Zeros();
	SetVal(tmp_out, ptr_entity, ptr_sample, output);
}

template<typename mode, typename Dtype>
void GraphInnerProduct<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 4, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 
	auto& sample_nodecnt = *(dynamic_cast< VectorVar<int>* >(operands[3].get())->vec);	

	auto grad_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();
	BpError(grad_out, ptr_entity, ptr_sample, tmp_out);

	auto& q_embed = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->value;
	auto& ans_embed = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;

	auto q_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[0].get())->grad.Full();
	auto ans_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->grad.Full();
	size_t row_idx = 0;
	for (size_t i = 0; i < sample_nodecnt.size(); ++i)
	{
		auto row_cnt = sample_nodecnt[i];
		auto cur_q = q_embed.GetRowRef(i, 1);
		auto cur_ans = ans_embed.GetRowRef(row_idx, row_cnt);

		auto cur_grad = tmp_out.GetRowRef(row_idx, row_cnt);

		auto cur_q_grad = q_grad.GetRowRef(i, 1);
		auto cur_ans_grad = ans_grad.GetRowRef(row_idx, row_cnt);

		cur_ans_grad.MM(cur_grad, cur_q, Trans::N, Trans::N, 1.0, 1.0);
		cur_q_grad.MM(cur_grad, cur_ans, Trans::T, Trans::N, 1.0, 1.0);
		
		row_idx += row_cnt;
	}
	assert(row_idx == tmp_out.shape.Count());
}

INSTANTIATE_CLASS(GraphInnerProduct)

}
