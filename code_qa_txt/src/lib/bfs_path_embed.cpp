#include "bfs_path_embed.h"
#include "var_sample.h"
#include "global.h"

namespace gnn
{

template<typename mode, typename Dtype>
BfsPathEmbed<mode, Dtype>::BfsPathEmbed(std::string _name, PropErr _properr) 
		: Factor(_name, _properr)
{
	ptr_in_info = new std::vector<Node*>();
	ptr_in_cnt = new std::vector<int>();
}

template<typename mode, typename Dtype>
void BfsPathEmbed<mode, Dtype>::GetOutInfo(std::vector<Node*>& out_node_info, std::vector<int>& out_num_node)
{
	auto& in_node_info = *ptr_in_info;
	auto& in_num_node = *ptr_in_cnt;

	src_positions.clear();
	rel_types.clear();

	int cur_in_pos = 0;
	for (size_t i = 0; i < in_num_node.size(); ++i)
	{
		std::map<int, int> next_nodes;
		for (int j = 0; j < in_num_node[i]; ++j, ++cur_in_pos)
		{
			auto* src_node = in_node_info[cur_in_pos];

			for (auto& edge : src_node->adj_list)
			{
				if (!next_nodes.count(edge.second->idx))
				{
					next_nodes[edge.second->idx] = out_node_info.size();
					out_node_info.push_back(edge.second);
					src_positions.push_back(std::set<int>());
					rel_types.push_back(std::set<int>());					
				}
				auto out_pos = next_nodes[edge.second->idx];
				src_positions[out_pos].insert(cur_in_pos);
				rel_types[out_pos].insert(edge.first);
			}
		}
		out_num_node[i] = next_nodes.size();
	}
	assert(cur_in_pos == (int)in_node_info.size());
}

template<typename mode, typename Dtype>
void BfsPathEmbed<mode, Dtype>::ConstructRelSp(std::vector<Node*>& out_node_info, std::vector<int>& out_num_node, size_t rel_nums)
{
	uint nnz_rel = 0;
	for (size_t i = 0; i < rel_types.size(); ++i)
		nnz_rel += rel_types[i].size();
	cpu_rel_mat.Reshape({out_node_info.size(), rel_nums});
	cpu_rel_mat.ResizeSp(nnz_rel, out_node_info.size() + 1);

	nnz_rel = 0;
	for (size_t i = 0; i < out_node_info.size(); ++i)
	{
		cpu_rel_mat.data->row_ptr[i] = nnz_rel;
		for (auto rel : rel_types[i])
		{
			cpu_rel_mat.data->val[nnz_rel] = 1.0;
			cpu_rel_mat.data->col_idx[nnz_rel] = rel;
			nnz_rel += 1;
		}
	}
	cpu_rel_mat.data->row_ptr[out_node_info.size()] = nnz_rel;
	assert((int)nnz_rel == cpu_rel_mat.data->nnz);
	rel_mat.CopyFrom(cpu_rel_mat);
}

template<typename mode, typename Dtype>
void BfsPathEmbed<mode, Dtype>::ConstructNodeSp(std::vector<Node*>& out_node_info, std::vector<int>& out_num_node, size_t num_in)
{
	uint nnz_node = 0;
	for (size_t i = 0; i < src_positions.size(); ++i)
		nnz_node += src_positions[i].size();
	cpu_node_mat.Reshape({out_node_info.size(), num_in});
	cpu_node_mat.ResizeSp(nnz_node, out_node_info.size() + 1);

	nnz_node = 0;
	for (size_t i = 0; i < out_node_info.size(); ++i)
	{
		cpu_node_mat.data->row_ptr[i] = nnz_node;
		for (auto pos : src_positions[i])
		{
			cpu_node_mat.data->val[nnz_node] = 1.0 / src_positions[i].size();
			cpu_node_mat.data->col_idx[nnz_node] = pos;
			nnz_node += 1;
		}
	}
	cpu_node_mat.data->row_ptr[out_node_info.size()] = nnz_node;
	assert((int)nnz_node == cpu_node_mat.data->nnz);
	node_mat.CopyFrom(cpu_node_mat);
}

template<typename mode, typename Dtype>
void BfsPathEmbed<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 			std::vector< std::shared_ptr<Variable> >& outputs, 
									 Phase phase)
{
	ASSERT(operands.size() == 6 || operands.size() == 4, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 3, "unexpected output size for " << StrType()); 

	auto& samples = *(dynamic_cast< SampleVar* >(operands[0].get())->samples);
	auto& rel_embed = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->value;
	auto& w_path_recur = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[2].get())->value;	

	auto& output = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	auto* var_node_info = dynamic_cast< VectorVar<Node*>* >(outputs[1].get());
	auto* var_num_node = dynamic_cast< VectorVar<int>* >(outputs[2].get());
	if (!var_node_info->vec)
		var_node_info->vec = new std::vector<Node*>();
	if (!var_num_node->vec)
		var_num_node->vec = new std::vector<int>();
	auto& out_node_info = *(var_node_info->vec);
	auto& out_num_node = *(var_num_node->vec);	

	if ((int)operands.size() == 4)
	{
		auto& st_nodes = *(dynamic_cast< VectorVar<Node*>* >(operands[3].get())->vec);

		ptr_in_cnt->resize(samples.size());
		ptr_in_info->clear();
		for (size_t i = 0; i < samples.size(); ++i)
    	{
			assert(st_nodes[i]);			
			ptr_in_info->push_back(st_nodes[i]);
			(*ptr_in_cnt)[i] = 1;
		}
	} else {
		ptr_in_info = dynamic_cast< VectorVar<Node*>* >(operands[4].get())->vec;
		ptr_in_cnt = dynamic_cast< VectorVar<int>* >(operands[5].get())->vec;
	}

	out_node_info.clear();
	out_num_node.resize(samples.size());

	GetOutInfo(out_node_info, out_num_node);

	ConstructRelSp(out_node_info, out_num_node, rel_embed.rows());
	output.MM(rel_mat, rel_embed, Trans::N, Trans::N, 1.0, 0.0);

	if ((int)operands.size() == 6)
	{
		auto& prev_embed = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[3].get())->value;
		node_trans.MM(prev_embed, w_path_recur, Trans::N, Trans::N, 1.0, 0.0);

		ConstructNodeSp(out_node_info, out_num_node, ptr_in_info->size());
		output.MM(node_mat, node_trans, Trans::N, Trans::N, 1.0, 1.0);
	}
}

template<typename mode, typename Dtype>
void BfsPathEmbed<mode, Dtype>::Backward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< bool >& isConst, 
						 			std::vector< std::shared_ptr<Variable> >& outputs)
{
	ASSERT(operands.size() == 6 || operands.size() == 4, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 3, "unexpected output size for " << StrType());

	auto grad_out = dynamic_cast<DTensorVar<mode, Dtype>*>(outputs[0].get())->grad.Full();

	auto rel_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[1].get())->grad.Full();

	if ((int)operands.size() == 4) // first hop
	{
		rel_grad.MM(rel_mat, grad_out, Trans::T, Trans::N, 1.0, 1.0);
	} else {
		node_grad.MM(node_mat, grad_out, Trans::T, Trans::N, 1.0, 0.0);

		auto& prev_embed = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[3].get())->value;
		auto prev_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[3].get())->grad.Full();
		auto& w_path_recur = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[2].get())->value;
		auto w_grad = dynamic_cast<DTensorVar<mode, Dtype>*>(operands[2].get())->grad.Full();

		prev_grad.MM(node_grad, w_path_recur, Trans::N, Trans::T, 1.0, 1.0);
		w_grad.MM(prev_embed, node_grad, Trans::T, Trans::N, 1.0, 1.0);
	}
}

INSTANTIATE_CLASS(BfsPathEmbed)

}
