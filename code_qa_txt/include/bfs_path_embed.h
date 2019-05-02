#ifndef BFS_PATH_EMBED_H
#define BFS_PATH_EMBED_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"
#include "var_sample.h"

namespace gnn
{

template<typename mode, typename Dtype>
class BfsPathEmbed : public Factor
{
public:
	static std::string StrType()
	{
		return "BfsPathEmbed";
	}

	using OutType = std::tuple< std::shared_ptr< DTensorVar<mode, Dtype> >, 
								std::shared_ptr< VectorVar<Node*> >, 
								std::shared_ptr< VectorVar<int> > >;

	OutType CreateOutVar()
	{
		auto o0 = std::make_shared< DTensorVar<mode, Dtype> >( fmt::sprintf("%s:out_0", this->name) );
		auto o1 = std::make_shared< VectorVar<Node*> >( fmt::sprintf("%s:out_1", this->name) );
		auto o2 = std::make_shared< VectorVar<int> >( fmt::sprintf("%s:out_2", this->name) );

		return std::make_tuple(o0, o1, o2);
	}

	BfsPathEmbed(std::string _name, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						std::vector< bool >& isConst, 
						std::vector< std::shared_ptr<Variable> >& outputs) override;

	void GetOutInfo(std::vector<Node*>& out_node_info, std::vector<int>& out_num_node);

	void ConstructRelSp(std::vector<Node*>& out_node_info, std::vector<int>& out_num_node, size_t rel_nums);
	void ConstructNodeSp(std::vector<Node*>& out_node_info, std::vector<int>& out_num_node, size_t num_in);

	SpTensor<CPU, Dtype> cpu_rel_mat, cpu_node_mat;
	SpTensor<mode, Dtype> rel_mat, node_mat;
	std::vector<Node*>* ptr_in_info;
	std::vector<int>* ptr_in_cnt;	

	std::vector< std::set<int> > src_positions;
	std::vector< std::set<int> > rel_types;

	DTensor<mode, Dtype> node_trans, node_grad;
};

}

#endif