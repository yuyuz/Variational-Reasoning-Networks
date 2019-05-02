#ifndef NODE_SELECT_H
#define NODE_SELECT_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"
#include "util/fmt.h"
#include "var_sample.h"

class Node;

namespace gnn
{

class NodeSelect : public Factor
{
public:
	static std::string StrType()
	{
		return "NodeSelect";
	}

	using OutType = std::shared_ptr< VectorVar<Node*> >;

	OutType CreateOutVar()
	{
        auto out_name = fmt::sprintf("%s:out_0", this->name);
		return std::make_shared< VectorVar<Node*> >(out_name);
	}

	NodeSelect(std::string _name);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

private:
    void SetupNodes(const int len, const int* idxes, std::vector<Node*>& dst);
};

}

#endif
