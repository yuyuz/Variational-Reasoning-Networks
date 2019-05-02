#include "node_select.h"
#include "global.h"

namespace gnn
{

NodeSelect::NodeSelect(std::string _name) 
		: Factor(_name, PropErr::N)
{

}

void NodeSelect::SetupNodes(const int len, const int* idxes, std::vector<Node*>& dst)
{
    dst.resize(len);
    for (int i = 0; i < len; ++i)
        dst[i] = kb.node_list[idxes[i]];
}

void NodeSelect::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						    std::vector< std::shared_ptr<Variable> >& outputs,
                            Phase phase)
{
	ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

    auto& node_info = *(dynamic_cast< VectorVar<Node*>* >(outputs[0].get())->vec);
    
	MAT_MODE_SWITCH(operands[0]->GetMode(), matMode, {        
        auto& indexes = dynamic_cast<DTensorVar<matMode, int>*>(operands[0].get())->value;        
        DTensor<CPU, int> t_idxes;

        int* ptr = indexes.data->ptr;
        if (indexes.GetMatMode() == MatMode::gpu)
        {
            t_idxes.CopyFrom(indexes);
            ptr = t_idxes.data->ptr;
        }        
        SetupNodes(indexes.shape.Count(), ptr, node_info);
	});
}

}