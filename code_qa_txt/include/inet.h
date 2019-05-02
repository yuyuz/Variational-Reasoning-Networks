#ifndef INET_H
#define INET_H

#include <map>
#include <string>
#include <vector>
#include "config.h"
#include "tensor/tensor.h"
#include "nn/variable.h"

using namespace gnn;

struct Sample;
class INet
{
public:
    INet();

    virtual void BuildNet() = 0;
    virtual void BuildBatchGraph(std::vector< Sample* >& mini_batch, Phase phase);

    SpTensor<CPU, Dtype> q_bow_input, ans_output;
    SpTensor<mode, Dtype> m_q_bow_input, m_ans_output;

    DTensor<CPU, int> y_idxes;

    std::map< std::string, void* > inputs;
    SpTensor<CPU, Dtype> entity_bow;
    SpTensor<mode, Dtype> m_entity_bow;
    std::shared_ptr< DTensorVar<mode, Dtype> > loss, hit_rate, pos_probs, pred, q_embed_query, q_y_bow_match;
    std::shared_ptr< DTensorVar<CPU, int> > sampled_y_idx, hitk;
};

#endif