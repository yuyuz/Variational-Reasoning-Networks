#ifndef NET_LATENT_Y_H
#define NET_LATENT_Y_H

#include "inet.h"
#include "tensor/tensor_all.h"
#include "bfs_path_embed.h"
#include "graph_inner_product.h"
#include "node_select.h"
#include "var_sample.h"

#include <map>
#include <random>

using namespace gnn;

class NetLatentY : public INet
{
public:
    NetLatentY();
    virtual void BuildNet() override;
    virtual void BuildBatchGraph(std::vector< Sample* >& mini_batch, Phase phase) override;

    std::shared_ptr< DTensorVar<mode, Dtype> > GetCritic(std::shared_ptr< SpTensorVar<mode, Dtype> > q_bow);

    std::shared_ptr< DTensorVar<mode, Dtype> > GetMatchScores(std::shared_ptr< DTensorVar<mode, Dtype> >& q_embed, 
                                                std::shared_ptr<SampleVar>& samples, 
                                                std::shared_ptr< DTensorVar<mode, Dtype> >& rel_embed, 
                                                std::shared_ptr< DTensorVar<mode, Dtype> >& w_recur,
                                                std::shared_ptr< VectorVar<Node*> >& start_nodes);

    std::vector<Node*> answer_dst_nodes;
};

#endif