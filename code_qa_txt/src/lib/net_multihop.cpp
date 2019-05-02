#include "net_multihop.h"
#include "knowledge_base.h"
#include "dataset.h"
#include "var_sample.h"
#include "util/graph_struct.h"
#include "nn/nn_all.h"
#include <algorithm>
#include <set>
#include "global.h"

using namespace gnn;

NetMultiHop::NetMultiHop() : INet()
{
    inputs["q_bow"] = &m_q_bow_input;
    inputs["ans_output"] = &m_ans_output;    
    inputs["a_dst_nodes"] = &answer_dst_nodes;
}

void NetMultiHop::BuildBatchGraph(std::vector< Sample* >& mini_batch, Phase phase)
{    
    q_bow_input.Reshape({mini_batch.size(), word_dict.size()});
    y_idxes.Reshape({mini_batch.size(), (size_t)1});
    ans_output.Reshape({mini_batch.size(), kb.node_dict.size()});
    int nnz_q_bow = 0, nnz_ans = 0;
    for (auto* s : mini_batch)
    {
        nnz_q_bow += s->q_word_list.size();  
        nnz_ans += s->answer_entities.size();
    }
    q_bow_input.ResizeSp(nnz_q_bow, mini_batch.size() + 1);
    ans_output.ResizeSp(nnz_ans, mini_batch.size() + 1);
    
    nnz_q_bow = 0; nnz_ans = 0;
    answer_dst_nodes.clear();
    for (int i = 0; i < (int)mini_batch.size(); ++i)
    {
        auto* sample = mini_batch[i];

        assert(sample->q_entities.size());
        y_idxes.data->ptr[i] = sample->q_entities[0]->idx;
        int idx = rand() % sample->answer_entities.size();
        answer_dst_nodes.push_back(sample->answer_entities[idx]);

        q_bow_input.data->row_ptr[i] = nnz_q_bow;
        ans_output.data->row_ptr[i] = nnz_ans;

        int base_idx = nnz_q_bow;
        for (auto e : sample->q_word_list)
        {
            q_bow_input.data->val[nnz_q_bow] = 1.0;
            q_bow_input.data->col_idx[nnz_q_bow] = e;
            nnz_q_bow += 1;
        }
        std::sort(q_bow_input.data->col_idx + base_idx, q_bow_input.data->col_idx + nnz_q_bow);

        base_idx = nnz_ans;
        for (auto e : sample->answer_entities)
        {
            ans_output.data->val[nnz_ans] = 1.0;
            ans_output.data->col_idx[nnz_ans] = e->idx;
            nnz_ans += 1; 
        }
        std::sort(ans_output.data->col_idx + base_idx, ans_output.data->col_idx + nnz_ans);
    }
    q_bow_input.data->row_ptr[mini_batch.size()] = nnz_q_bow;
    ans_output.data->row_ptr[mini_batch.size()] = nnz_ans;
    assert(nnz_q_bow == q_bow_input.data->nnz);
    assert(nnz_ans == ans_output.data->nnz);

    m_q_bow_input.CopyFrom(q_bow_input);
    m_ans_output.CopyFrom(ans_output);
    inputs["sample_var"] = &mini_batch;
    inputs["y_idxes"] = &y_idxes;
}

std::shared_ptr< DTensorVar<mode, Dtype> > NetMultiHop::GetMatchScores(std::shared_ptr< DTensorVar<mode, Dtype> >& q_embed, 
                                                        std::shared_ptr<SampleVar>& samples, 
                                                        std::shared_ptr< DTensorVar<mode, Dtype> >& rel_embed, 
                                                        std::shared_ptr< DTensorVar<mode, Dtype> >& w_recur,
                                                        std::shared_ptr< VectorVar<Node*> >& start_nodes)
{
    std::vector< std::shared_ptr< Variable > > args = { samples, rel_embed, w_recur, start_nodes};
    auto tp = af< BfsPathEmbed<mode, Dtype> >(fg, args);
    //std::get<0>(tp) = af< ReLU >(fg, {std::get<0>(tp)});
    int h = 1;
    while (h < cfg::nhop_subg)
    {
        h++;    
        args = {samples, rel_embed, w_recur, std::get<0>(tp), std::get<1>(tp), std::get<2>(tp)};
        tp = af< BfsPathEmbed<mode, Dtype> >(fg, args);
        //std::get<0>(tp) = af< ReLU >(fg, {std::get<0>(tp)});
    }
    args = {q_embed, std::get<0>(tp), std::get<1>(tp), std::get<2>(tp)};
    auto match_scores = af< GraphInnerProduct<mode, Dtype> >(fg, args, kb.node_dict.size());
    return match_scores;
}

std::shared_ptr< DTensorVar<mode, Dtype> > NetMultiHop::GetCritic(std::shared_ptr< SpTensorVar<mode, Dtype> > q_bow)                                                                 
{
    auto critic_w_embed = add_diff<DTensorVar>(model, "critic_w_embed", {word_dict.size(), (size_t)cfg::n_embed});
    auto w1 = add_diff<DTensorVar>(model, "critic_w1", {(size_t)(cfg::n_embed + 1), (size_t)cfg::n_hidden});
    auto w2 = add_diff<DTensorVar>(model, "critic_w2", {(size_t)(cfg::n_hidden + 1), (size_t)1});

    critic_w_embed->value.SetRandN(0, cfg::w_scale);
    w1->value.SetRandN(0, cfg::w_scale);
    w2->value.SetRandN(0, cfg::w_scale);
    fg.AddParam(critic_w_embed);
    fg.AddParam(w1);
    fg.AddParam(w2);

    auto q_embed = af< MatMul >(fg, {q_bow, critic_w_embed});
    q_embed = af< ReLU >(fg, {q_embed});

    auto h1 = af< FullyConnected >(fg, {q_embed, w1});
    h1 = af< ReLU >(fg, {h1});

    auto h2 = af< FullyConnected >(fg, {h1, w2});
    h2 = af< ReLU >(fg, {h2});
    return h2;
}

void NetMultiHop::BuildNet()
{
    // inputs
    auto q_bow = add_const< SpTensorVar<mode, Dtype> >(fg, "q_bow", true);
    auto entity_bow = add_const< SpTensorVar<mode, Dtype> >(fg, "entity_bow", true);
    auto ans_output = add_const< SpTensorVar<mode, Dtype> >(fg, "ans_output", true);
    auto samples = add_const< SampleVar >(fg, "sample_var", true);    
    auto a_dst_nodes = add_const< VectorVar<Node*> >(fg, "a_dst_nodes", true);
    auto true_y_idx = add_const< DTensorVar<CPU, int> >(fg, "y_idxes", true);

    // parameters
    auto w_entity_match = add_diff<DTensorVar>(model, "w_entity_match", {word_dict.size(), (size_t)cfg::n_embed});
    auto w_path_query = add_diff<DTensorVar>(model, "w_path_query", {word_dict.size(), (size_t)cfg::n_embed});

    auto rel_embed = add_diff<DTensorVar>(model, "rel_embedding", {relation_dict.size() * (size_t)2, (size_t)cfg::n_embed});
    auto w_a2y_recur = add_diff<DTensorVar>(model, "w_a2y_recur", {cfg::n_embed, cfg::n_embed});
    auto w_y2a_recur = add_diff<DTensorVar>(model, "w_y2a_recur", {cfg::n_embed, cfg::n_embed});

    auto moving_mean = add_nondiff<DTensorVar>(model, "moving_mean", {(size_t)1, (size_t)1});
    auto moving_inv_std = add_nondiff<DTensorVar>(model, "moving_inv_std", {(size_t)1, (size_t)1});

    moving_mean->value.Fill(0.0);
    moving_inv_std->value.Fill(1.0);
    fg.AddParam(moving_mean);
    fg.AddParam(moving_inv_std);

    w_entity_match->value.SetRandN(0, cfg::w_scale);
    w_path_query->value.SetRandN(0, cfg::w_scale);
    rel_embed->value.SetRandN(0, cfg::w_scale);
    w_a2y_recur->value.SetRandN(0, cfg::w_scale);
    w_y2a_recur->value.SetRandN(0, cfg::w_scale);    
    fg.AddParam(w_entity_match);
    fg.AddParam(w_path_query);
    fg.AddParam(rel_embed);
    fg.AddParam(w_a2y_recur);
    fg.AddParam(w_y2a_recur);

    auto q_embed_entity = af< MatMul >(fg, {q_bow, w_entity_match});
    auto q_embed_query = af< MatMul >(fg, {q_bow, w_path_query});
    auto entity_bow_embed = af< MatMul >(fg, {entity_bow, w_entity_match}); 
    auto q_y_bow_match = af< MatMul >(fg, {q_embed_entity, entity_bow_embed}, Trans::N, Trans::T);


    //================ q_y_given_qa ===========================
    auto y_q_path_match = GetMatchScores(q_embed_entity, samples, rel_embed, w_a2y_recur, a_dst_nodes);
    auto q_y_given_qa_scores = af< ElewiseAdd >(fg, {q_y_bow_match, y_q_path_match});

    auto sampled_y_nodes = af< NodeSelect >(fg, {true_y_idx});


    //================ p_a_given_qy ===========================
    auto a_q_path_math = GetMatchScores(q_embed_query, samples, rel_embed, w_y2a_recur, sampled_y_nodes);

    //================ -log p(y | q) - log p(a | y, q) ===========================
    auto ce_ans = af< CrossEntropy >(fg, {a_q_path_math, ans_output}, true); 
    auto one_hot_sampled_y = af< OneHot<mode, Dtype> >(fg, {true_y_idx}, kb.node_dict.size());
    auto ce_y = af< CrossEntropy >(fg, {q_y_bow_match, one_hot_sampled_y}, true);

    //================ \abla log Q(y | q, a) * score ===========================
    auto ce_joint = af< ElewiseAdd >(fg, {ce_ans, ce_y});

    auto truey_onehot = af< OneHot<mode, Dtype> >(fg, {true_y_idx}, kb.node_dict.size());
    auto pos_neg_loss = af< CrossEntropy >(fg, {q_y_given_qa_scores, truey_onehot}, true);
    auto baseline = GetCritic(q_bow);
    auto normed_signal = af< MovingNorm >(fg,{ce_joint, moving_mean, moving_inv_std}, 0.1, PropErr::N);
    auto learning_signal = af< ElewiseMinus >(fg, {normed_signal, baseline}, PropErr::N);

    //=========== baseline mse ================== 
//    auto square_error = af< SquareError > (fg, {baseline, normed_signal});
//    std::vector<Dtype> coeff = {1.0, 1.0, 1.0};
//    loss = af< ElewiseAdd >(fg, {ce_joint, pos_neg_loss, square_error}, coeff);
    loss = af< ElewiseAdd >(fg, {ce_joint, pos_neg_loss});
    loss = af< ReduceMean >(fg, {loss});

    //================ inference ===========================
    auto argmax_y = true_y_idx;
    auto infer_y_nodes = af< NodeSelect >(fg, {argmax_y});
    auto pred = GetMatchScores(q_embed_query, samples, rel_embed, w_y2a_recur, infer_y_nodes);

    auto hitk = af< HitAtK >(fg, {pred, ans_output});
    auto real_hitk = af< TypeCast<mode, Dtype> >(fg, {hitk});
    hit_rate = af< ReduceMean >(fg, {real_hitk});
}
