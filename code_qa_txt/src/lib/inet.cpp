#include "inet.h"
#include "knowledge_base.h"
#include "dataset.h"
#include "util/graph_struct.h"

#include <algorithm>
#include "global.h"

INet::INet()
{    
    inputs.clear();
    entity_bow.Reshape({kb.node_dict.size(), word_dict.size()});
    
    int nnz = 0;
    for (auto& p : kb.node_dict)
        nnz += p.second->word_idx_list.size();
    entity_bow.ResizeSp(nnz, kb.node_dict.size() + 1);

    nnz = 0;
    for (size_t i = 0; i < kb.node_list.size(); ++i)
    {
        auto* node = kb.node_list[i];
        entity_bow.data->row_ptr[i] = nnz;
        for (size_t j = 0; j < node->word_idx_list.size(); ++j)
        {
            entity_bow.data->col_idx[nnz] = node->word_idx_list[j];
            entity_bow.data->val[nnz] = 1.0;
            nnz++;
        }
    }
    entity_bow.data->row_ptr[kb.node_list.size()] = nnz;
    assert(nnz == entity_bow.data->nnz);

    m_entity_bow.CopyFrom(entity_bow);
    inputs["entity_bow"] = &m_entity_bow;    
}

void INet::BuildBatchGraph(std::vector< Sample* >& mini_batch, Phase phase)
{
    /*
    q_bow_input.Reshape({mini_batch.size(), side_word_dict.size()});    
    q_entity_input.Reshape({mini_batch.size(), kb.node_dict.size()});
    ans_output.Reshape({mini_batch.size(), kb.node_dict.size()});
    int nnz_q_bow = 0, nnz_q_entity = 0, nnz_ans = 0;
    for (auto* s : mini_batch)
    {
        nnz_q_bow += s->q_side_word_list.size();
        nnz_q_entity += s->q_entities.size();
        nnz_ans += s->answer_entities.size();
    }
    q_bow_input.ResizeSp(nnz_q_bow, mini_batch.size() + 1);
    q_entity_input.ResizeSp(nnz_q_entity, mini_batch.size() + 1);
    if (phase == Phase::TEST)
        ans_output.ResizeSp(nnz_ans, mini_batch.size() + 1);
    else
        ans_output.ResizeSp(mini_batch.size(), mini_batch.size() + 1);

    nnz_q_bow = 0; nnz_q_entity = 0; nnz_ans = 0;
    for (int i = 0; i < (int)mini_batch.size(); ++i)
    {
        auto* sample = mini_batch[i];
        q_bow_input.data->row_ptr[i] = nnz_q_bow;
        q_entity_input.data->row_ptr[i] = nnz_q_entity;
        ans_output.data->row_ptr[i] = nnz_ans;

        int base_idx = nnz_q_bow;
        for (auto e : sample->q_side_word_list)
        {
            q_bow_input.data->val[nnz_q_bow] = 1.0;
            q_bow_input.data->col_idx[nnz_q_bow] = e;
            nnz_q_bow += 1;
        }
        std::sort(q_bow_input.data->col_idx + base_idx, q_bow_input.data->col_idx + nnz_q_bow);

        base_idx = nnz_q_entity;
        for (auto e : sample->q_entities)
        {
            q_entity_input.data->val[nnz_q_entity] = 1.0;
            q_entity_input.data->col_idx[nnz_q_entity] = e->idx;
            nnz_q_entity += 1;
        }
        std::sort(q_entity_input.data->col_idx + base_idx, q_entity_input.data->col_idx + nnz_q_entity);

        if (phase == Phase::TEST)
        {
            base_idx = nnz_ans;
            for (auto e : sample->answer_entities)
            {
                ans_output.data->val[nnz_ans] = 1.0;
                ans_output.data->col_idx[nnz_ans] = e->idx;
                nnz_ans += 1; 
            }
            std::sort(ans_output.data->col_idx + base_idx, ans_output.data->col_idx + nnz_ans);
        } else {
            int e_idx = rand() % sample->answer_entities.size();
            assert(e_idx == 0);
            ans_output.data->val[nnz_ans] = 1.0;
            ans_output.data->col_idx[nnz_ans] = sample->answer_entities[e_idx]->idx;
            nnz_ans += 1;
        }
    }
    q_bow_input.data->row_ptr[mini_batch.size()] = nnz_q_bow;
    q_entity_input.data->row_ptr[mini_batch.size()] = nnz_q_entity;
    ans_output.data->row_ptr[mini_batch.size()] = nnz_ans;
    assert(nnz_q_bow == q_bow_input.data->nnz);
    assert(nnz_q_entity == q_entity_input.data->nnz);
    assert(nnz_ans == ans_output.data->nnz);

    m_q_bow_input.CopyFrom(q_bow_input);
    m_q_entity_input.CopyFrom(q_entity_input);
    m_ans_output.CopyFrom(ans_output); */
}
