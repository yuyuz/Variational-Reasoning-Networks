#ifndef DATASET_H
#define DATASET_H

#include "config.h"
#include "knowledge_base.h"

struct Sample
{
    int s_idx;
    std::vector< int > q_word_list, q_side_word_list;
    std::vector< Node* > q_entities, answer_entities;
    Sample(); 
};

class Dataset
{
public:

    Dataset();

    void Load(const char* suffix);

    void SetupStream(bool randomized = false);

    bool GetMiniBatch(int batch_size, std::vector< Sample* >& mini_batch);
    bool GetSplitMiniBatch(int batch_size, std::vector< Sample* >& mini_batch);

    std::vector< Sample* > orig_samples;
    std::vector< Sample* > split_samples;

private:

    bool GetData(int batch_size, std::vector< Sample* >& mini_batch, std::vector< Sample* >& samples);
    bool randomized;
    int cur_pos;
    std::vector<int> idxes;
};

#endif