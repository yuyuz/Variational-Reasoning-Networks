#include "dataset.h"
#include "util.h"
#include <fstream>
#include <map>
#include <string>
#include "global.h"

Sample::Sample()
{
    q_word_list.clear(); 
    q_entities.clear();
    answer_entities.clear(); 
    q_side_word_list.clear();
}

Dataset::Dataset()
{
    orig_samples.clear();
    split_samples.clear();
    idxes.clear();
}

void Dataset::Load(const char* suffix)
{
    auto filename = fmt::sprintf("%s/%d-hop/%s/qa_%s.txt", 
                            cfg::data_root, cfg::nhop_subg, cfg::dataset, suffix);
    std::set<int> train_idxes;
    if (!strcmp(suffix, "train") && cfg::init_idx_file)
    {
        std::ifstream fin(cfg::init_idx_file);
        int idx;
        while (fin >> idx)
        {
            train_idxes.insert(idx);
        }
    }
    std::ifstream fin(filename);

    std::string st;
    std::vector<std::string> buf;
    int s_idx = 0, line_num = 0;
    std::map<int, int> n_q;
    while (std::getline(fin, st))
    {        
        line_num++;
        if (train_idxes.size() && !train_idxes.count(line_num - 1))
            continue;
        str_split(st, '\t', buf);
        assert(buf.size() == 2);

        auto* cur_sample = new Sample();
        auto q = buf[0], a = buf[1];

        size_t pos = 0;
        while (pos < q.size())
        {
            if (q[pos] == '[')
            {
                auto ed = pos + 1;
                while (ed < q.size() && q[ed] != ']')
                    ed++;
                auto e_name = q.substr(pos + 1, ed - pos - 1);
                if (kb.node_dict.count(e_name))
                    cur_sample->q_entities.push_back(kb.node_dict[e_name]);
                pos = ed;
            }
            pos++;
        }
        if (cur_sample->q_entities.size() == 0)
            continue;
        if (!n_q.count(cur_sample->q_entities.size()))
            n_q[cur_sample->q_entities.size()] = 0;
        n_q[cur_sample->q_entities.size()]++;
        assert(cur_sample->q_entities.size() <= 1); // at most have one entity in query
        
        cur_sample->s_idx = s_idx;
        s_idx++;
        str_replace(q, "[", "");
        str_replace(q, "]", "");

        str_split(q, ' ', buf);
        for (auto w : buf)
        {
            if (word_dict.count(w))
                cur_sample->q_word_list.push_back(word_dict[w]);
            if (side_word_dict.count(w))
                cur_sample->q_side_word_list.push_back(side_word_dict[w]);            
        }

        str_split(a, '|', buf);
        for (auto e : buf)
        {
            cur_sample->answer_entities.push_back(kb.GetNodeOrDie(e));
        }

        orig_samples.push_back(cur_sample);

        for (auto e : cur_sample->answer_entities)
        {
            auto* s = new Sample();
            s->q_word_list = cur_sample->q_word_list;
            s->q_side_word_list = cur_sample->q_side_word_list;
            s->q_entities = cur_sample->q_entities;
            s->answer_entities.push_back(e);
            split_samples.push_back(s);
        }
    }
    std::cerr << suffix << " has " << orig_samples.size() << " samples" << " and split into " << split_samples.size() << std::endl;
    for (auto p : n_q)
        std::cerr << "n_q: " << p.first << " # samples: " << p.second << std::endl;
}

void Dataset::SetupStream(bool randomized)
{
    this->randomized = randomized;
    cur_pos = 0;

    if (randomized)
    {
        std::random_shuffle(orig_samples.begin(), orig_samples.end());
        std::random_shuffle(split_samples.begin(), split_samples.end());
    }        
}

bool Dataset::GetData(int batch_size, std::vector< Sample* >& mini_batch, std::vector< Sample* >& samples)
{
    if (cur_pos + batch_size > (int)samples.size() && randomized)
    {
        std::random_shuffle(samples.begin(), samples.end());
        cur_pos = 0;
    }
    if (cur_pos + batch_size > (int)samples.size())
        batch_size = samples.size() - cur_pos;
    if (batch_size <= 0)
        return false;

    mini_batch.resize(batch_size);
    for (int i = cur_pos; i < cur_pos + batch_size; ++i)
    {
        mini_batch[i - cur_pos] = samples[i];
    }

    cur_pos += batch_size;
    return true;
}

bool Dataset::GetMiniBatch(int batch_size, std::vector< Sample* >& mini_batch)
{
    return GetData(batch_size, mini_batch, orig_samples);
}

bool Dataset::GetSplitMiniBatch(int batch_size, std::vector< Sample* >& mini_batch)
{
    return GetData(batch_size, mini_batch, split_samples);
}
