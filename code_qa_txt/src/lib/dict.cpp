#include "dict.h"
#include "util.h"
#include <string>

std::map<std::string, int> GetRelations()
{
    auto relations = {"directed_by", 
                      "has_genre",
                      "has_imdb_rating",
                      "has_imdb_votes",
                      "has_plot",
                      "has_tags",
                      "in_language",
                      "release_year",
                      "starred_actors",
                      "written_by"};
    std::map<std::string, int> result;
    result.clear();
    int t = 0;
    for (auto& st : relations)
    {
        result[st] = t;
        t++;
    }
        
    std::cerr << "num relations: " << result.size() << std::endl;
    std::cerr << "[";
    for (auto i : relations)
        std::cerr << " " << i;
    std::cerr << " ]" << std::endl;

    return result;
}

std::map<std::string, int> GetVocab()
{
    std::map<std::string, int> word_dict;
    word_dict.clear();

    for (auto suffix : {"train", "test", "dev"})
    {
        auto file = fmt::format("{0}/{1}-hop/{2}/qa_{3}.txt", 
                            cfg::data_root, cfg::nhop_subg, cfg::dataset, suffix);

        std::ifstream fin(file);

        std::string st;
        std::vector<std::string> buf;

        while (std::getline(fin, st))
        {        
            str_split(st, '\t', buf);
            assert(buf.size() == 2);
            auto q = buf[0];
            str_replace(q, "[", "");
            str_replace(q, "]", "");

            str_split(q, ' ', buf);
            for (auto w : buf)
            {
                if (word_dict.count(w) == 0)
                {
                    int t = word_dict.size();
                    word_dict[w] = t;
                }
            }
        }
    }
    // words from kb
    auto kb_file = fmt::format("{0}/kb.txt", cfg::data_root);
    std::ifstream fin(kb_file);

    std::string st;
    std::vector<std::string> buf, word_buf;
    while (std::getline(fin, st))
    {
        str_split(st, '|', buf);
        assert(buf.size() == 3);

        str_split(buf[0], ' ', word_buf);
            for (auto& w : word_buf)
            {
                if (word_dict.count(w) == 0)
                {
                    int t = word_dict.size();
                    word_dict[w] = t;
                }
            }

        str_split(buf[2], ' ', word_buf);
            for (auto& w : word_buf)
            {
                if (word_dict.count(w) == 0)
                {
                    int t = word_dict.size();
                    word_dict[w] = t;
                }
            }
    }

    std::cerr << "size of vocab: " << word_dict.size() << std::endl;
    return word_dict;    
}

std::map<std::string, int> GetSideWordDict()
{
    auto file = fmt::format("{0}/{1}-hop/{2}/qa_train.txt", 
                            cfg::data_root, cfg::nhop_subg, cfg::dataset);
    std::ifstream fin(file);

    std::string st;
    std::vector<std::string> buf;

    std::map<std::string, int> word_dict;
    word_dict.clear();

    while (std::getline(fin, st))
    {        
        str_split(st, '\t', buf);
        assert(buf.size() == 2);
        auto q = buf[0];
        str_split(q, ' ', buf);
        bool in_entity = false;
        for (auto w : buf)
        {
            if (w == "1")
                continue;
            if (w[0] == '[')
                in_entity = true;
            if (in_entity)
            {
                if (w[w.size() - 1] == ']')
                    in_entity = false;
            } else {
                if (word_dict.count(w) == 0)
                {
                    int t = word_dict.size();
                    word_dict[w] = t;
                }
            }
        }
    }

    std::cerr << "size of side_vocab: " << word_dict.size() << std::endl;
    return word_dict;    
}