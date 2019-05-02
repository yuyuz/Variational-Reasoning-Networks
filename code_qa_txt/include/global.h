#ifndef GLOBAL_H
#define GLOBAL_H

#include "config.h"
#include "nn/factor_graph.h"
#include "util/graph_struct.h"
#include "nn/param_set.h"
#include "knowledge_base.h"
#include <map>

using namespace gnn;

extern KnowledgeBase kb;
extern std::map<std::string, int> side_word_dict;
extern ParamSet<mode, Dtype> model;
extern FactorGraph fg;
extern std::map<std::string, int> word_dict;
extern std::map<std::string, int> relation_dict;

#endif