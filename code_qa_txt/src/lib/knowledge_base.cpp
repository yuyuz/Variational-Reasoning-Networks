#include "knowledge_base.h"
#include "config.h"
#include "util.h"
#include "dataset.h"
#include <string>
#include <vector>
#include <queue>
#include "global.h"

Node::Node(std::string _name, int _idx) : name(_name), idx(_idx)
{
    adj_list.clear();
    triplet_info.clear();
    word_idx_list.clear();

    std::vector<std::string> buf;
    str_split(name, ' ', buf);

    for (auto& w : buf)
    {
        if (word_dict.count(w) == 0)
            std::cerr << w << std::endl;
        assert(word_dict.count(w));
        word_idx_list.push_back(word_dict[w]);
    }
    std::sort(word_idx_list.begin(), word_idx_list.end());    
}

void Node::AddNeighbor(int rel_type, Node* y, int kb_idx, bool isReverse)
{
    adj_list.push_back(std::make_pair(rel_type, y));
    triplet_info.push_back(std::make_pair(kb_idx, isReverse));
}

SubGraph::SubGraph(Sample* _sample) : sample(_sample)
{
    nodes.clear();
    edges.clear();

    assert(sample->q_entities.size());

    node_map.clear();
    edge_set.clear();
    for (auto* e : sample->q_entities)
    {
        BFS(e);      
    }
}

void SubGraph::AddNode(Node* node)
{
    if (node_map.count(node->name))
        return;
    node_map[node->name] = nodes.size();
    nodes.push_back(node);
}

void SubGraph::AddEdge(Node* src, int r_type, Node* dst)
{
    assert(node_map.count(src->name));
    assert(node_map.count(dst->name));

    int x = node_map[src->name], y = node_map[dst->name];
    assert(x < (int)nodes.size());
    assert(y < (int)nodes.size());
    if (x > y){
        int t = x; x = y; y = t;
    }
    auto p = std::make_tuple(x, r_type, y);
    if (!edge_set.count(p))
    {
        edge_set.insert(p);
        edges.push_back(p);
    }
}

void SubGraph::BFS(Node* root)
{
    AddNode(root);
    std::queue< std::pair<Node*, int> > q_node;
    while (!q_node.empty())
        q_node.pop();
    q_node.push( std::make_pair(root, 0) );
    while (!q_node.empty())
    {
        auto tt = q_node.front();
        if (tt.second >= cfg::nhop_subg)
            break;
        q_node.pop();
        for (auto p : tt.first->adj_list)
        {
            if (!node_map.count(p.second->name))
            {
                AddNode(p.second);
                q_node.push(std::make_pair(p.second, tt.second + 1));
            }
            AddEdge(tt.first, p.first, p.second);
        }    
    }
}

KnowledgeBase::KnowledgeBase()
{
    node_dict.clear();
    node_list.clear();
}

Node* KnowledgeBase::GetOrAddNode(std::string name)
{
    if (node_dict.count(name) == 0)
    {
        assert(node_dict.size() == node_list.size());
        int t = node_dict.size();
        auto* node = new Node(name, t);
        node_dict[name] = node;
        node_list.push_back(node);
    }
    return node_dict[name];
}

Node* KnowledgeBase::GetNodeOrDie(std::string name)
{
    assert(node_dict.count(name));
    return node_dict[name];   
}

int KnowledgeBase::NodeIdx(std::string name)
{
    assert(node_dict.count(name));
    return node_dict[name]->idx;
}

void KnowledgeBase::ParseKnowledgeFile()
{
    auto kb_file = fmt::format("{0}/kb.txt", cfg::data_root);
    std::ifstream fin(kb_file);

    std::string st;
    std::vector<std::string> buf;
    int n_lines = 0;
    while (std::getline(fin, st))
    {
        str_split(st, '|', buf);
        assert(buf.size() == 3);

        auto* src = GetOrAddNode(buf[0]);
        assert(relation_dict.count(buf[1]));
        auto* dst = GetOrAddNode(buf[2]);

        src->AddNeighbor(relation_dict[buf[1]], dst, n_lines, false);
        dst->AddNeighbor(relation_dict.size() + relation_dict[buf[1]], src, n_lines, true);
        n_lines += 1;
    }
    std::cerr << n_lines << " knowledge triples loaded" << std::endl;
    n_knowledges = n_lines;
    std::cerr << "#entities in kb: " << node_dict.size() << std::endl;

    for (auto& p : node_dict)
    {
        auto* node = p.second;
        std::sort(node->adj_list.begin(), node->adj_list.end(), 
            [](const std::pair< int, Node* >& x, const std::pair< int, Node* >& y){
                return x.second->idx < y.second->idx;
        });
    }
}

void KnowledgeBase::ParseEntityInAnswers(const char* suffix)
{
    auto file = fmt::sprintf("{0}/{1}-hop/{2}/qa_{3}.txt", 
                            cfg::data_root, cfg::nhop_subg, cfg::dataset, suffix);

    std::ifstream fin(file);

    std::string st;
    std::vector<std::string> buf;

    while (std::getline(fin, st))
    {        
        str_split(st, '\t', buf);
        assert(buf.size() == 2);
        st = buf[1];
        str_split(st, '|', buf);

        for (auto e : buf)
        {
            GetOrAddNode(e);
        }
    }
}
