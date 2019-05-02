#ifndef KNOWLEDGE_BASE_H
#define KNOWLEDGE_BASE_H

#include <map>
#include <string>
#include <vector> 
#include <set>

class Node
{
public:    

    Node(std::string _name, int _idx);

    void AddNeighbor(int rel_type, Node* y, int kb_idx, bool isReverse);

    std::string name;
    int idx;
    std::vector< int > word_idx_list;
    std::vector< std::pair< int, Node* > > adj_list;
    std::vector< std::pair< int, bool > > triplet_info;
};

struct Sample;
class SubGraph
{
public:
    SubGraph(Sample* _sample);

    std::vector< Node* > nodes;
    std::vector< std::tuple< int, int, int > > edges;
    Sample* sample;

protected:
    void AddNode(Node* node);
    void AddEdge(Node* src, int r_type, Node* dst);
    std::map<std::string, int> node_map;
    std::set< std::tuple<int, int, int> > edge_set;
    void BFS(Node* root);
};

class KnowledgeBase
{
public:

        KnowledgeBase();

        void ParseKnowledgeFile();

        void ParseEntityInAnswers(const char* suffix);

        Node* GetOrAddNode(std::string name);
        
        Node* GetNodeOrDie(std::string name);

        int NodeIdx(std::string name);

        std::map<std::string, Node*> node_dict;
        std::vector<Node*> node_list;
        int n_knowledges;
};

#endif