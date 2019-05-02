#include "config.h"
#include "nn/nn_all.h"
#include "dataset.h"
#include <algorithm>
#include "dict.h"
//#include "graph_struct.h"
#include "knowledge_base.h"
#include "global.h"
#include "net_multihop.h"
#include "net_latent_y.h"

Dataset train_set, val_set, test_set;
std::vector< Sample* > mini_batch;

void EvalSet(std::string prefix, Dataset& dset, INet* net)
{
	dset.SetupStream(false);
	Dtype loss_total = 0.0;
	while (dset.GetMiniBatch(cfg::batch_size, mini_batch))
	{		
		net->BuildBatchGraph(mini_batch, Phase::TEST);
		fg.FeedForward({net->hit_rate}, net->inputs, Phase::TEST);

		loss_total += mini_batch.size() * net->hit_rate->value.AsScalar();
	}
	std::cerr << prefix << "@iter: " << cfg::iter;
	std::cerr << "\thit_rate@1: " << loss_total / dset.orig_samples.size();
	std::cerr << std::endl;
}

std::vector<int> idx_buf; 
void GetTopK(DTensor<mode, Dtype>& prob, std::vector< std::vector< std::pair<int, Dtype> > >& idx_list)
{
	idx_list.resize(prob.rows());
	
	for (size_t i = 0; i < prob.rows(); ++i)
	{
		idx_list[i].clear();
		Dtype* ptr = prob.data->ptr + i * prob.cols();

		std::sort(idx_buf.begin(), idx_buf.end(), [&](const int& i, const int& j) {
			return ptr[i] > ptr[j];
		});
		for (int j = 0; j < cfg::test_tpok; ++j)
			idx_list[i].push_back(std::make_pair(idx_buf[j], ptr[idx_buf[j]]));
	}
}

void Print2File(FILE* fid, DTensor<mode, Dtype>& prob, std::vector< std::vector< std::pair<int, Dtype> > >& idx_list)
{
	for (size_t i = 0; i < prob.rows(); ++i)
	{
		for (int j = 0; j < cfg::test_tpok; ++j)
		{
			auto& p = idx_list[i][j];

			auto* node = kb.node_list[p.first];
			if (j)
				fprintf(fid, "|");
			fprintf(fid, "(");
			for (size_t j = 0; j < node->name.size(); ++j)
				fprintf(fid, "%c", node->name[j]);
			fprintf(fid, ",%.6f)", p.second);
		}
		fprintf(fid, "\n");	
	}
}

void SavePred(std::string prefix, Dataset& dset, INet* net)
{
	dset.SetupStream(false);
	idx_buf.resize(kb.node_dict.size());
	for (size_t i = 0; i < idx_buf.size(); ++i)
		idx_buf[i] = i;
	
	FILE* fy = fopen(fmt::sprintf("%s/%s_ypred.txt", cfg::save_dir, prefix).c_str(), "w");
//	FILE* fa = fopen(fmt::sprintf("%s/%s_apred.txt", cfg::save_dir, prefix).c_str(), "w");

	while (dset.GetMiniBatch(cfg::batch_size, mini_batch))
	{		
		net->BuildBatchGraph(mini_batch, Phase::TEST);
//		fg.FeedForward({net->pos_probs, net->pred}, net->inputs, Phase::TEST);
		fg.FeedForward({net->q_y_bow_match}, net->inputs, Phase::TEST);

		auto& y_prob = net->q_y_bow_match->value;
		y_prob.Softmax();
		std::vector< std::vector< std::pair<int, Dtype> > > idx_list;
		GetTopK(y_prob, idx_list);
		Print2File(fy, y_prob, idx_list);

//		auto& pred = net->pred->value;
//		pred.Softmax();
//		GetTopK(pred, idx_list);
//		Print2File(fa, pred, idx_list);
	}
	fclose(fy);
//	fclose(fa);
}

void MainLoop(INet* inet)
{
	//MomentumSGDOptimizer<mode, Dtype> learner(&model, cfg::lr, cfg::momentum, cfg::l2_penalty);
	AdamOptimizer<mode, Dtype> learner(&model, cfg::lr, cfg::l2_penalty);

	int max_iter = (long long)cfg::max_iter;
	int init_iter = cfg::iter;
	if (init_iter > 0)
	{
		std::cerr << fmt::sprintf("loading model for iter=%d", init_iter) << std::endl;
		model.Load(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, init_iter));
	}

	train_set.SetupStream(true);		
	
	for (; cfg::iter <= max_iter; ++cfg::iter)
	{
		if (cfg::iter != init_iter && cfg::iter % cfg::test_interval == 0)
		{	
			//EvalSet("train", train_set, inet);
			EvalSet("dev", val_set, inet);			
			EvalSet("test", test_set, inet);
		}
		if (cfg::iter % cfg::save_interval == 0 && cfg::iter != init_iter)
		{			
			printf("saving model for iter=%d\n", cfg::iter);
			model.Save(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, cfg::iter));
		}

		assert(train_set.GetSplitMiniBatch(cfg::batch_size, mini_batch));
		inet->BuildBatchGraph(mini_batch, Phase::TRAIN);
		
		fg.FeedForward({inet->loss}, inet->inputs, Phase::TRAIN);

    	if (cfg::iter % cfg::report_interval == 0)
		{	
			std::cerr << "iter: " << cfg::iter;	
			std::cerr << "\tloss: " << inet->loss->value.AsScalar();
			// for (auto t : inet->targets)
			// 	std::cerr << "\t" << t->name << ": " << dynamic_cast<TensorVar<mode, Dtype>*>(t.get())->AsScalar();
			std::cerr << std::endl;
		}
		fg.BackPropagate({inet->loss});
		learner.Update();
	}
}

std::vector< std::string> q_types;
size_t total_ntypes;

void LoadTestTypes()
{
    auto file = fmt::format("{0}/{1}-hop/qa_test_qtype.txt", 
                            cfg::data_root, cfg::nhop_subg);
    std::ifstream fin(file);
	std::string st;
	q_types.clear();
	total_ntypes = 0;
	std::set<std::string> ss;
	while (fin >> st)
	{
		q_types.push_back(st);
		ss.insert(st);
	}
	total_ntypes = ss.size();
	assert(q_types.size() == test_set.orig_samples.size());
        std::cerr << "ntypes: " << total_ntypes << std::endl;
}

void OutputScores(INet* net)
{
	FILE* fout = fopen(fmt::sprintf("%s/test_vis.txt", cfg::save_dir).c_str(), "w");
	for (size_t i = 0; i < kb.node_list.size(); ++i)
		fprintf(fout, "%s\n", kb.node_list[i]->name.c_str());
	std::set< std::string > selected;

	auto& rel_embed = model.params["rel_embedding"]->value;

	DTensor<CPU, Dtype> mat; 
	size_t idx = 0;

	std::vector< std::string > rel_list(relation_dict.size());
	for (auto& p : relation_dict)
	{
		rel_list[p.second] = p.first;
	}

	while (test_set.GetMiniBatch(cfg::batch_size, mini_batch))
	{		
		net->BuildBatchGraph(mini_batch, Phase::TEST);
		fg.FeedForward({net->q_embed_query, net->pred, net->hitk}, net->inputs, Phase::TEST);

		auto& q_embed = net->q_embed_query->value;
		mat.MM(q_embed, rel_embed, Trans::N, Trans::T, 1.0, 0.0);

		for (size_t j = 0; j < q_embed.rows(); ++j)
		{
			auto& tt = q_types[idx + j];
			if (selected.count(tt))
				continue;
			if (net->hitk->value.data->ptr[j] == 0)
				continue;
			selected.insert(tt);
                        std::cerr << tt << std::endl;

			fprintf(fout, "%d\n", (int)idx + (int)j);
			
			auto* cur_pred = net->pred->value.data->ptr + j * net->pred->value.cols();
			for (size_t k = 0; k < net->pred->value.cols(); ++k)
			{
				if (k)
					fprintf(fout, " ");
				fprintf(fout, "%.6f", cur_pred[k]);
			}
			fprintf(fout, "\n");

			cur_pred = mat.data->ptr + j * mat.cols();
			for (size_t k = 0; k < mat.cols(); ++k)
			{
				std::string tttt;
				if (k < rel_list.size())
					tttt = rel_list[k];
				else
					tttt = rel_list[k - rel_list.size()] + "-inv";
				fprintf(fout, "%s %.6f\n", tttt.c_str(), cur_pred[k]);
			}
		}
                std::cerr << selected.size() << " " << total_ntypes << std::endl;
		if (selected.size() == total_ntypes)
                {
                        std::cerr << "job done" << std::endl;
			break;
                }
		idx += q_embed.rows();
	}
        std::cerr << idx << std::endl;
	fclose(fout);
}

int main(const int argc, const char** argv)
{
	srand(time(NULL));

	cfg::LoadParams(argc, argv);
	GpuHandle::Init(cfg::dev_id, 1);

	relation_dict = GetRelations();
	word_dict = GetVocab();
	side_word_dict = GetSideWordDict();
	kb.ParseKnowledgeFile();
	for (auto suffix : {"train", "test", "dev"})
	{
		kb.ParseEntityInAnswers(suffix);
	}
	std::cerr << "# entites in total: " << kb.node_dict.size() << std::endl;

	train_set.Load("train");
	val_set.Load("dev");
	test_set.Load("test");

	std::cerr << "building net..." << std::endl;
	INet* net = nullptr;
	if (!strcmp(cfg::net_type, "NetMultiHop"))
	{
		net = new NetMultiHop();
	}
	else if (!strcmp(cfg::net_type, "NetLatentY"))
	{
		assert(cfg::init_idx_file == nullptr);
		net = new NetLatentY();
	}
	else {
		std::cerr << "unknown net type: " << cfg::net_type << std::endl;
		return 0;
	}
	
	net->BuildNet();
	std::cerr << "done" << std::endl;
	if (cfg::test_only)
	{
		model.Load(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, cfg::iter));
		SavePred("dev", val_set, net);			
		SavePred("test", test_set, net);
	} else if (cfg::vis_score)
	{
		model.Load(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, cfg::iter));
		LoadTestTypes();
		OutputScores(net);
	} else
		MainLoop(net);

        //GpuHandle::Destroy();
	return 0;	
}
