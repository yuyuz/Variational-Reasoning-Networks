#ifndef cfg_H
#define cfg_H

#include <iostream>
#include <cstring>
#include <fstream>
#include <set>
#include <map>

#include "util/fmt.h"
#include "util/gnn_macros.h"
typedef float Dtype;
typedef gnn::CPU mode;

struct cfg
{
    static int iter, max_bp_iter, max_q_iter, nhop_subg;
    static int num_neg;
    static int n_hidden;
    static int test_tpok;
    static unsigned batch_size, dev_id;
    static unsigned n_embed;
    static unsigned max_iter; 
    static unsigned test_interval; 
    static unsigned report_interval; 
    static unsigned save_interval;
    static bool test_only;
    static bool vis_score;
    static Dtype lr;
    static Dtype p_pos;
    static Dtype l2_penalty; 
    static Dtype momentum;
    static Dtype margin; 
    static Dtype w_scale;
    static const char *save_dir, *data_root, *dataset, *loss_type, *net_type, *init_idx_file;

    static void LoadParams(const int argc, const char** argv)
    {
        for (int i = 1; i < argc; i += 2)
        {
            if (strcmp(argv[i], "-test_tpok") == 0)
		        test_tpok = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-data_root") == 0)
		        data_root = argv[i + 1];
            if (strcmp(argv[i], "-loss_type") == 0)
		        loss_type = argv[i + 1];                
            if (strcmp(argv[i], "-dataset") == 0)
                dataset = argv[i + 1];
            if (strcmp(argv[i], "-net_type") == 0)
                net_type = argv[i + 1];
		    if (strcmp(argv[i], "-lr") == 0)
		        lr = atof(argv[i + 1]);
            if (strcmp(argv[i], "-n_hidden") == 0)
                n_hidden = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-max_bp_iter") == 0)
                max_bp_iter = atoi(argv[i + 1]);                
            if (strcmp(argv[i], "-num_neg") == 0)
                num_neg = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-max_q_iter") == 0)
                max_q_iter = atoi(argv[i + 1]);     
            if (strcmp(argv[i], "-nhop_subg") == 0)
                nhop_subg = atoi(argv[i + 1]);    
            if (strcmp(argv[i], "-dev_id") == 0)
                dev_id = atoi(argv[i + 1]);                         
            if (strcmp(argv[i], "-cur_iter") == 0)
                iter = atoi(argv[i + 1]);                      
		    if (strcmp(argv[i], "-embed") == 0)
			    n_embed = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-max_iter") == 0)
	       		max_iter = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-batch_size") == 0)
	       		batch_size = atoi(argv[i + 1]);                   
		    if (strcmp(argv[i], "-int_test") == 0)
    			test_interval = atoi(argv[i + 1]);
    	   	if (strcmp(argv[i], "-int_report") == 0)
    			report_interval = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-int_save") == 0)
    			save_interval = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-l2") == 0)
    			l2_penalty = atof(argv[i + 1]);
            if (strcmp(argv[i], "-margin") == 0)
    			margin = atof(argv[i + 1]);                
            if (strcmp(argv[i], "-w_scale") == 0)
                w_scale = atof(argv[i + 1]);
    		if (strcmp(argv[i], "-m") == 0)
    			momentum = atof(argv[i + 1]);	
    		if (strcmp(argv[i], "-svdir") == 0)
    			save_dir = argv[i + 1];
    		if (strcmp(argv[i], "-init_idx_file") == 0)
    			init_idx_file = argv[i + 1];
    		if (strcmp(argv[i], "-test_only") == 0)
    			test_only = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-vis_score") == 0)
    			vis_score = atoi(argv[i + 1]);
        }
        
        if (vis_score)
        {
            std::cerr << "vis score" << std::endl;
            assert(iter);
        }
        if (test_only)
        {
            std::cerr << "test only" << std::endl;
            std::cerr << "test_tpok = " << test_tpok << std::endl;
            assert(iter);
        }
        if (init_idx_file)
        {
            std::cerr << "init network" << std::endl;
        }
        std::cerr << "net_type = " << net_type << std::endl;
        std::cerr << "n_hidden = " << n_hidden << std::endl;
        std::cerr << "dev_id = " << dev_id << std::endl;
        std::cerr << "loss_type = " << loss_type << std::endl;
        std::cerr << "nhop_subg = " << nhop_subg << std::endl;
        std::cerr << "margin = " << margin << std::endl;
        std::cerr << "num_neg = " << num_neg << std::endl;
        std::cerr << "max_q_iter = " << max_q_iter << std::endl;
        std::cerr << "max_bp_iter = " << max_bp_iter << std::endl;
        std::cerr << "batch_size = " << batch_size << std::endl;
        std::cerr << "dataset = " << dataset << std::endl;
        std::cerr << "n_embed = " << n_embed << std::endl;
        std::cerr << "max_iter = " << max_iter << std::endl;
    	std::cerr << "test_interval = " << test_interval << std::endl;
    	std::cerr << "report_interval = " << report_interval << std::endl;
    	std::cerr << "save_interval = " << save_interval << std::endl;
    	std::cerr << "lr = " << lr << std::endl;
        std::cerr << "w_scale = " << w_scale << std::endl;
    	std::cerr << "l2_penalty = " << l2_penalty << std::endl;
    	std::cerr << "momentum = " << momentum << std::endl;
    	std::cerr << "init iter = " << iter << std::endl;	
    }    
};

#endif
