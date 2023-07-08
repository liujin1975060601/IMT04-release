#pragma once

#include "../layers/layer.h"
#include "../network.h"

void expand_filter(layer *pL, int expand_n);
void cut_filter(layer *pL, int filter_id);

typedef struct LayerOperate
{
	int start;
	int n;
}LayerOperate;
typedef struct NodeLayer
{
	int start;
	std::vector<LayerOperate> ops;
}NodeLayer;

int expand_layer_weights(layer *pL, std::vector<LayerOperate> &ops);
int expand_layer_weights_follow(layer *pL, std::vector<LayerOperate> &ops);

void expand_network_weights1(network *net, int start, float val);
void expand_network_weights_follow_simple(network *net, int start, std::vector<LayerOperate> &ops);
void expand_network_weights_follow(network *net, int start, std::vector<LayerOperate> &ops);



typedef struct CfgOperate
{
	int lay_id;
	int op_type;//1=add 0=sub 2=jmp
	int op_n;
}CfgOperate;
int operate_cfg(network *net, std::vector<CfgOperate> &ops);