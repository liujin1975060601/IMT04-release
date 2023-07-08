#include <assert.h>
#include <algorithm>
#include "../math/blas.h"
#include "../layers/layer.h"
#include "../network.h"
#include "../layers/convolutional_layer.h"
#include "model_trans.h"

bool cmpStart(LayerOperate lsh, LayerOperate rsh)
{
	if (lsh.start < rsh.start)
		return true;
	else
		return false;
}
void insert_weights_src(layer *pL,
	float *src_weights_gpu, float *src_weight_updates_gpu, float *src_biases_gpu, float *src_bias_updates_gpu,
	float *src_scales_gpu, float *src_rolling_mean_gpu, float *src_rolling_variance_gpu,
	int src_start, int n, int dst_start, float fill_val)
{
	int size2 = pL->size*pL->size;
	int channel_size = size2 * pL->c;
	if (src_weights_gpu)
	{
		copy_gpu(channel_size*n, src_weights_gpu + src_start*channel_size, 1, pL->weights_gpu + dst_start*channel_size, 1);
	}
	else
	{
		fill_gpu(channel_size*n, fill_val, pL->weights_gpu + dst_start*channel_size, 1);
	}
}
int expand_layer_weights(layer *pL, std::vector<LayerOperate> &ops)
{
	assert(pL->n == pL->out_c);
	int expand_n = 0;
	for (int i = 0; i < ops.size();)
	{
		if (ops[i].start > pL->n)
		{
			ops.erase(ops.begin() + i);
		}
		else
		{
			expand_n += ops[i].n;
			i++;
		}
	}
	std::sort(ops.begin(), ops.end(), cmpStart);
	//
	if (expand_n > 0)
	{
		int size2 = pL->size*pL->size;
		int new_nweights = (pL->c / pL->groups) * (pL->n + expand_n) * size2;
		float *src_weights_gpu = pL->weights_gpu; pL->weights_gpu = cuda_make_array(0, new_nweights);
		float *src_weight_updates_gpu = pL->weight_updates_gpu; if (pL->weight_updates_gpu) pL->weight_updates_gpu = cuda_make_array(0, new_nweights);
		float *src_biases_gpu = pL->biases_gpu;
		float *src_bias_updates_gpu = pL->bias_updates_gpu;
		if (pL->biases_gpu)
		{
			pL->biases_gpu = cuda_make_array(0, pL->n + expand_n);
			if (pL->bias_updates_gpu)pL->bias_updates_gpu = cuda_make_array(0, pL->n + expand_n);
		}
		float *src_scales_gpu = pL->scales_gpu;
		float *src_rolling_mean_gpu = pL->rolling_mean_gpu;
		float *src_rolling_variance_gpu = pL->rolling_variance_gpu;
		if (pL->batch_normalize)
		{
			pL->scales_gpu = cuda_make_array(0, pL->n + expand_n);
			pL->rolling_mean_gpu = cuda_make_array(0, pL->n + expand_n);
			pL->rolling_variance_gpu = cuda_make_array(0, pL->n + expand_n);
		}
		//
		int ops_pt = 0;
		int dst_start = 0;
		for (int src_start = 0; src_start < pL->n; src_start++)
		{
			while (ops_pt < ops.size() && ops[ops_pt].start <= src_start)
			{
				insert_weights_src(pL,
					0, 0, 0, 0,
					0, 0, 0,
					0, ops[ops_pt].n, dst_start, 0);
				ops_pt++;
				dst_start += ops[ops_pt].n;
			}
			insert_weights_src(pL,
				src_weights_gpu, src_weight_updates_gpu, src_biases_gpu, src_bias_updates_gpu,
				0, 0, 0,
				src_start, 1, dst_start, 0);
			dst_start++;
		}
		if (ops_pt<ops.size() && ops[ops_pt].start == pL->n)
		{
			insert_weights_src(pL,
				0, 0, 0, 0,
				0, 0, 0,
				0, ops[ops_pt].n, dst_start, 0);
			dst_start += ops[ops_pt].n;
		}
		assert(dst_start == pL->n + expand_n);

		if (pL->batch_normalize)
		{
			cuda_free(src_scales_gpu);
			cuda_free(src_rolling_mean_gpu);
			cuda_free(src_rolling_variance_gpu);
		}
	}
	//
	expand_filter(pL, expand_n);
	//
	return expand_n;
}

void insert_weights_follow(layer *pL,
	int channel, float *src_weights_gpu, float *src_weight_updates_gpu,
	int expand_n,
	int src_start, int n, int dst_start, float fill_val)
{
	int size2 = pL->size*pL->size;
	int channel_size = size2 * pL->c;
	int src_offset = channel * channel_size + src_start*size2;
	int dst_channel_size = size2 * (pL->c + expand_n);
	int dst_offset = channel*dst_channel_size + dst_start*size2;
	assert(dst_start + n <= pL->c + expand_n);
	if (src_weights_gpu)
	{
		copy_gpu(size2*n, src_weights_gpu + src_offset, 1, pL->weights_gpu + dst_offset, 1);
	}
	else
	{
		fill_gpu(size2*n, fill_val, pL->weights_gpu + dst_offset, 1);
	}
}
int expand_layer_weights_follow(layer *pL, std::vector<LayerOperate> &ops)
{
	assert(pL->n == pL->out_c);
	int expand_n = 0;
	std::sort(ops.begin(), ops.end(), cmpStart);
	//
	if (expand_n > 0)
	{
		int size2 = pL->size*pL->size;
		int new_nweights = ((pL->c + expand_n) / pL->groups) * pL->n * size2;
		float *src_weights_gpu = pL->weights_gpu; pL->weights_gpu = cuda_make_array(0, new_nweights);
		float *src_weight_updates_gpu = pL->weight_updates_gpu; if (pL->weight_updates_gpu) pL->weight_updates_gpu = cuda_make_array(0, new_nweights);
		//
		for (int i = 0; i < pL->n; i++)
		{
			int ops_pt = 0;
			int dst_start = 0;
			for (int src_start = 0; src_start < pL->c; src_start++)
			{
				insert_weights_follow(pL, i,
					0, 0, expand_n,
					src_start, 1, dst_start, 0);
				dst_start++;
			}
			if (ops_pt < ops.size() && ops[ops_pt].start == pL->c)
			{
				insert_weights_follow(pL, i,
					0, 0, expand_n,
					0, ops[ops_pt].n, dst_start, 0);
				dst_start += ops[ops_pt].n;
			}
			assert(dst_start == pL->c + expand_n);
		}
		//
		cuda_free(src_weights_gpu);
		if (src_weight_updates_gpu)cuda_free(src_weight_updates_gpu);
	}
	//
	return expand_n;
}


void expand_network_weights1(network *net, int start, float val)
{
	layer *pStart = &net->layers[start];
	if (start + 1<net->n)
	{
		layer *pL = &net->layers[start + 1];
		if (pL->type == CONVOLUTIONAL || pL->type == CONVOLUTIONAL_ADV)
		{
			assert(pL->groups == 1);
			int expand_n = pStart->out_c - pL->c;
			if (expand_n > 0)
			{
				int size2 = pL->size*pL->size;
				assert(pL->n == pL->out_c);
				pL->weights_gpu = gpu_expand_groups(pL->weights_gpu, size2, pL->c, expand_n, pL->n, 0);

				pL->c += expand_n;
				assert(pL->c == pStart->out_c);
				pL->nweights += size2*expand_n*pL->n;
				assert(pL->nweights == pL->n*pL->c*size2);
				pL->inputs += pL->w*pL->h*expand_n;
				assert(pL->inputs == pL->w*pL->h*pL->c);

				pull_convolutional_layer(*pL);
				pL->bflops = (2.0 * pL->n * size2*pL->c / pL->groups * pL->out_h*pL->out_w) / 1000000000.;
				pL->workspace_size = get_workspace_size(*pL);
			}
		}
	}
}
int expand_layer_weights_follow(layer *pStart, layer *pL, std::vector<LayerOperate> &ops)
{
	assert(pL->groups == 1);
	int expand_n = pStart->out_c - pL->c;
	if (expand_n > 0)
	{
		int expand_n_dst = expand_layer_weights_follow(pL, ops);

		int size2 = pL->size*pL->size;
		assert(pL->n == pL->out_c);

		pL->c += expand_n;
		assert(pL->c == pStart->out_c);
		pL->nweights += size2*expand_n*pL->n;
		assert(pL->nweights == pL->n*pL->c*size2);

#ifdef CUDNN
		cudnn_convolutional_setup(pL, cudnn_fastest, 0);
#endif
		if (pL->weights)free(pL->weights); pL->weights = (float*)malloc(pL->nweights*sizeof(float));
		if (pL->weight_updates)
		{
			free(pL->weight_updates); pL->weight_updates = (float*)malloc(pL->nweights*sizeof(float));
		}
		pull_convolutional_layer(*pL);

		pL->bflops = (2.0 * pL->n * size2*pL->c / pL->groups * pL->out_h*pL->out_w) / 1000000000.;
		pL->workspace_size = get_workspace_size(*pL);
	}
	return expand_n;
}
void expand_network_weights_follow_simple(network *net, int start, std::vector<LayerOperate> &ops)
{
	int expand_n_globe = -1;
	layer *pStart = &net->layers[start];
	if (start + 1<net->n)
	{
		layer *pL = &net->layers[start + 1];
		if (pL->type == CONVOLUTIONAL || pL->type == CONVOLUTIONAL_ADV)
		{
			int expand_n = expand_layer_weights_follow(pStart, pL, ops);
			assert(expand_n_globe == -1 || expand_n == expand_n_globe);
			expand_n_globe = expand_n;
		}
	}
}

void expand_network_weights_follow(network *net, int start, std::vector<LayerOperate> &ops)
{
	std::vector<NodeLayer> layers_ops(net->n);
	layers_ops[start].ops = ops;
	layer *pStart = &net->layers[start];
	int expand_n_globe = -1;
	for (int i = start + 1; i < net->n; i++)
	{
		layer *pL = &net->layers[i];
		if (pL->type == CONVOLUTIONAL)
		{
			int fid = -1, fid_lay_idx = -1, fid_offset_c = -1;
			int offset_c = 0;
			for (int j = 0; j < pL->n; j++)
			{
				int lay_idx = pL->input_layers[j];
				assert(lay_idx >= 0 && lay_idx<net->n);
				if (layers_ops[lay_idx].ops.size()>0)
				{
					fid = j;
					fid_lay_idx = lay_idx;
					fid_offset_c = offset_c;
					break;
				}
				offset_c += net->layers[lay_idx].out_c;
			}
			if (fid >= 0)
			{
				assert(fid < pL->n && fid_lay_idx == pL->input_layers[fid]);
				int old_size = pL->input_sizes[fid];
				pL->input_sizes[fid] = net->layers[fid_lay_idx].outputs;
				assert(pL->input_sizes[fid]>old_size);
				pL->outputs /= pL->groups;
				assert(pL->outputs % (pL->out_w*pL->out_h) == 0);
				int old_out_c = pL->out_c;

				assert(pL->input_sizes[fid] % (pL->out_w*pL->out_h) == 0);
				assert((pL->input_sizes[fid] - old_size) % (pL->out_w*pL->out_h) == 0);
				pL->inputs = pL->outputs;
				//
				assert(fid_lay_idx >= 0 && fid_lay_idx<net->n);
				assert(fid_offset_c >= 0 && fid_offset_c <= pL->out_c);
				layers_ops[i].ops = layers_ops[fid_lay_idx].ops;
			}
			else
			{
				assert(fid == -1 && offset_c == pL->out_c);
			}
		}
		else
		{
			if ((i == start + 1 || i>start && layers_ops[i - 1].ops.size()>0) && (pL->type == CONVOLUTIONAL || pL->type == CONVOLUTIONAL_ADV))
			{
				pStart = &net->layers[i - 1];
				assert(i == start + 1 || pStart->type == ROUTE);
				int expand_n = expand_layer_weights_follow(pStart, pL, layers_ops[i].ops);
				assert(expand_n_globe == -1 || expand_n == expand_n_globe);
				expand_n_globe = expand_n;
			}
		}
	}
}


#include "../layers/convolutional_layer.h"
void realloc_layer(layer *pL, int realloc_roll_value, int adam)
{
	int steps = 1, total_batch = pL->batch*steps;
	if (pL->activation == SWISH || pL->activation == MISH)
	{
		if (pL->activation_input)free(pL->activation_input);
		pL->activation_input = (float*)calloc(total_batch*pL->outputs, sizeof(float));
		if (pL->activation_input_gpu)cuda_free(pL->activation_input_gpu);
		pL->activation_input_gpu = cuda_make_array(NULL, total_batch*pL->outputs);
	}
	if (pL->batch_normalize)
	{
		if (pL->mean)free(pL->mean); pL->mean = (float *)calloc(pL->n, sizeof(float));
		if (pL->variance)free(pL->variance); pL->variance = (float *)calloc(pL->n, sizeof(float));

		if (pL->mean_delta)free(pL->mean_delta); pL->mean_delta = (float *)calloc(pL->n, sizeof(float));
		if (pL->variance_delta)free(pL->variance_delta); pL->variance_delta = (float *)calloc(pL->n, sizeof(float));

		//pL->mean_delta_gpu = cuda_make_array(pL->mean, n);
		//pL->variance_delta_gpu = cuda_make_array(pL->variance, n);

		assert(pL->outputs == pL->out_h*pL->out_w*pL->n);
		if (pL->x_norm_gpu)
		{
			cuda_free(pL->x_norm_gpu);
			pL->x_norm_gpu = cuda_make_array(pL->output, pL->batch*pL->outputs);
		}
	}
	if (adam)
	{
		if (pL->m)free(pL->m); pL->m = (float *)calloc(pL->nweights, sizeof(float));
		if (pL->v)free(pL->v); pL->v = (float *)calloc(pL->nweights, sizeof(float));
		if (pL->bias_m)free(pL->bias_m); pL->bias_m = (float *)calloc(pL->n, sizeof(float));
		if (pL->scale_m)free(pL->scale_m); pL->scale_m = (float *)calloc(pL->n, sizeof(float));
		if (pL->bias_v)free(pL->bias_v); pL->bias_v = (float *)calloc(pL->n, sizeof(float));
		if (pL->scale_v)free(pL->scale_v); pL->scale_v = (float *)calloc(pL->n, sizeof(float));
	}
	//
	if (pL->activation == SWISH || pL->activation == MISH)
	{
		if (pL->activation_input_gpu)cuda_free(pL->activation_input_gpu); pL->activation_input_gpu = cuda_make_array(pL->activation_input, total_batch*pL->outputs);
	}
	//
	assert(pL->nweights == (pL->c / pL->groups) * pL->n * pL->size * pL->size);
	assert(pL->outputs == pL->out_c*pL->out_w*pL->out_h);
	//
	pL->bflops = (2.0 * pL->n * pL->size*pL->size*pL->c / pL->groups * pL->out_h*pL->out_w) / 1000000000.;
}


void expand_filter(layer *pL, int expand_n)
{
	assert(pL->outputs = pL->out_h * pL->out_w * pL->out_c);
	int expand_unit = pL->out_h * pL->out_w * expand_n * pL->batch;
	int size2 = pL->size * pL->size;
	assert(pL->n == pL->out_c);
	//GPU
	{
		if (pL->output_gpu)cuda_free(pL->output_gpu);
		pL->output_gpu = cuda_make_array(0, pL->outputs*pL->batch + expand_unit);
		if (pL->delta_gpu)cuda_free(pL->delta_gpu);
		pL->delta_gpu = cuda_make_array(0, pL->outputs*pL->batch + expand_unit);
	}
	//CPU
	int new_nweights = (pL->c / pL->groups) * (pL->n + expand_n) * size2;
	if (expand_n > 0)
	{
		if (pL->output)free(pL->output); pL->output = (float *)malloc((pL->outputs*pL->batch + expand_unit)*sizeof(float));
		if (pL->delta)free(pL->delta); pL->delta = (float *)malloc((pL->outputs*pL->batch + expand_unit)*sizeof(float));
		if (pL->weights)free(pL->weights); pL->weights = (float *)malloc(new_nweights*sizeof(float));

		if (pL->biases)
		{
			pL->biases = (float *)malloc((pL->n + expand_n)*sizeof(float));
			if (pL->bias_updates)pL->bias_updates = (float *)malloc((pL->n + expand_n)*sizeof(float));
		}
		if (pL->batch_normalize)
		{
			pL->scales = (float *)malloc((pL->n + expand_n)*sizeof(float));
			pL->rolling_mean = (float *)malloc((pL->n + expand_n)*sizeof(float));
			pL->rolling_variance = (float *)malloc((pL->n + expand_n)*sizeof(float));
		}
	}
	//
	pL->out_c += expand_n;
	pL->n += expand_n;
	pL->outputs += expand_n*pL->out_h * pL->out_w;
	assert(pL->outputs = pL->out_h * pL->out_w * pL->out_c);
	pL->nbiases = pL->n;
	pL->nweights = (pL->c / pL->groups) * pL->n * size2;
	assert(pL->nweights == new_nweights);
	//
	realloc_layer(pL, 0, 0);
	//
	pL->workspace_size = get_workspace_size(*pL);
}
void cut_filter(layer *pL, int filter_id)
{
	assert(filter_id >= 0 && filter_id < pL->out_c);
	assert(pL->outputs = pL->out_h * pL->out_w * pL->out_c);
	//int expand_unit = pL->out_h * pL->out_w * expand_n * pL->batch;
	int size2 = pL->size * pL->size;
	assert(pL->n == pL->out_c);
	//GPU
	{
		pL->output_gpu = gpu_del(pL->output_gpu, pL->outputs*pL->batch, -1, pL->batch*pL->out_h * pL->out_w);
		pL->delta_gpu = gpu_del(pL->delta_gpu, pL->outputs*pL->batch, -1, pL->batch*pL->out_h * pL->out_w);
		//void get_channel_weight(convolutional_layer *pL, float *c_weight)
		pL->weights_gpu = gpu_del(pL->weights_gpu, pL->n*pL->c*size2, filter_id*pL->c*size2, pL->c*size2);
		if (pL->biases_gpu)
		{
			pL->biases_gpu = gpu_del(pL->biases_gpu, pL->n, filter_id, 1);
			if (pL->bias_updates_gpu)pL->bias_updates_gpu = gpu_del(pL->bias_updates_gpu, pL->n, filter_id, 1);
		}
		if (pL->batch_normalize)
		{
			pL->scales_gpu = gpu_del(pL->scales_gpu, pL->n, filter_id, 1);
			pL->rolling_mean_gpu = gpu_del(pL->rolling_mean_gpu, pL->n, filter_id, 1);
			pL->rolling_variance_gpu = gpu_del(pL->rolling_variance_gpu, pL->n, filter_id, 1);
		}
	}
	//CPU
	{
		pL->output = cpu_del(pL->output, pL->outputs*pL->batch, -1, pL->batch*pL->out_h * pL->out_w);
		//void get_channel_weight(convolutional_layer *pL, float *c_weight)
		pL->weights = cpu_del(pL->weights, pL->n*pL->c*size2, filter_id*pL->c*size2, pL->c*size2);
		if (pL->biases)
		{
			pL->biases = cpu_del(pL->biases, pL->n, filter_id, 1);
			if (pL->bias_updates)pL->bias_updates = cpu_del(pL->bias_updates, pL->n, filter_id, 1);
		}
		if (pL->batch_normalize)
		{
			pL->scales = cpu_del(pL->scales, pL->n, filter_id, 1);
			pL->rolling_mean = cpu_del(pL->rolling_mean, pL->n, filter_id, 1);
			pL->rolling_variance = cpu_del(pL->rolling_variance, pL->n, filter_id, 1);
		}
	}
	//
	pL->out_c--;
	pL->n--;
	pL->outputs -= pL->out_h * pL->out_w;
	assert(pL->outputs = pL->out_h * pL->out_w * pL->out_c);
	pL->nbiases = pL->n;
	pL->nweights = (pL->c / pL->groups) * pL->n * size2;
	//
	realloc_layer(pL, 0, 0);
	//
	pL->workspace_size = get_workspace_size(*pL);
}



int operate_cfg(network *net, std::vector<CfgOperate> &ops)
{
	int old_workspace_size = net->workspace_size;
	int succes_count = 0;
	for (int i = 0; i < ops.size(); i++)
	{
		CfgOperate *pOp = &ops[i];
		layer *pL = &net->layers[pOp->lay_id];
		int legal = 0;
		switch (pOp->op_type)
		{
		case 0://sub
		{
			if (pL->weights_gpu && pL->type == CONVOLUTIONAL)
			{
				cut_filter(pL, pOp->op_n);
				legal = 1;
				succes_count++;
			}
		}
		break;
		case 1://add
		{
			if (pL->weights_gpu && pL->type == CONVOLUTIONAL)
			{
				std::vector<LayerOperate> ops;
				LayerOperate a; a.start = pL->n; a.n = pOp->op_n;
				ops.push_back(a);
				int expand_n = expand_layer_weights(pL, ops);
				expand_network_weights_follow(net, pOp->lay_id, ops);
				//
				legal = 1;
				succes_count++;
			}
		}
		break;
		case 2://jmp
		{
			;
		}
		break;
		}
		if (legal && pL->workspace_size > net->workspace_size)
			net->workspace_size = pL->workspace_size;
	}
	if (net->workspace_size != old_workspace_size)
		refresh_workspace(net);
	return succes_count;
}