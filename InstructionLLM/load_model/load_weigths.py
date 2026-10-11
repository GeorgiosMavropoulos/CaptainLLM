### this file contains the load weights class. This class contains the main function which loads the pretrained weigths from gpt 2 into our gpt model
import numpy as np
import torch

class LoadWeigths:
    def __init__(self):
        pass


    #left side is our trainable weigths and the right the ones we want to load (the pretrained ones from gpt)
    def assign(left, right): #this method returns an error message if left tensor does not has the same shape with the right one
       if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
       "Right: {right.shape}"
       )
       return torch.nn.Parameter(torch.tensor(right)) #create the right shape into a tensor since we want to load it

    @staticmethod
    def load_weights_into_gpt(gpt, params):
        gpt.pos_emb.weight = LoadWeigths.assign(gpt.pos_emb.weight, params['wpe']) #set positional and token embeddings to those specified in params
        gpt.tok_emb.weight = LoadWeigths.assign(gpt.tok_emb.weight, params['wte'])
        for b in range(len(params["blocks"])): #iterate over each transformer block in the model
            #The np.split function is used to divide the attention and bias weights into three equal parts for the query, key, and value components.
            q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            gpt.trf_blocks[b].attention.W_query.weight = LoadWeigths.assign(
            gpt.trf_blocks[b].attention.W_query.weight, q_w.T)
            gpt.trf_blocks[b].attention.W_key.weight = LoadWeigths.assign(
            gpt.trf_blocks[b].attention.W_key.weight, k_w.T)
            gpt.trf_blocks[b].attention.W_value.weight = LoadWeigths.assign(
            gpt.trf_blocks[b].attention.W_value.weight, v_w.T)
            q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            gpt.trf_blocks[b].attention.W_query.bias = LoadWeigths.assign(
            gpt.trf_blocks[b].attention.W_query.bias, q_b)
            gpt.trf_blocks[b].attention.W_key.bias = LoadWeigths.assign(
            gpt.trf_blocks[b].attention.W_key.bias, k_b)
            gpt.trf_blocks[b].attention.W_value.bias = LoadWeigths.assign(
            gpt.trf_blocks[b].attention.W_value.bias, v_b)
            gpt.trf_blocks[b].attention.out_proj.weight = LoadWeigths.assign(
            gpt.trf_blocks[b].attention.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].attention.out_proj.bias = LoadWeigths.assign(
            gpt.trf_blocks[b].attention.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
            gpt.trf_blocks[b].ff.layers[0].weight = LoadWeigths.assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = LoadWeigths.assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = LoadWeigths.assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = LoadWeigths.assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
            gpt.trf_blocks[b].norm1.scale = LoadWeigths.assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = LoadWeigths.assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = LoadWeigths.assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = LoadWeigths.assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

            #change our model's random weigths with gpt's
            gpt.final_norm.scale = LoadWeigths.assign(gpt.final_norm.scale, params["g"])
            gpt.final_norm.shift = LoadWeigths.assign(gpt.final_norm.shift, params["b"])
            gpt.out_head.weight = LoadWeigths.assign(gpt.out_head.weight, params["wte"])


