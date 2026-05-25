import sys
from safetensors.torch import load_model
import time
import torch
import torch.distributed as dist
import argparse
import os, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xop.ops.dsl.pkt_util import TorchRef, PerfReporter, Qwen3Info
from xop.ops.dsl.mk_layers import MkLayers, MkLayersHybridLayout
import xop.util as xutil

# os.environ["PYTORCH_CUDA_GRAPH_CAPTURE_DEBUG"] = "2"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ['CUDA_GRAPH_STRICT'] = '1'        
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
# torch.cuda.graphs.set_debug_level("STRICT")    # PyTorch 官方严格模式
# torch.cuda.graphs.set_require_complete(True)  # 必须完整捕获，否则抛异常

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.getenv("XOP_HOME", default=None)+"/megakernel/gen", help="Output files directory")
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
    parser.add_argument("--profiling", action="store_true", help="Use Profiler to generate trace")
    parser.add_argument("--nc", action="store_true", help="no-compile: Use the specified compiled library instead of recompiling it")
    args = parser.parse_args()
    
    torch.set_default_dtype(torch.bfloat16)

    model_tag = "qwen3_06b"
    batch = 1
    q_seqlen = 1
    max_kv_seqlen = 4096
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, num_hidden_layers \
        = Qwen3Info.get_basic_params(model_tag)
    model, tokenizer = Qwen3Info.load_model(0, model_tag, page_size=max_kv_seqlen)
    
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    all_position_embeddings = model.model.rotary_emb(positions)
 
 
    hidden_states = torch.randn((batch, q_seqlen, hidden_size), dtype=torch.bfloat16, device="cuda")
    attention_mask = None    
    step = torch.full((1, ), 0, dtype=torch.int32, device="cuda")
    stream = None
    print("before forward", hidden_states)
    
    layer_num = num_hidden_layers
    layers = MkLayers(model_tag, instance_id=0, kernel_num=layer_num, world_size=1, rank=0, max_batch_size=1, trace_name=args.trace_name, profiling=args.profiling)
    params, io_pt, public_pt = MkLayers.qwen3_alloc_torch_buffer(model_tag, layer_num, batch, q_seqlen=1, max_kv_seqlen=max_kv_seqlen)   
    layers.qwen3_create_decoder_layer(model, layer_num, params, io_pt, public_pt, is_long_kv=True, is_no_compile=args.nc, output_dir=args.output_dir)
    
    # layers = MkLayersHybridLayout(model_tag, instance_num=2, kernel_num=layer_num, world_size=1, rank=0, max_batch_size=1, trace_name=args.trace_name, profiling=args.profiling)
    # layers.qwen3_create_decoder_layer(model_tag, model, layer_num, batch, is_no_compile=args.nc, output_dir=args.output_dir)

    #####################################
    cur_pos = 65
    position_embeddings=(all_position_embeddings[0][:, cur_pos-1:cur_pos], all_position_embeddings[1][:, cur_pos-1:cur_pos])

    def mk_run_one_step(iter):
        mk_out = layers(cur_pos, position_embeddings, hidden_states.view(batch*q_seqlen, hidden_size))
        return mk_out#, layers.public_pt.key_cache_5d, layers.public_pt.value_cache_5d 
    
    def torch_ref_tmp_one_step(iter):
        step.fill_(cur_pos - 1)
        torch_io = hidden_states.clone()
        with torch.no_grad():
            for layer_id in range(layer_num):
                out = model.model.layers[layer_id].forward(torch_io, attention_mask, position_embeddings, step, stream)
                torch_io = out[0]
            return torch_io#, model.model.kv_cache[0], model.model.kv_cache[1]
    # ######################################
    # start_pos = 65
    # decode_limit = 66
    # def mk_run_multi_step():
    #     mk_io = hidden_states.view(batch*q_seqlen, hidden_size).clone()
    #     for cur_pos in range(start_pos, decode_limit):
    #         position_embeddings=(all_position_embeddings[0][:, cur_pos-1:cur_pos], all_position_embeddings[1][:, cur_pos-1:cur_pos])
    #         mk_out = layers(cur_pos, position_embeddings, mk_io)
    #         mk_io.copy_(mk_out)
    #     return mk_out, layers.public_pt.key_cache_5d, layers.public_pt.value_cache_5d # [layer_num, batch, kv_seqlen, kv_heads, head_dim]

    # def torch_ref_tmp_multi_step():
    #     torch_io = hidden_states.clone()    
    #     with torch.no_grad():
    #         for cur_pos in range(start_pos, decode_limit):
    #             step.fill_(cur_pos - 1)
    #             position_embeddings=(all_position_embeddings[0][:, cur_pos-1:cur_pos], all_position_embeddings[1][:, cur_pos-1:cur_pos])
    #             for layer_id in range(layer_num):
    #                 out = model.model.layers[layer_id].forward(torch_io, attention_mask, position_embeddings, step, stream)
    #                 torch_io = out[0]
    #         return torch_io, model.model.kv_cache[0], model.model.kv_cache[1]
    # ######################################
    
    mk_run = mk_run_one_step
    torch_ref_tmp = torch_ref_tmp_one_step
    
    # class PersistentKernelWrapper(torch.autograd.Function):
    #     @staticmethod
    #     def forward():
    #         stream = torch.cuda.Stream()
    #         with torch.cuda.stream(stream):
    #             output = mk_run_one_step()
    #         return output

    # cuda_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(cuda_graph, stream=stream):
    #     output = PersistentKernelWrapper.forward()
    
    # # stream = torch.cuda.Stream()
    # # with torch.cuda.stream(stream):
    # #     with torch.cuda.graph(cuda_graph := torch.cuda.CUDAGraph(), capture_error_mode="relaxed"):
    # #         output = mk_run_one_step()

    # torch.cuda.synchronize()
    # for _ in range(10):
    #     cuda_graph.replay()  # <-- 直接运行你刚才捕获的函数
    #     print("output", output)
    
    # def mk_run():
    #     cuda_graph.replay()
    #     return output 
    # mk_run = mk_run_multi_step
    # torch_ref_tmp = torch_ref_tmp_multi_step
    
    if 0:
        graph, ref_output = TorchRef.compile_capture(torch_ref_tmp, is_compile=False) # 搜 “kv_seq_len = step + 1” 改成=> 1
        def torch_ref(iter):
            graph.replay()
            return ref_output
    else:
        def torch_ref(iter):
            return torch_ref_tmp()
    
    # for i in range(10):
    #     torch_ref()
    
    # torch_ref()    
    # mk_run()
    
    # print("mk_out: ", mk_run())    
    # print("torch_ref: ", torch_ref())
    
    
    # q1,k1,v1 = mk_run() # [:, 0, :, :, :]
    # q2,k2,v2 = torch_ref()
    # # # print("q", PerfReporter.assert_similar(q1, q2))
    # # # print("k", PerfReporter.assert_similar(k1, k2))
    # # # print("v", PerfReporter.assert_similar(v1, v2))
    
    # # 第1层没问题，第二层的split方案的结果不对
    # print("mpk query0:\n", q1)
    # print("torch query0:\n", q2)
    
    # print("mpk key0:\n", k1[0, 0, cur_pos-1:cur_pos, :, :])
    # print("torch key0:\n", k2[0, 0, cur_pos-1:cur_pos, :, :])
    # print("mpk key1:\n", k1[1, 0, cur_pos-1:cur_pos, :, :])
    # print("torch key1:\n", k2[1, 0, cur_pos-1:cur_pos, :, :])
    
    # print("mpk value0:\n", v1[0, 0, cur_pos-1:cur_pos, :, :])
    # print("torch value0:\n", v2[0, 0, cur_pos-1:cur_pos, :, :])
    # print("mpk value1:\n", v1[1, 0, cur_pos-1:cur_pos, :, :])
    # print("torch value1:\n", v2[1, 0, cur_pos-1:cur_pos, :, :])
    
    # print("mpk key0:\n", k1[0, 0, start_pos-1:decode_limit-1, :, :])
    # print("torch key0:\n", k2[0, 0, start_pos-1:decode_limit-1, :, :])
    # print("mpk key1:\n", k1[1, 0, start_pos-1:decode_limit-1, :, :])
    # print("torch key1:\n", k2[1, 0, start_pos-1:decode_limit-1, :, :])
 
    xutil.profile(mk_run, torch_ref)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cnt = 100

    torch.cuda.synchronize()
    # starter.record()
    start_time = time.perf_counter()    
    for i in range(cnt):
        torch_ref()
    # ender.record()
    torch.cuda.synchronize()
    # print("torch time: ", starter.elapsed_time(ender) / cnt)
    print("torch time: ", (time.perf_counter()-start_time)*1000 / cnt)

    # torch.cuda.synchronize()
    # # starter.record()      
    # start_time = time.perf_counter()
    # for i in range(cnt):
    #     mk_run()
    # # ender.record()
    # torch.cuda.synchronize()
    # # print("mpk time: ", starter.elapsed_time(ender) / cnt)
    # print("mpk time: ", (time.perf_counter()-start_time)*1000 / cnt)
    
    # torch.cuda.synchronize()
    # start_time = time.perf_counter()          
    # for i in range(cnt):
    #     mk_run()
    # torch.cuda.synchronize()
    # print("mpk time: ", (time.perf_counter()-start_time)*1000 / cnt)
    
    t = 0
    for i in range(cnt):
        torch.cuda.synchronize()
        start_time = time.perf_counter()  
        mk_run()
        torch.cuda.synchronize()
        t += (time.perf_counter()-start_time)*1000    
    print("mpk time: ", t / cnt)
    
