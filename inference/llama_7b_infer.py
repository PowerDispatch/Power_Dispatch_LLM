import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import conf as co
from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments
)
from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm, inference_stream_vllm
)
from swift.tuners import Swift
from swift.llm import infer_main, sft_main, app_ui_main
import gradio as gr
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from swift.llm.utils import (History, InferArguments, inference_stream, limit_history_length)

best_ckpt_dir = '/tf/python/output/llama2-7b-chat/v1-20231127-095459/checkpoint-22138/'
# model_path = '/tf/model/Llama-2-7b-chat-ms/'
model_type = ModelType.llama2_7b_chat
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type, inference_stream
)

template_type = get_default_template_type(model_type)


# llm_engine = get_vllm_engine(model_type, model_dir='/tf/python/output/llama2-7b-chat/v1-20231127-095459/checkpoint-22138-merged')
llm_engine = get_vllm_engine(model_type, model_dir=co.best_ckpt_dir)
template = get_template(template_type, llm_engine.tokenizer)
llm_engine.generation_config.max_new_tokens = 2048

request_list = [{'query': 'how to compute opf'}, {'query': 'who are u?'}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")


def clear_session() -> History:
    return []

def llamma_bot(content):
    return content

# def gradio_chat_demo():
#     def model_chat(query: str, history: History) -> Tuple[str, History]:
#         old_history, history = limit_history_length(template, query, history, 2048)
#         gen = inference_stream_vllm(llm_engine, template,
#                                     [{
#                                         'query': query,
#                                         'history': history
#                                     }])
#         for resp_list in gen:
#             history = resp_list[0]['history']
#             total_history = old_history + history
#             yield '', total_history

#     model_name = model_type
#     with gr.Blocks() as demo:
#         gr.Markdown(f'<center><font size=8>{model_name} Bot</center>')

#         chatbot = gr.Chatbot(label=f'{model_name}')
#         message = gr.Textbox(lines=2, label='Input')
#         with gr.Row():
#             clear_history = gr.Button('🧹 清除历史对话')
#             send = gr.Button('🚀 发送')
#         send.click(
#             model_chat, inputs=[message, chatbot], outputs=[message, chatbot])
#         clear_history.click(
#             fn=clear_session, inputs=[], outputs=[chatbot], queue=False)
#     demo.queue().launch(height=1000, share=False, server_port=8080, server_name='0.0.0.0')
#     return demo

# demo = gradio_chat_demo()
    
# def gradio_chat_demo():
#     def model_chat(query: str, history: History) -> Tuple[str, History]:
       
#         old_history, history = limit_history_length(template, query, history, 2048)
#         gen = inference_stream_vllm(llm_engine, template,
#                                     [{
#                                         'query': query,
#                                         'history': history
#                                     }])
        
#         for resp_list in gen:
#             history = resp_list[0]['history']
#             total_history = old_history + history
#             yield '', total_history

#     def Factualness(history: History) -> Tuple[History]:
#         query = co.actual
#         old_history, history = limit_history_length(template, query, history, 2048)
#         gen = inference_stream_vllm(llm_engine, template,
#                                     [{
#                                         'query': query,
#                                         'history': history
#                                     }])
#         for resp_list in gen:
#             history = resp_list[0]['history']
#             total_history = old_history + history
#             yield total_history


#     def safty(history: History) -> Tuple[History]:
#         query = co.safty
#         old_history, history = limit_history_length(template, query, history, 2048)
#         gen = inference_stream_vllm(llm_engine, template,
#                                     [{
#                                         'query': query,
#                                         'history': history
#                                     }])
#         for resp_list in gen:
#             history = resp_list[0]['history']
#             total_history = old_history + history
#             yield total_history
    

#     def power_output_adjustment(history: History) -> Tuple[History]:
#         query = co.power_adjust
#         old_history, history = limit_history_length(template, query, history, 2048)
#         gen = inference_stream_vllm(llm_engine, template,
#                                     [{
#                                         'query': query,
#                                         'history': history
#                                     }])
#         for resp_list in gen:
#             history = resp_list[0]['history']
#             total_history = old_history + history
#             yield total_history
    

#     def voltage_excursion(history: History) -> Tuple[History]:
#         query = co.voltage
#         old_history, history = limit_history_length(template, query, history, 2048)
#         gen = inference_stream_vllm(llm_engine, template,
#                                     [{
#                                         'query': query,
#                                         'history': history
#                                     }])
#         for resp_list in gen:
#             history = resp_list[0]['history']
#             total_history = old_history + history
#             yield total_history
    

#     def restore_order(history: History) -> Tuple[History]:
#         query = co.restored
#         old_history, history = limit_history_length(template, query, history, 2048)
#         gen = inference_stream_vllm(llm_engine, template,
#                                     [{
#                                         'query': query,
#                                         'history': history
#                                     }])
         
#         for resp_list in gen:
#             history = resp_list[0]['history']
#             total_history = old_history + history
           
#             yield total_history


#     model_name = model_type
#     with gr.Blocks() as demo:
#         gr.Markdown(f'<center><font size=8>{model_name} Bot</center>')
#         with gr.Row():
#             with gr.Column():
#                 gr.Image('role.png', width=1669, height=260)
#         with gr.Row():
#             with gr.Column(scale=6):
#                 chatbot = gr.Chatbot(label=f'{model_name}')
#                 message = gr.Textbox(lines=2, label='Input')
#                 with gr.Row():
#                     clear_history = gr.Button('🧹 清除历史对话')
#                     send = gr.Button('🚀 发送')
#             with gr.Column(scale=4):
#                 with gr.Row():
#                     actual_buttons = gr.Button("通用：事实性")
#                     safty_buttons = gr.Button("通用：安全性")
#                     power_adjust_buttons = gr.Button("调度：出力调整")
#                 with gr.Row():
#                     voltage_buttons = gr.Button("监控操作：电压越界")
#                     restored_buttons = gr.Button("黑启动：恢复顺序")
    
                
#         send.click(
#             fn=model_chat, inputs=[message, chatbot], outputs=[message, chatbot])
#         clear_history.click(
#             fn=clear_session, inputs=[], outputs=[chatbot], queue=False)
        
#         actual_buttons.click(fn=Factualness, inputs=[chatbot], outputs=[chatbot])
#         safty_buttons.click(fn=safty, inputs=[chatbot], outputs=[chatbot])
#         power_adjust_buttons.click(fn=power_output_adjustment, inputs=[chatbot], outputs=[chatbot])
#         voltage_buttons.click(fn=voltage_excursion, inputs=[chatbot], outputs=[chatbot])
#         restored_buttons.click(fn=restore_order, inputs=[chatbot], outputs=[chatbot])

#     demo.queue().launch(height=1000, share=True, server_port=8080, server_name='0.0.0.0')
#     return demo

# demo = gradio_chat_demo()

def gradio_chat_demo():
    def model_chat(query: str, history: History) -> Tuple[str, History]:
        old_history, history = limit_history_length(template, query, history, 2048)
        gen = inference_stream_vllm(llm_engine, template,
                                    [{
                                        'query': query,
                                        'history': history
                                    }])
        for resp_list in gen:
            history = resp_list[0]['history']
            total_history = old_history + history
            yield '', total_history

    model_name = model_type
    with gr.Blocks() as demo:
        gr.Markdown(f'<center><font size=8>{model_name} Bot</center>')

        chatbot = gr.Chatbot(label=f'{model_name}')
        message = gr.Textbox(lines=2, label='Input')
        with gr.Row():
            clear_history = gr.Button('🧹 清除历史对话')
            send = gr.Button('🚀 发送')
        send.click(
            model_chat, inputs=[message, chatbot], outputs=[message, chatbot])
        clear_history.click(
            fn=clear_session, inputs=[], outputs=[chatbot], queue=False)
    demo.queue().launch(root_path="/7b-app", height=1000, share=False, server_port=8080, server_name='0.0.0.0')
    return demo

demo = gradio_chat_demo()
