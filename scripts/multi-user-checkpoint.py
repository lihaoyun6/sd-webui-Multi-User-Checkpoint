import gradio as gr
from modules import scripts

from modules.ui import refresh_symbol
from modules.ui_components import ToolButton
from modules.sd_models import select_checkpoint, load_model
from modules.shared_items import sd_vae_items, refresh_vae_list
from modules.shared import opts, list_checkpoint_tiles, refresh_checkpoints

class MultiUserCKPT(scripts.Script):
    def title(self):
        return 'Multi User Checkpoint'
    
    def describe(self):
        return "Allow multiple clients to create task queues with different checkpoints."
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
            
    def ui(self, is_img2img):
        return [self.checkpoint, self.vae]
    
    def refresh_ckpts(self):
        refresh_checkpoints()
        return {"choices": ["Do not change"]+list_checkpoint_tiles(), "__type__": "update"}
    
    def refresh_vaes(self):
        refresh_vae_list()
        return {"choices": ["Do not change"]+sd_vae_items(), "__type__": "update"}
    
    def before_component(self, component, **kwargs):
        if kwargs.get("label") == f"Sampling method":
            self.checkpoint = gr.Dropdown(
                elem_id="muc_checkpoint",
                label="SD Checkpoint for you",
                value="Do not change",
                choices=["Do not change"]+list_checkpoint_tiles(),
                interactive=True
            )
            self.refresh_ckpt = ToolButton(value=refresh_symbol, elem_id="muc_refresh_ckpt")
            self.refresh_ckpt.click(
                fn=self.refresh_ckpts,
                inputs=[],
                outputs=[self.checkpoint]
            )
            
            self.vae = gr.Dropdown(
                elem_id="muc_vae",
                label="SD VAE for you",
                value="Do not change",
                choices=["Do not change"]+sd_vae_items(),
                interactive=True
            )
            self.refresh_vae = ToolButton(value=refresh_symbol, elem_id="muc_refresh_vae")
            self.refresh_vae.click(
                fn=self.refresh_vaes,
                inputs=[],
                outputs=[self.vae]
            )
    
    def process(self, p, ckpt:str, vae:str):
        if vae != opts.sd_vae and vae != "Do not change":
            opts.sd_vae = vae
        if ckpt != opts.sd_model_checkpoint and ckpt != "Do not change":
            opts.sd_model_checkpoint = ckpt
            load_model(checkpoint_info=select_checkpoint())

            