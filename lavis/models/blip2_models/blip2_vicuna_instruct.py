"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

import json
import transformers

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train

from lavis.datasets.data_utils import compute_metrics, compute_raw_metrics

class IngredientClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim = 1024):
        super(IngredientClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class QuantityClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim = 1024):
        super(QuantityClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@registry.register_model("blip2_vicuna_instruct")
class Blip2VicunaInstruct(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        ingr_layer = False,
        quantity_layer = False,
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"        
        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input

        ##
        
        self.num_ingr = 1488 ### 1486 

        if ingr_layer:
            self.ingr_classifier = IngredientClassifier(input_dim = self.llm_model.config.hidden_size, num_labels=self.num_ingr, hidden_dim=1024)
            self.ingr_criterion = nn.BCEWithLogitsLoss()

            self.ingr_loss_constant = 1e06 # 1000...
        self.ingr_layer = ingr_layer

        if quantity_layer:
            self.quantity_classifier = QuantityClassifier(input_dim = self.llm_model.config.hidden_size, num_labels=self.num_ingr, hidden_dim=1024)
            self.quantity_criterion = nn.MSELoss()
        self.quantity_layer = quantity_layer

        self.idx2ingr = json.load(open('/nfs_share2/code/donghee/LAVIS/ingr_cluster.json', 'r'))

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward(self, samples, eval = False):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        bs = image.size(0)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        ## tracking eos pos
        eos_pos = (llm_tokens['input_ids'] == self.llm_tokenizer.eos_token_id)
        eos_pos[:, 0] = False ## ignore SOS token (SOS token = EOS token)
        eos_pos = (eos_pos).nonzero(as_tuple=True)[1]


        if self.ingr_layer:
            with self.maybe_autocast():
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                    output_hidden_states = True,
                )

                # hidden_states = self.llm_model(
                #     inputs_embeds=inputs_embeds, ## TODO input_ids로 해야할듯.. ## 하 근데 image token 때문에 안되네..
                #     attention_mask=attention_mask,
                #     return_dict=True,
                #     labels=targets,
                #     only_hidden = True,
                # )

            hidden_states = outputs['hidden_states'][-1]
            llm_loss = outputs.loss
            # Extract the hidden states of the EOS tokens
            batch_indices = torch.arange(hidden_states.size(0)).to(hidden_states.device)
            num_query = inputs_llm.size(1) ## 32
            eos_hidden = hidden_states[batch_indices, num_query + eos_pos, :] ## (batch, 4096)
            
            # expanded_attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            # masked_hidden_states = hidden_states * expanded_attention_mask
            # sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
            # sum_attention_mask = attention_mask.sum(dim=1, keepdim=True)
            # mean_hidden_states = sum_hidden_states / sum_attention_mask
            
            # hidden_states = torch.mean(hidden_states,dim=1) ## shape (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
            ingr_prediction = self.ingr_classifier(eos_hidden)

            ingr_gt = samples['ingredient_int']
            ingr_gt_tensor = torch.zeros((hidden_states.size(0), self.num_ingr)) ## (batch_size, 1488)

            for i, labels in enumerate(ingr_gt):
                ingr_gt_tensor[i, labels] = 1
                ingr_gt_tensor[i, [0, self.num_ingr-1]] = 0
            
            ingr_gt_tensor = ingr_gt_tensor.to(ingr_prediction.device) ## 0, 1487 제거 ## 원래 ingr_gt_tensor[:, 1:-1]
            ingr_loss = self.ingr_criterion(ingr_prediction, ingr_gt_tensor)

            f1, iou = compute_metrics(torch.sigmoid(ingr_prediction), ingr_gt_tensor) ## list
            # Compute raw metrics
            # TPs, FPs, FNs = compute_raw_metrics(torch.sigmoid(ingr_prediction), ingr_gt_tensor)

            if self.quantity_layer:
                ## quantity
                quantity_prediction = self.quantity_classifier(hidden_states)

                quantity_gt = samples['weight'].to(quantity_prediction.device) ## (batch, 1486) ?? 
                # quantity_gt_tensor = torch.zeros((hidden_states.shape[0], self.num_ingr+2)) ## (batch_size, 1488)
                quantity_loss = self.quantity_criterion(quantity_prediction, quantity_gt)
                
                loss = self.ingr_loss_constant * ingr_loss + quantity_loss + llm_loss
            else:
                quantity_loss = 0
                loss = ingr_loss + llm_loss

            ##
            if eval:
                ingr_text = []
                pred = (torch.sigmoid(ingr_prediction) > 0.5).float() ##
                for i in range(pred.size(0)):
                    one_hot = pred[i]
                    indices = (one_hot ==1).nonzero().squeeze()

                    ingr_names = []

                    if indices.dim() == 0:
                        ingr_name = self.idx2ingr[str(indices.item())]
                        ingr_names.append(ingr_name)
                    
                    else:
                        for idx in indices:
                            ingr_name = self.idx2ingr[str(idx.item())]
                            ingr_names.append(ingr_name)

                    ingr_text.append(ingr_names)
                
                # return {"loss": loss, "TPs": TPs, "FPs": FPs, 'FNs': FNs, "ingr_loss": ingr_loss, "quantity_loss": quantity_loss, 'prediction': ingr_text}
                return {"loss": loss, "f1": f1, "iou": iou, "ingr_loss": ingr_loss, "quantity_loss": quantity_loss, 'llm_loss': llm_loss, 'prediction': ingr_text}
            
            else:
                # return {"loss": loss, "TPs": TPs, "FPs": FPs, 'FNs': FNs, "ingr_loss": ingr_loss, "quantity_loss": quantity_loss}
                return {"loss": loss, "f1": f1, "iou": iou, "ingr_loss": ingr_loss, "quantity_loss": quantity_loss, 'llm_loss': llm_loss}
        
        else:
            with self.maybe_autocast():
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss

            return {"loss": loss}
    
    @torch.no_grad()    
    def eval_metrics(self, samples, eval = False): ## self.ingr_num 1486으로 되어있음. 다시 해야할 것임
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) ## (batch_size, 257)

        bs = image.size(0)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) ## 그냥 복사인듯?
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device) ## (batch_size, 32)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1) ## (batch_size, 49)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            ) ## (batch_size, 49, 768)
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            ) ## (batch_size, 32, 768)

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)


        if self.ingr_layer:
            with torch.no_grad():
                with self.maybe_autocast():
                    hidden_states = self.llm_model( ## return_hidden
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=None, ## None
                        only_hidden = True,
                        output_hidden_states=True,
                    )

                expanded_attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                masked_hidden_states = hidden_states * expanded_attention_mask
                sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
                sum_attention_mask = attention_mask.sum(dim=1, keepdim=True)
                mean_hidden_states = sum_hidden_states / sum_attention_mask
                
                # hidden_states = torch.mean(hidden_states,dim=1) ## shape (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
                ingr_prediction = self.ingr_classifier(mean_hidden_states)

            ingr_gt = samples['ingredient_int']
            ingr_gt_tensor = torch.zeros((hidden_states.shape[0], self.num_ingr)) ## (batch_size, 1488)

            for i, labels in enumerate(ingr_gt):
                ingr_gt_tensor[i, labels] = 1
                ingr_gt_tensor[i, [0, self.num_ingr-1]] = 0
            
            ingr_gt_tensor = ingr_gt_tensor.to(ingr_prediction.device) ## 0, 1487 제거
            ingr_loss = self.ingr_criterion(ingr_prediction, ingr_gt_tensor)

            f1, iou = compute_metrics(torch.sigmoid(ingr_prediction), ingr_gt_tensor) ## list
            # Compute raw metrics
            # TPs, FPs, FNs = compute_raw_metrics(torch.sigmoid(ingr_prediction), ingr_gt_tensor)

            if self.quantity_layer:
                ## quantity
                quantity_prediction = self.quantity_classifier(hidden_states)

                quantity_gt = samples['weight'].to(quantity_prediction.device) ## (batch, 1486) ?? 
                # quantity_gt_tensor = torch.zeros((hidden_states.shape[0], self.num_ingr+2)) ## (batch_size, 1488)
                quantity_loss = self.quantity_criterion(quantity_prediction, quantity_gt)
                
                loss = self.ingr_loss_constant * ingr_loss + quantity_loss
            else:
                quantity_loss = 0
                loss = ingr_loss

            ##
            if eval:
                ingr_text = []
                pred = (torch.sigmoid(ingr_prediction) > 0.5).float() ##
                for i in range(pred.size(0)):
                    one_hot = pred[i]
                    indices = (one_hot ==1).nonzero().squeeze()

                    ingr_names = []

                    if indices.dim() == 0:
                        ingr_name = self.idx2ingr[str(indices.item())]
                        ingr_names.append(ingr_name)
                    
                    else:
                        for idx in indices:
                            ingr_name = self.idx2ingr[str(idx.item())]
                            ingr_names.append(ingr_name)

                    ingr_text.append(ingr_names)
                
                # return {"loss": loss, "TPs": TPs, "FPs": FPs, 'FNs': FNs, "ingr_loss": ingr_loss, "quantity_loss": quantity_loss, 'prediction': ingr_text}
                return {"loss": loss, "f1": f1, "iou": iou, "ingr_loss": ingr_loss, "quantity_loss": quantity_loss, 'prediction': ingr_text}
            
            else:
                # return {"loss": loss, "TPs": TPs, "FPs": FPs, 'FNs': FNs, "ingr_loss": ingr_loss, "quantity_loss": quantity_loss}
                return {"loss": loss, "f1": f1, "iou": iou, "ingr_loss": ingr_loss, "quantity_loss": quantity_loss}
        
        else:
            # with self.maybe_autocast():
            #     outputs = self.llm_model(
            #         inputs_embeds=inputs_embeds,
            #         attention_mask=attention_mask,
            #         return_dict=True,
            #         labels=targets,
            #     )
            # loss = outputs.loss

            # return {"loss": loss}
            return None

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
        eval = False,
    ):
        self.llm_tokenizer.padding_side = "left"

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        # For video data
        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:]) ## query_token.size(1) (seq_length)로 자르는 이유는, exclude any states that might correspond to other parts of the input like instruction toekns or padding
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

        if self.ingr_layer: ## 여기 다시 TODO
            with self.maybe_autocast():
                outputs = self.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=use_nucleus_sampling, # false
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams, ## 1, greedy decoding
                    max_length=max_length,
                    min_length=min_length,
                    # eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    # output_attentions = True, -> 이거 하면 overload 많이 됨
                ) ## token id..

            # ---- TODO greedy 말고..? - sequence 어쩌구 그 hugging face 그거 참고
            generated_sequences = outputs.sequences
            eos_pos = torch.nonzero((generated_sequences == self.llm_tokenizer.eos_token_id), as_tuple=True)[1]            

            if len(eos_pos) != bs: # if no EOS
                eos_list = []
                for sequence in generated_sequences:
                    eos_token_pos = torch.nonzero((sequence == self.llm_tokenizer.eos_token_id), as_tuple=True)[0]
                    if len(eos_token_pos) == 0:
                        eos_token_pos = len(sequence) -1
                    eos_list.append(eos_token_pos)
                eos_pos = torch.tensor(eos_list, device = eos_pos.device)
                assert len(eos_pos) == bs
            
            eos_pos = eos_pos -1 ## SOS remove ## TODO chat-gpt 는 아니라네..
            eos_hidden = [outputs.hidden_states[pos][-1][i] for i, pos in enumerate(eos_pos)]
            eos_hidden = torch.stack(eos_hidden).squeeze(1)
            # -----

            # Extract the hidden states of the EOS tokens
            # batch_indices = torch.arange(bs).to(eos_pos.device)
            # num_query = inputs_llm.size(1) ## 32
            # eos_hidden = hidden_states[batch_indices, num_query+eos_pos, :]
            # eos_hidden = hidden_states[-1]

            ingr_prediction = self.ingr_classifier(eos_hidden)

            ## 여기!
            # hidden_states = self.llm_model(
            #     inputs_embeds=inputs_embeds,
            #     attention_mask=attention_mask,
            #     only_hidden=True
            # ) ## (batch, 50, 4096)

            ## 이 아래 이걸로 해야하지 않을까 싶은데..
            # hidden_states = self.llm_model( ## return_hidden
            #         inputs_embeds=inputs_embeds,
            #         attention_mask=attention_mask,
            #         return_dict=True,
            #         labels=None, ## None
            #         only_hidden = True,
            #         output_hidden_states=True,
            #     )

            # expanded_attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            # masked_hidden_states = hidden_states * expanded_attention_mask
            # sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
            # sum_attention_mask = attention_mask.sum(dim=1, keepdim=True) + 1e-9
            # mean_hidden_states = sum_hidden_states / sum_attention_mask
        
            # ingr_prediction = self.ingr_classifier(mean_hidden_states)

            # with self.maybe_autocast():
            #     transformer_outputs = self.llm_model(
            #         inputs_embeds=inputs_embeds,
            #         attention_mask=attention_mask,
            #         return_dict=True,
            #         output_hidden_states = True,
            #     )
            #     # hidden_states = transformer_outputs[0] ## hidden state
            #     hidden_states = transformer_outputs.hidden_states[-1] ## TODO - 이거 last hidden state 맞는지
            #     logits = self.ingr_classifier(hidden_states)

            # batch_size = inputs_embeds.shape[0]

            # if self.llm_tokenizer.pad_token_id is None and batch_size != 1:
            #     raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            # if self.llm_tokenizer.pad_token_id is None:
            #     sequence_lengths = -1
            # else:
            #     if llm_tokens['input_ids'] is not None:
            #         sequence_lengths = (torch.ne(llm_tokens['input_ids'], self.llm_tokenizer.pad_token_id).sum(-1) - 1).to(logits.device)
            #     else:
            #         sequence_lengths = -1

            # pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

            loss_fct = nn.BCEWithLogitsLoss()

            ingr_gt = samples['ingredient_int']
            ingr_gt_tensor = torch.zeros((bs, self.num_ingr)) ## (batch_size, 1488)

            for i, labels in enumerate(ingr_gt):
                ingr_gt_tensor[i, labels] = 1
                ingr_gt_tensor[i, [0,self.num_ingr-1]] = 0 ## pad
    
            ingr_gt_tensor = ingr_gt_tensor.to(ingr_prediction.device) ## 원래는 [:, 1:-1]
            
            loss = loss_fct(ingr_prediction, ingr_gt_tensor)
            
            ###

            ingr_prediction = torch.sigmoid(ingr_prediction)
            ingr_prediction = (ingr_prediction > 0.5).float()
            ingr_pred_int = [torch.nonzero(row, as_tuple=False).squeeze(-1).tolist() for row in ingr_prediction]
            # ingr_pred_int = [[item + 1 for item in sublist] for sublist in ingr_pred_int]


            quantity_prediction = []
            if self.quantity_layer:
                quantity_prediction = self.quantity_classifier(hidden_states)
                quantity_prediction = [quantity_prediction[i][ingr_pred] for i, ingr_pred in enumerate(ingr_pred_int)]
                # assert quantity_prediction.shape == ingr_pred_int.shape
            
            return_metric = {'f1':None, 'iou': None, 'prediction':None, 'ingr_prediction': ingr_prediction, 'ingr_pred_int': ingr_pred_int, 'quantity_prediction':quantity_prediction, 'loss': loss}

            if eval:
                # ingr_gt = samples['ingredient_int']
                # ingr_gt_tensor = torch.zeros((hidden_states.shape[0], self.num_ingr)) ## (batch_size, 1488)

                # for i, labels in enumerate(ingr_gt):
                #     ingr_gt_tensor[i, labels] = 1
        
                # ingr_gt_tensor = ingr_gt_tensor.to(ingr_prediction.device) ## 0, 1487 제거
                f1, iou = compute_metrics(ingr_prediction, ingr_gt_tensor)

                return_metric['f1'] = f1
                return_metric['iou'] = iou

                ingr_text = []
                pred = ingr_prediction
                for i in range(pred.size(0)):
                    one_hot = pred[i]
                    indices = (one_hot ==1).nonzero().squeeze()

                    ingr_names = []

                    if indices.dim() == 0: ## TODO
                        ingr_name = self.idx2ingr.get(str(indices.item())) ## 
                        ingr_names.append(ingr_name)
                    
                    else:
                        for idx in indices:
                            if idx == 0 or idx == self.num_ingr -1:
                                continue
                            ingr_name = self.idx2ingr.get(str(idx.item()))
                            if ingr_name is not None:
                                ingr_names.append(ingr_name)

                    ingr_text.append(ingr_names)
                return_metric['prediction'] = ingr_text

            return return_metric

        else:
            with self.maybe_autocast():
                outputs = self.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    # eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )
            
            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]

            return output_text
        

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                    for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len,
        ).to(image.device)

        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=torch.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(image.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens['input_ids']
                this_llm_atts = this_llm_tokens['attention_mask']
                # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
                # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)
                inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)

                this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100)
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        ingr_layer = cfg.get("ingr_layer", False)
        quantity_layer = cfg.get("quantity_layer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            ingr_layer = ingr_layer,
            quantity_layer = quantity_layer,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)

        return model
