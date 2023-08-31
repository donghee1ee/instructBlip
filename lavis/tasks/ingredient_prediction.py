import json
import os

from lavis.common.dist_utils import main_process
from lavis.common.logger import MetricLogger
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.datasets.data_utils import prepare_sample
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized

import numpy as np
import torch.distributed as dist
import torch

import ast
import logging

@registry.register_task("ingredient_prediction")
class IngredientPredictionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric
    
    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset
        
        self.ingr_vocab_size = datasets[name]['val'].get_ingrs_vocab_size()
        self.ingrs_vocab = datasets[name]['val'].ingrs_vocab ## ingrs_vocab.idx2word , word2idx?
        self.max_num_labels = datasets[name]['val'].max_num_labels

        return datasets
    
    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )
    
    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        recipe_ids = samples['recipe_id']
        ingr_gts = samples['text_input']

        for ingr_gt, caption, recipe_id in zip(ingr_gts, captions, recipe_ids):
            results.append({'ingredients_gt': ingr_gt, 'ingredients_pred': caption, 'recipe_id': recipe_id})
        
        return results

        # img_ids = samples["image_id"]
        # for caption, img_id in zip(captions, img_ids):
        #     results.append({"caption": caption, "image_id": int(img_id)})

        # return results
    
    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 50 ##

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    # ## from retreival
    # def evaluation(self, model, data_loader, **kwargs):
    #     # score_i2t, score_t2i = model.compute_sim_matrix(model, data_loader)
    #     score_i2t, score_t2i = model.compute_sim_matrix(data_loader, task_cfg=self.cfg)

    #     if is_main_process():
    #         eval_result = self._report_metrics(
    #             score_i2t,
    #             score_t2i,
    #             data_loader.dataset.txt2img,
    #             data_loader.dataset.img2txt,
    #         )
    #         logging.info(eval_result)
    #     else:
    #         eval_result = None

    #     return eval_result
    
    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="recipe_id",
        )
        

        if self.report_metric:
            eval_metrics, error_types = self._report_metrics(
                val_result = val_result
            )
        else:
            eval_metrics = {"agg_metrics": 0.0}
        

        if is_dist_avail_and_initialized(): ## ??
            dist.barrier()
        

        ## TODO eval_metrics, error_types 합치기 - 그냥 네개 평균?

        if is_main_process(): ## TODO error_types concatenation
            gathered_eval = {}
            for k, v in eval_metrics.items():
                gathered_values = [torch.zeros_like(v) for _ in range(get_world_size())]
                dist.all_gather(tensor_list = gathered_values, tensor = v)
                gathered_eval[k] = torch.stack(gathered_values)
                # gathered_eval[k] = gathered_values
            
            eval_metrics = gathered_eval
            log_stats = {'eval_metrics': {k: v for k, v in eval_metrics.items()}}

            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
            
            logging.info(log_stats) ##
            print("gathered eval metrics stored in ", os.path.join(registry.get_path("output_dir"), "evaluate.txt"))

        return eval_metrics
    
    # @main_process
    def _report_metrics(self, val_result):

        metric_dict = {'iou': [], 'f1': []}
        error_types = {'tp_i': 0, 'fp_i': 0, 'fn_i': 0, 'tn_i': 0,
                           'tp_all': 0, 'fp_all': 0, 'fn_all': 0}

        ##
        ## TODO batch..어떻게 합치는지... distributed에서도 어떻게..
        
        ingr_pred_ids = []
        ingr_gt_ids = []

        for entry in val_result:
            ingr_pred = entry['ingredients_pred'].split('with ')[-1].replace(', ', ',').replace(' ', '_')
            ingr_pred = ingr_pred.split(',')
            if len(ingr_pred[-1]) == 0:
                ingr_pred = ingr_pred[:-1]
            # ingr_preds.append(ingr_pred)
            # ingr_gts.append(ingr_gt)
            ingr_pred = list(set(ingr_pred))

            ingr_pred_id = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>')
            pos = 0

            for i in range(self.max_num_labels):
                if i >= len(ingr_pred):
                    label = '<pad>'
                else:
                    label = ingr_pred[i]
                
                try:
                    label_recipe_idx = self.ingrs_vocab.word2idx[label]
                except:
                    print("not detected ingr")
                    continue

                if label_recipe_idx not in ingr_pred_id:
                    ingr_pred_id[pos] = label_recipe_idx
                    pos += 1

            ingr_pred_id[pos] = self.ingrs_vocab('<end>')
            ingr_pred_id = torch.from_numpy(ingr_pred_id).long()
            ingr_pred_ids.append(ingr_pred_id)

            # for ingr in ingr_pred:
            #     try:
            #         ingr_pred_id.append(self.ingrs_vocab(ingr))
            #     except:
            #         continue
            # ingr_pred_ids.append(ingr_pred_id)
            ingr_gt = entry['ingredients_gt'].replace(', ', ',').replace(' ', '_')
            ingr_gt = ingr_gt.split(',')

            ingr_gt_id = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>')
            pos = 0

            for i in range(self.max_num_labels):
                if i >= len(ingr_gt):
                    label = '<pad>'
                else:
                    label = ingr_gt[i]
                
                label_recipe_idx = self.ingrs_vocab(label)

                if label_recipe_idx not in ingr_gt_id:
                    ingr_gt_id[pos] = label_recipe_idx
                    pos += 1

            ingr_gt_id[pos] = self.ingrs_vocab('<end>')
            ingr_gt_id = torch.from_numpy(ingr_gt_id).long()
            ingr_gt_ids.append(ingr_gt_id)

        device_num = torch.cuda.current_device()##
        device = 'cuda:'+str(device_num)

        ingr_gt_tensor = torch.stack(ingr_gt_ids).to(device)
        ingr_pred_tensor = torch.stack(ingr_pred_ids).to(device)

        mask = self.mask_from_eos(ingr_pred_tensor, eos_value=0, mult_before=False)
        ingr_pred_tensor[mask == 0] = self.ingr_vocab_size-1
        pred_one_hot = self.label2onehot(ingr_pred_tensor, self.ingr_vocab_size-1)
        target_one_hot = self.label2onehot(ingr_gt_tensor, self.ingr_vocab_size-1)
        iou = self.softIoU(pred_one_hot, target_one_hot)
        iou = iou.sum() / (torch.nonzero(iou.data).size(0) + 1e-6)
        metric_dict['iou'] = torch.tensor([iou.item()])

        self.update_error_types(error_types, pred_one_hot, target_one_hot)

        ret_metrics = {'accuracy': [], 'f1': [], 'jaccard': [], 'f1_ingredients': [], 'dice': []}
        self.compute_metrics(ret_metrics, error_types,
                        ['accuracy', 'f1', 'jaccard', 'f1_ingredients', 'dice'], eps=1e-10,
                        weights=None)
        metric_dict['f1'] = torch.tensor(ret_metrics['f1']).to(device)

        return metric_dict, error_types

        # TODO better way to define this
        # coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
        # coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)

        # agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        # log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        # with open(
        #     os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        # ) as f:
        #     f.write(json.dumps(log_stats) + "\n")

        # coco_res = {k: v for k, v in coco_val.eval.items()}
        # coco_res["agg_metrics"] = agg_metrics

        # return coco_res
    
    def mask_from_eos(self, ids, eos_value, mult_before=True):
        device_num = torch.cuda.current_device()##
        device = 'cuda:'+str(device_num)

        # ids = ids.to(device)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mask = torch.ones(ids.size()).to(device).byte()
        mask_aux = torch.ones(ids.size(0)).to(device).byte()

        # find eos in ingredient prediction
        for idx in range(ids.size(1)):
            # force mask to have 1s in the first position to avoid division by 0 when predictions start with eos
            if idx == 0:
                continue
            if mult_before:
                mask[:, idx] = mask[:, idx] * mask_aux
                mask_aux = mask_aux * (ids[:, idx] != eos_value)
            else:
                mask_aux = mask_aux * (ids[:, idx] != eos_value)
                mask[:, idx] = mask[:, idx] * mask_aux
        return mask

    def label2onehot(self, labels, pad_value):
        device_num = torch.cuda.current_device()##
        device = 'cuda:'+str(device_num)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # input labels to one hot vector
        inp_ = torch.unsqueeze(labels, 2)
        one_hot = torch.FloatTensor(labels.size(0), labels.size(1), pad_value + 1).zero_().to(device)
        one_hot.scatter_(2, inp_, 1)
        one_hot, _ = one_hot.max(dim=1)
        # remove pad position
        one_hot = one_hot[:, :-1]
        # eos position is always 0
        one_hot[:, 0] = 0

        return one_hot
    
    def softIoU(self, out, target, e=1e-6, sum_axis=1):

        num = (out*target).sum(sum_axis, True)
        den = (out+target-out*target).sum(sum_axis, True) + e
        iou = num / den

        return iou
    
    def update_error_types(self, error_types, y_pred, y_true):

        error_types['tp_i'] += (y_pred * y_true).sum(0).cpu().data.numpy()
        error_types['fp_i'] += (y_pred * (1-y_true)).sum(0).cpu().data.numpy()
        error_types['fn_i'] += ((1-y_pred) * y_true).sum(0).cpu().data.numpy()
        error_types['tn_i'] += ((1-y_pred) * (1-y_true)).sum(0).cpu().data.numpy()

        error_types['tp_all'] += (y_pred * y_true).sum().item()
        error_types['fp_all'] += (y_pred * (1-y_true)).sum().item()
        error_types['fn_all'] += ((1-y_pred) * y_true).sum().item()

    def compute_metrics(self, ret_metrics, error_types, metric_names, eps=1e-10, weights=None):

        if 'accuracy' in metric_names:
            ret_metrics['accuracy'].append(np.mean((error_types['tp_i'] + error_types['tn_i']) / (error_types['tp_i'] + error_types['fp_i'] + error_types['fn_i'] + error_types['tn_i'])))
        if 'jaccard' in metric_names:
            ret_metrics['jaccard'].append(error_types['tp_all'] / (error_types['tp_all'] + error_types['fp_all'] + error_types['fn_all'] + eps))
        if 'dice' in metric_names:
            ret_metrics['dice'].append(2*error_types['tp_all'] / (2*(error_types['tp_all'] + error_types['fp_all'] + error_types['fn_all']) + eps))
        if 'f1' in metric_names:
            pre = error_types['tp_i'] / (error_types['tp_i'] + error_types['fp_i'] + eps)
            rec = error_types['tp_i'] / (error_types['tp_i'] + error_types['fn_i'] + eps)
            f1_perclass = 2*(pre * rec) / (pre + rec + eps)
            if 'f1_ingredients' not in ret_metrics.keys():
                ret_metrics['f1_ingredients'] = [np.average(f1_perclass, weights=weights)]
            else:
                ret_metrics['f1_ingredients'].append(np.average(f1_perclass, weights=weights))

            pre = error_types['tp_all'] / (error_types['tp_all'] + error_types['fp_all'] + eps)
            rec = error_types['tp_all'] / (error_types['tp_all'] + error_types['fn_all'] + eps)
            f1 = 2*(pre * rec) / (pre + rec + eps)
            ret_metrics['f1'].append(f1) 