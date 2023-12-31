"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
import torch
import numpy as np

import lavis.common.dist_utils as dist_utils
import torch.distributed as dist
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
from lavis.tasks.base_task import BaseTask
from lavis.datasets.data_utils import prepare_sample
from lavis.common.dist_utils import is_dist_avail_and_initialized, get_rank, get_world_size, is_main_process
from lavis.common.logger import MetricLogger, SmoothedValue


@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file

                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})

        return pred_qa_pairs

    def evaluation(self, model, data_loader, cuda_enabled=True): ## TODO
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        f1 = []
        iou = []
        loss = []

        global_TPs = 0
        global_FPs = 0
        global_FNs = 0

        # snapme_paths = ['/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/00aed7f846529795993f19942c0d.jpeg', '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/0b90e149404986da7d5930438421.jpeg', '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/0b4297584ab4a5813c4664fde485.jpeg', '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/0d96e4264601b476277c7ee51232.jpeg', '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/0d0258474903a20af0b59f165e3d.jpeg']

        # snapmes = [p.split('/')[-1] for p in snapme_paths]
        # snapme_ids = set(snapmes)
        model.eval()
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled) ## (batch_size, --)

            # for i, img_id in enumerate(samples['img_id']):
            #     if img_id in snapme_ids:
            #         logging.info(f'{i}, {img_id}')
            #         with torch.no_grad():
            #             with model.maybe_autocast():
            #                 loss_dict = model.forward(samples, eval = True)
            #                 logging.info(loss_dict['prediction'])

            with torch.no_grad():
                with model.maybe_autocast():
                    # loss_dict = model.eval_metrics(samples) ## eval = True -> loss_dict['prediction']
                    loss_dict = model.generate(samples, eval=True, num_beams=1) ## greedy decoding # eval
                    
                f1.append(loss_dict['f1'].item())
                iou.append(loss_dict['iou'].item())
                loss.append(loss_dict['loss'].item())

                # f1.extend(loss_dict['f1'])
                # iou.extend(loss_dict['iou'])
                # loss.append(loss_dict['loss'].item())

                # global_TPs += loss_dict['TPs']
                # global_FPs += loss_dict['FPs']
                # global_FNs += loss_dict['FNs']
            
            sen = []
            for k, v in loss_dict.items():
                sen.append(f'{k}: {v}')
            sen = ", ".join(sen)
            # logging.info(sen)

            # if is_dist_avail_and_initialized():
            #     dist.barrier()

            avg_f1 = np.mean(np.array(f1))
            avg_iou = np.mean(np.array(iou))
            avg_loss = np.mean(np.array(loss))
            
            
        # avg_f1 = sum(f1) / len(f1)
        # avg_iou = sum(iou) / len(iou)
        # avg_loss = sum(loss) / len(loss)

        # avg_f1 = np.mean(np.array(f1))
        # avg_iou = np.mean(np.array(iou))
        # avg_loss = np.mean(np.array(loss))

        # with open('/nfs_share2/code/donghee/LAVIS/f1_batch1.json', 'w') as f: ##
        #     json.dump(f1, f)

        # Calculate global metrics
        # precision = global_TPs / (global_TPs + global_FPs + 1e-10)
        # recall = global_TPs / (global_TPs + global_FNs + 1e-10)

        # avg_f1 = 2 * precision * recall / (precision + recall + 1e-10)
        # avg_iou = global_TPs / (global_TPs + global_FPs + global_FNs + 1e-10)

        # logging.info(f'** average F1: {avg_f1}, average iou: {avg_iou}, average loss: {avg_loss}')
        logging.info(f'** average F1: {avg_f1}, average iou: {avg_iou}')
        
        return {'f1': avg_f1, 'iou': avg_iou, "agg_metrics": avg_f1}

        ## model.forward, loss 계산, f1, iou 계산..

    def after_evaluation(self, val_result, split_name, **kwargs):
        final_result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            ## 원래 remove_duplicate="question_id", 였음
        )

        # metrics = self._report_metrics(result_file=result_file, split=split_name)

        # return metrics

        if is_dist_avail_and_initialized():
            dist.barrier()

        with open(final_result_file, 'r') as f:
            result_metrics = json.load(f)

        result_metrics = result_metrics[-1]

        return result_metrics 

    @staticmethod
    def save_result(result, result_dir, filename):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        # json.dump([result], open(result_file, "a")) ## no [], 'w'

        # Check if the result_file already exists
        if os.path.exists(result_file):
            # If it exists, load the current list of dictionaries
            with open(result_file, 'r') as f:
                current_data = json.load(f)
        else:
            # If it doesn't exist, initialize an empty list
            current_data = []

        # Append the new result dictionary to the list
        current_data.append(result)

        # Write the updated list back to result_file
        with open(result_file, 'w') as f:
            json.dump(current_data, f, indent=4)
        

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = {'f1': 0.0, 'iou': 0.0, 'agg_metrics': 0.0}

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                res = res[-1]

                for k, v in res.items():
                    result[k] += v

            # average
            for k, v in result.items():
                result[k] = v/get_world_size()

            ## save final result file
            if os.path.exists(final_result_file):
                # If it exists, load the current list of dictionaries
                with open(final_result_file, 'r') as f:
                    current_data = json.load(f)
            else:
                # If it doesn't exist, initialize an empty list
                current_data = []

            # Append the new result dictionary to the list
            current_data.append(result)

            # Write the updated list back to result_file
            with open(final_result_file, 'w') as f:
                json.dump(current_data, f, indent=4)

            # json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
    
    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(
                resFile=result_file, quesFile=self.ques_files[split]
            )

            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info(
                    "%s : %.02f"
                    % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
                )
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")

        return metrics

@registry.register_task("gqa")
class GQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        
        for answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

        return pred_qa_pairs
        
    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        TODO: add other evaluation metrics for GQA
        """

        results = json.load(open(result_file, "r"))
        acc = []
        vqa_tool = VQAEval()

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            gt_ans = res["gt_ans"]
            pred = res["pred_ans"]

            if self.inference_method == "generate":
                pred = vqa_tool.processPunctuation(pred)
                pred = vqa_tool.processDigitArticle(pred)

            vqa_acc = 1 if pred == gt_ans else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics
        

@registry.register_task("aok_vqa")
class AOKVQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
        )

        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["direct_answers"]

        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            pred_qa_pairs.append(
                {"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer}
            )

        return pred_qa_pairs

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Implementing accuracy computation for AOKVQA, see
        https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py#L45 for details.
        """
        # TODO add evaluation for multi-choice

        results = json.load(open(result_file, "r"))
        acc = []

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            pred = res["pred_ans"]
            gt_ans = res["gt_ans"]

            num_match = sum([pred == gt for gt in gt_ans])
            vqa_acc = min(1.0, num_match / 3.0)

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics

    @dist_utils.main_process
    def _save_result_leaderboard(self, results):
        """
        Saving the results in the format required for leaderboard evaluation.

        [TODO] add support for multi-choice.
        """
        result_leaderboard = dict()
        for res in results:
            result_leaderboard[res["question_id"]] = {
                "direct_answer": res["pred_ans"],
                "multiple_choice": "",
            }

        result_file = registry.get_path("result_dir") + "_leaderboard.json"

        with open(result_file, "w") as f:
            json.dump(result_leaderboard, f)

        logging.info(f"Saved results for leaderboard evaluation at {result_file}")
