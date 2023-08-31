"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
import logging
import torch
from lavis.datasets.data_utils import prepare_sample
from lavis.common.logger import MetricLogger, SmoothedValue


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter= " ")
        # metric_logger.add_meter('val_loss', SmoothedValue(window_size=1, fmt="{value:.4f}"))
        # metric_logger.add_meter('val_loss_dict', SmoothedValue(window_size=1, fmt="{value:.4f}"))
        use_amp = True
        print_freq =10
        header = 'Evaluation'
        # if not hasattr(data_loader, '__next__'):
        #     data_loader = iter(data_loader)
        
        logging.info(
            "** Start evaluation **"
        )

        for i, samples in enumerate(data_loader):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            # samples.update

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    val_loss_dict = model(samples)
                    # val_loss = val_loss_dict['loss'].item()
                    if i%print_freq == 0:
                        logging.info(f" - {i}th eval loss: {val_loss_dict['loss'].item()}")
                    metric_logger.update(**val_loss_dict)
                    # metric_logger.update(val_loss)
            
        metric_logger.synchronize_between_processes()

        loss_dict_str = ''
        for k, v in metric_logger.meters.items():
            loss_dict_str += f'{k}: {v.avg} '

        logging.info(f'** Evaluation loss dict ** \n {loss_dict_str}')

        # return metric_logger.meters
