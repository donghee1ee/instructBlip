import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
import numpy as np
import os
from sklearn.metrics import f1_score
import json
import pickle
from keras.utils import to_categorical
from lavis.datasets.data_utils import Vocabulary
from tqdm import tqdm
from data_loader import get_loader, collate_fn

from lavis.common.utils import now
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.config import Config
import argparse
import torch.distributed as dist

CUDA_LAUNCH_BLOCKING=1

def get_f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # get the set of unique labels that appear in both lists
    labels = set(y_true) | set(y_pred)

    # calculate the true positives, false positives, and false negatives for each label
    tp = np.array([np.sum((y_true == label) & (y_pred == label)) for label in labels])
    fp = np.array([np.sum((y_true != label) & (y_pred == label)) for label in labels])
    fn = np.array([np.sum((y_true == label) & (y_pred != label)) for label in labels])

    # calculate the precision, recall, and F1 score for each label
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def get_iou(y_true, y_pred):
    gt_set = set(y_true)
    pred_set = set(y_pred)

    # find the intersection and union of the two sets
    intersection = gt_set.intersection(pred_set)
    union = gt_set.union(pred_set)

    # calculate the IoU or Jaccard score
    iou = len(intersection) / (len(union)+1e-06)

    return iou

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def snapme():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    with open('/nfs_share2/code/donghee/inversecooking/snapme/snapme.json', 'r') as f:
        im2ingr = json.load(f)

    with open('/nfs_share2/code/donghee/inversecooking/snapme/metrics.json', 'r') as f:
        metrics = json.load(f)

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)

    vis_processors.keys()

    ## finetuned: /nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230516075/checkpoint_9.pth
    ## from scratch: /nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230515184/checkpoint_9.pth

    # model.load_checkpoint('/nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230516075/checkpoint_9.pth')

    model.eval()
    print("Load Done")

    images = []
    img_ids = []
    for dirpath, dirnames, filenames in os.walk('/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb'):
        for f in filenames:
            try:
                img_path = os.path.join(dirpath, f)
                img = Image.open(img_path).convert('RGB')
                img = vis_processors["eval"](img).to(device)
                images.append(img)
                img_ids.append(f)
            except:
                print("load error")
                continue
        
    images = torch.stack(images).type(torch.cuda.HalfTensor)
    print(images.shape)

    samples = dict()
    samples['image'] = images
    # captions = []

    # for image in tqdm(images):
    #     caption = model.generate({'image': image})
    #     captions.append(caption[0])

    # model.generate({"image": images[:16], "prompt": "Question: What are the ingredients I need to make this food? Answer:"})


    batch_size = 24
    captions = []
    for i in tqdm(range(0, len(images), batch_size)):
        stop = min(i+batch_size, len(images))
        samples['image'] = images[i:stop]
        samples['prompt'] = 'Question: What are the ingredients I need to make this food? Answer:'
        caption = model.generate(samples, use_nucleus_sampling = False, num_beams=5, max_length=20, min_length=1)
        captions.extend(caption)

    assert len(captions) == len(images)
    print("caption done")
    ingrs_vocab_v = pickle.load(open('/nfs_share2/code/donghee/inversecooking/data/recipe1m_vocab_ingrs.pkl', 'rb'))
    ingr2idx = ingrs_vocab_v.word2idx

    # name = 'BLIP2_finetuned'
    # name = 'BLIP2_fromscratch'
    name = 'BLIP2_zeroshot'

    ingr_vocab_size = 1488

    metrics[name] = dict()
    # metrics[name]['f1'] = np.array([])
    # metrics[name]['iou'] = np.array([])

    pred_ids = []
    for i in range(len(captions)):
        # caption = captions[i].split('with ')[-1].replace(', ', ',').replace(' ', '_')
        # caption = caption.split(',')
        # if len(caption[-1]) == 0:
        #     caption = caption[:-1]
        # caption = list(set(caption))

        img_id = img_ids[i]
        im2ingr[img_id][name] = dict()
        im2ingr[img_id][name]['prediction'] = captions[i] ##

        # caption_id = []
        # for c in caption:
        #     try:
        #         caption_id.append(ingr2idx[c])
        #     except:
        #         print("no matching ingredient")
        #         continue
        # caption_id = list(set(caption_id))

        # gt = im2ingr[img_id]['GT_inverse'] 
        # y_true = to_categorical(gt, ingr_vocab_size-1)
        # y_pred = to_categorical(caption_id, ingr_vocab_size-1)
        # y_true = np.sum(y_true, axis=0)
        # y_pred = np.sum(y_pred, axis=0)

        # f1 = f1_score(y_true, y_pred)
        # # im2ingr[img_id][name]['f1'] = f1

        # iou = get_iou(gt, caption_id)
        # im2ingr[img_id][name]['iou'] = iou

        # metrics[name]['f1'] = np.append(metrics[name]['f1'], f1)
        # metrics[name]['iou'] =np.append(metrics[name]['iou'], iou)
        
    # metrics[name]['f1'] = metrics[name]['f1'].mean()
    # metrics[name]['iou'] = metrics[name]['iou'].mean()
        
    print(metrics)

    # with open('/nfs_share2/code/donghee/inversecooking/snapme/metrics.json', 'w') as f:
    #     json.dump(metrics, f, indent=4)

    with open('/nfs_share2/code/donghee/inversecooking/snapme/snapme.json', 'w') as f:
        json.dump(im2ingr, f, indent=4)

    print("DONE")

def eval():
    original_images = []
    images = []
    paths = ['/nfs_share2/code/donghee/inversecooking/data/images/test/9/1/2/6/9126297c7d.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/9/1/2/7/91273289fb.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/9/1/3/8/913851905f.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/9/1/5/1/915114360b.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/2/0002839c83.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/3/0003967721.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/4/0004a1d74e.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/4/0004d32dec.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/4/00049b8b85.jpg']
    ids = []
    for p in paths:
        image = Image.open(p).convert('RGB')
        images.append(image)
        original_images.append(image)
        ids.append(p.split('/')[-1])

    print(len(images))

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
    )

def is_main_process():
    return get_rank() == 0

def recipe1M_test():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # a = torch.tensor([3.5]).to(device)
    
    # torch.distributed.barrier()
    # if is_main_process():
    #     dist.all_reduce(a, op = dist.ReduceOp.SUM)
    #     mean_a = a / dist.get_world_size()

    # torch.distributed.barrier()

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)

    vis_processors.keys()

    paths = {
        'finetuned_10': '/nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230516075/checkpoint_9.pth',
        'from_scratch_10': '/nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230515184/checkpoint_9.pth',
        'from_scratch_30': '/nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230519073/checkpoint_9.pth',
        'finetuned_30': '/nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230520080/checkpoint_19.pth',
        'finetuned_30_40': '/nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230521074/checkpoint_19.pth',
        'ingr_only': '/nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230525191/checkpoint_11.pth'
    }

    version = 'ingr_only' ##

    model.load_checkpoint(paths[version])
    model.eval()

    print("Load Done. version: ", version)

    # with open("/nfs_share2/code/donghee/LAVIS/recipe1M_result.json", 'r') as f:
    #     total_metric = json.load(f)
    total_metric = []

    ingrs_vocab_v = pickle.load(open('/nfs_share2/code/donghee/inversecooking/data/recipe1m_vocab_ingrs.pkl', 'rb'))
    ingr2idx = ingrs_vocab_v.word2idx

    data_loaders = {}
    datasets = {}

    aux_data_dir = '/nfs_share2/code/donghee/inversecooking/data'
    data_dir = '/nfs_share2/code/donghee/inversecooking/data'
    maxseqlen = 15
    maxnuminstrs = 10
    maxnumlabels=20
    maxnumims = 5
    transform = vis_processors['eval']
    batch_size = 32 ##
    num_workers = 4
    max_num_samples = -1

    for split in ['test']:

        _, datasets[split] = get_loader(data_dir, aux_data_dir, split,
                                                            maxseqlen,
                                                            maxnuminstrs,
                                                            maxnumlabels,
                                                            maxnumims,
                                                            transform, batch_size,
                                                            shuffle=split == 'train', num_workers=num_workers,
                                                            drop_last=True,
                                                            max_num_samples=max_num_samples,
                                                            use_lmdb=True,
                                                            suff='')
        
        sampler = torch.utils.data.distributed.DistributedSampler(datasets[split])

        data_loaders[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size = batch_size,
            shuffle = False,
            sampler = sampler,
            num_workers = num_workers,
            collate_fn = collate_fn,
            drop_last = True
        )


    split = 'test'
    ingr_vocab_size = datasets[split].get_ingrs_vocab_size()
    instrs_vocab_size = datasets[split].get_instrs_vocab_size()
    # ingr_vocab_size = 1488
    
    f1s = []
    ious = []
    pred_miss = 0
    for loader in tqdm(data_loaders[split]):
        img_inputs, captions, ingr_gt, recipe_ids, img_id = loader
        img_inputs = img_inputs.type(torch.cuda.HalfTensor)
        prediction = model.generate({'image': img_inputs})
        
        for i in range(len(prediction)):
            pred = prediction[i].split('made with ')[-1].replace(', ', ',').replace(' ','_')
            pred = pred.split(',')
            if len(pred[-1]) == 0:
                pred = pred[:-1]
            pred = list(set(pred))

            pred_id = []
            for ingr in pred:
                try:
                    pred_id.append(ingr2idx[ingr])
                except:
                    # print("no matching ingredient")
                    continue
            pred_id = list(set(pred_id))

            gt = ingr_gt[i]
            gt = gt.cpu().numpy()

            gt_id = []
            for id in gt:
                if id == 0:
                    break
                gt_id.append(id)
            gt = np.array(gt_id)

            if len(pred_id) == 0:
                print("No prediction")
                pred_miss += 1
                continue

            y_true = to_categorical(gt, ingr_vocab_size-1)
            y_pred = to_categorical(pred_id, ingr_vocab_size-1)
            y_true = np.sum(y_true, axis=0)
            y_pred = np.sum(y_pred, axis=0)

            f1 = f1_score(y_true, y_pred)
            f1s.append(f1)
            # im2ingr[img_id][name]['f1'] = f1

            iou = get_iou(gt, pred_id)
            ious.append(iou)
    
    torch.distributed.barrier()


    mean_iou = sum(ious) / len(ious)
    mean_f1 = sum(f1s) / len(f1s)

    iou_tensor = torch.tensor([mean_iou], dtype = torch.float).to(device)
    f1_tensor = torch.tensor([mean_f1], dtype = torch.float).to(device)
    pred_miss_tensor = torch.tensor([pred_miss], dtype = torch.float).to(device)

    print(f'===== {version} =====')
    print("mean_iou: ", mean_iou)
    print("mean_f1: ", mean_f1)
    print("# miss: ", pred_miss)

    torch.distributed.barrier()

    # if is_main_process():
    #     dist.all_reduce(iou_tensor)
    #     dist.all_reduce(f1_tensor)
    #     dist.all_reduce(pred_miss_tensor)

    #     mean_iou = iou_tensor / 4
    #     mean_f1 = f1_tensor / 4
    #     pred_miss = pred_miss_tensor.item()

    #     metrics = dict()
    #     metrics[version] = {
    #         'iou': mean_iou.item(),
    #         'f1': mean_f1.item(),
    #         'no_prediction': pred_miss.item()
    #     }
    #     total_metric.append(metrics)
    #     print("** mean_iou: ", mean_iou)
    #     print("** mean_f1: ", mean_f1)
    #     print("** # miss: ", pred_miss)

    #     with open("/nfs_share2/code/donghee/LAVIS/recipe1M_result.json", 'w') as f:
    #         json.dump(total_metric, f, indent=4)
    
    # torch.distributed.barrier()

    print("==== Done ====")

if __name__ == '__main__':
    recipe1M_test()


