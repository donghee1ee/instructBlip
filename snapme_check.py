from lavis.datasets.data_utils import compute_metrics
import torch
from PIL import Image
from matplotlib import pyplot as plt
from lavis.models import load_model_and_preprocess
import os
import json
from tqdm import tqdm
import numpy as np


def one_hot_encode(arr, num_category=1486):
        # unique_elements = list(set(arr))
        one_hot = np.array([0] * num_category)

        for elem in arr:
            temp = np.array([0] * num_category)
            temp[int(elem)] = 1
            one_hot += temp

        return one_hot

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    model, vis_processors,_ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type = 'vicuna7b', is_eval=True, device = device) ## blip2_t5_instruct/ flant5xxl
    model.load_checkpoint("/nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230828023/checkpoint_22.pth")
    model.eval()

    ## snapme
    with open('/nfs_share2/code/donghee/LAVIS/snapme/snapme_final.json', 'r') as f:
        im2ingr = json.load(f)

    import numpy as np

    # original_images = []
    snapme_images = []
    # snapme_paths = ['/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/00aed7f846529795993f19942c0d.jpeg', '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/0b90e149404986da7d5930438421.jpeg', '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/0b4297584ab4a5813c4664fde485.jpeg', '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/0d96e4264601b476277c7ee51232.jpeg', '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/0d0258474903a20af0b59f165e3d.jpeg']
    snapme_ids = []
    original_snapme_images = []

    base_path = '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb'

    for entry in im2ingr:
        p = os.path.join(base_path, entry['id'])
        image = Image.open(p).convert('RGB')
        original_snapme_images.append(image)
        image = vis_processors["eval"](image).unsqueeze(0).to(device)
        snapme_images.append(image)
        snapme_ids.append(entry['id'])

    print(len(snapme_images))

    snapme_answers = np.array([]).reshape(0, 1486)
    prompt = 'Question: What are the ingredients I need to make this food? Answer:'

    ## batch 1
    for image in tqdm(snapme_images):
        # answer = model.generate({"image": image, "prompt": "Question: List the ingredients for this food in the format: {quantity} {unit} of {ingredient}. Answer:"}) 
        ## TODO - 여기서 answer가 뭘 받는 건지 디버깅
        answer = model.generate({"image": image, "prompt": prompt})  
        # snapme_answers.append(answer[0][0].cpu().numpy()) ## answer[0]=ingr_prediction (one-hot)
        snapme_answers = snapme_answers.vstack((snapme_answers, answer[0][0].cpu().numpy()))

    ## batch 10
    # step_size = 10
    # for i in range(0, len(snapme_images), step_size):
    #     answers = model.generate({'image': snapme_images[i:i+step_size], 'prompt': 'Question: What are the ingredients I need to make this food? Answer:'})
    #     for answer in answers:
    #         snapme_answers.append(answer)
        


    f1s = [],
    ious = []
    for entry, pred in zip(im2ingr, snapme_answers):
        gt_int = np.array(entry['gt_int']) - 1

        # one-hot
        gt_int = one_hot_encode(gt_int) ## np array
        # pred = one_hot_encode(pred)

        assert gt_int.shape[1] == 1486 and pred.shape[1] == 1486

        f1, iou = compute_metrics(pred, gt_int)
        print(f"f1: {f1}, iou: {iou}")
        f1s.append(f1)
        ious.append(iou)
    
    avg_f1 = sum(f1s) / len(f1s)
    avg_iou = sum(ious) / len(ious)

    print(f"** Final F1: {avg_f1}, IoU: {avg_iou}")

    for i in range(len(snapme_images)):
        print("===== Last classifier SNAPMe ====")
        plt.imshow(original_snapme_images[i])
        plt.title(snapme_ids[i])
        plt.show()
        print("- ID: ", snapme_ids[i])
        print(f"- F1: {f1s[i]}, IoU: {ious[i]}")
        print("- Prediction: ", snapme_answers[i])
        print("- GT: ", np.array(im2ingr[i]['gt'])-1)
        print()

if __name__ == '__main__':
    main()