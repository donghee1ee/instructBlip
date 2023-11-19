import pandas as pd
from lavis.datasets.data_utils import Vocabulary
import pickle
import json
import ast

def snapme_process(): ## manual_description 으로 바꾸기

    df = pd.read_excel('/nfs_share/code/donghee/inversecooking/snapme/supplemental/supplemental_file_1.xlsx', sheet_name=1)
    ingrs_vocab = pickle.load(open('/nfs_share2/code/donghee/inversecooking/data/recipe1m_vocab_ingrs.pkl', 'rb'))
    ingr2idx = ingrs_vocab.word2idx

    snapme_mapping = json.load(open('snapme/snapme_mapping.json', 'r'))
    
        
    inverse_vocab = set(ingr2idx.keys())
    mapping = set(snapme_mapping.keys())
    manual = set()
    cnt = 0
    snapme = list()

    for i, ingrs in enumerate(df['manual_description']):
        try:
            result = ast.literal_eval(ingrs)
            assert type(result) == set
        except:
            result = None
            # while type(result) == set:
                # cnt += 1
            print(i)
            print(ingrs)
            correction = input(f"{ingrs} Correction: ")
            result = ast.literal_eval(correction)
            
        assert type(result) == set
        manual.update(result)

        gt_int = list()
        for ingr in result:
            if ingr in mapping:
                idx = snapme_mapping[ingr]
                if idx != None:
                    gt_int.append(idx)
                else:
                    cnt += 1 ## no matching
                    continue

            elif ingr in inverse_vocab:
                gt_int.append(ingr2idx[ingr])
            
            else:
                print("")
                idx = int(input(f"input idx for {ingr}: "))
                gt_int.append(idx)

        snapme.append({
            'id': df['filename'][i],
            'gt': list(result),
            'gt_int': gt_int
        })


    print(cnt)
    print(manual)
    print(len(manual))
    print(snapme[100])

def snapme_test():
    import torch
    from PIL import Image
    from matplotlib import pyplot as plt
    from lavis.models import load_model_and_preprocess
    import os
    import json
    import numpy as np

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model, vis_processors,_ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type = 'vicuna7b', is_eval=True, device = device) ## blip2_t5_instruct/ flant5xxl


    model.load_checkpoint("/nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230930023/checkpoint_best.pth")
    # /nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230828023/checkpoint_22.pth
    # /nfs_share2/code/donghee/LAVIS/lavis/output/BLIP2/Pretrain_stage2/20230930023/checkpoint_best.pth
    model.eval()

    model.ingr_layer = True ###
    model.quantity_layer = False

    with open('/nfs_share2/code/donghee/LAVIS/snapme_final2.json', 'r') as f:
        im2ingr = json.load(f)

    snapme_images = []
    snapme_paths = ['/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/00aed7f846529795993f19942c0d.jpeg', '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/0b90e149404986da7d5930438421.jpeg', '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/0b4297584ab4a5813c4664fde485.jpeg', '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/0d96e4264601b476277c7ee51232.jpeg', '/nfs_share2/code/donghee/inversecooking/snapme/snapme_mydb/0d0258474903a20af0b59f165e3d.jpeg']

    snapme_ids = []
    original_snapme_images = []
    images_for_tensor = []
    for p in snapme_paths:
        image = Image.open(p).convert('RGB')
        original_snapme_images.append(image)
        image = vis_processors['eval'](image)
        # image = vis_processors["eval"](image).unsqueeze(0).to(device)
        images_for_tensor.append(image.to(device))
        snapme_images.append(image.unsqueeze(0).to(device))
        snapme_ids.append(p.split('/')[-1])

    images_for_tensor = torch.stack(images_for_tensor)
    snapme_answers = []





    ## recipe1M test
    import numpy as np
    import os
    import pickle

    test_data = pickle.load(open('/nfs_share2/code/donghee/inversecooking/data/recipe1m_test.pkl', 'rb'))

    samples = []
    recipe_gt = []
    i=0
    while len(samples)<20:
        if len(test_data[i]['images']) != 0:
            samples.append(test_data[i])
            recipe_gt.append(test_data[i]['ingredients']) ## TODO
        i += 1

    original_recipe_images = []
    recipe_images = []
    base_path = '/nfs_share2/code/donghee/inversecooking/data/images/test'
    recipe_paths = []
    for sample in samples:
        id = sample['images'][0]
        p = os.path.join(base_path, id[0], id[1],id[2],id[3],id)
        recipe_paths.append(p)

    # recipe_paths = ['/nfs_share2/code/donghee/inversecooking/data/images/test/9/1/2/6/9126297c7d.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/9/1/2/7/91273289fb.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/9/1/3/8/913851905f.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/9/1/5/1/915114360b.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/2/0002839c83.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/3/0003967721.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/4/0004a1d74e.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/4/0004d32dec.jpg', '/nfs_share2/code/donghee/inversecooking/data/images/test/0/0/0/4/00049b8b85.jpg']

    recipe_ids = []
    images_for_tensor_r = []
    for p in recipe_paths[:10]:
        image = Image.open(p).convert('RGB')
        original_recipe_images.append(image)
        image = vis_processors["eval"](image)
        recipe_images.append(image.unsqueeze(0).to(device))
        images_for_tensor_r.append(image.to(device))
        recipe_ids.append(p.split('/')[-1])

    images_for_tensor_r = torch.stack(images_for_tensor_r)
    print(len(recipe_images))


    prompt = "Question: What are the ingredients I need to make this food? Answer:"
    print(images_for_tensor_r.shape)
    # answer_r = model.generate({"image": images_for_tensor_r, "prompt": prompt})
    # answer = model.generate({"image": images_for_tensor, "prompt": prompt})

    recipe_answers = []
    for gt, image in zip(recipe_gt, recipe_images):
        # answer = model.generate({"image": image, "prompt": prompt})  
        answer = model.eval_metrics({'image': image, 'text_input': prompt, 'text_output': ['apple, salt, pepper']})

        recipe_answers.append(answer)

    print("Need to check")

    for i in range(len(recipe_images)):
        print("===== InstructBlip vicuna7b finetuned (snapme) ====")
        plt.imshow(original_snapme_images[i])
        plt.title(snapme_ids[i])
        plt.show()
        print("- Prediction: ", snapme_answers[i][0]) ## TODO np array로 바꾸고 +1?
        # gt = np.array(snapme_ids[i]['gt_int']) + 1 ## TODO gt 접근해야함 -> snapme_answers + 1?
        # print("- GT: ", gt)
        print()


if __name__ == '__main__':
    # snapme_process()
    snapme_test()





