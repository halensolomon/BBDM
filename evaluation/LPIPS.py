import os
import random
import lpips
import torch
from tqdm.autonotebook import tqdm

loss_fn = lpips.LPIPS(net='alex', version='0.1').to(torch.device('cuda:0'))


@torch.no_grad()
def calc_LPIPS(data_dir, gt_dir, num_samples=1):
    dir_list = os.listdir(data_dir)
    dir_list.sort()

    total = len(dir_list)
    total_lpips_distance = 0
    for i in tqdm(range(total), total=total, smoothing=0.01):
        gt_name = os.path.join(gt_dir, f'{str(i)}.png')
        gt_img = torch.load(gt_name).to(torch.device('cuda:0'))
        # Need to change format from 1x1xHxW to Nx3xHxW
        gt_img = gt_img.unsqueeze(0).unsqueeze(0)
        gt_img.expand(num_samples, 3, gt_img.shape[2], gt_img.shape[3])
        
        # need to change this to allow .pt files
        for j in range(num_samples):
            if num_samples == 1:
                img_name = os.path.join(os.path.join(data_dir, f'{str(i)}.pt'))
            else:
                img_name = os.path.join(os.path.join(data_dir, str(i), f'output_{str(j)}.pt'))

            img_calc = torch.load(img_name).to(torch.device('cuda:0'))
            current_lpips_distance = loss_fn.forward(gt_img, img_calc.unsqueeze(0).unsqueeze(0).expand(num_samples, 3, img_calc.shape[2], img_calc.shape[3]))
            total_lpips_distance = total_lpips_distance + current_lpips_distance
        
        
        
        # for j in range(num_samples):
        #     if num_samples == 1:
        #         img_name = os.path.join(os.path.join(data_dir, f'{str(i)}.png'))
        #     else:
        #         img_name = os.path.join(os.path.join(data_dir, str(i), f'output_{str(j)}.png'))

            img_calc = torch.load(img_name).to(torch.device('cuda:0'))
            current_lpips_distance = loss_fn.forward(gt_img, img_calc)
            total_lpips_distance = total_lpips_distance + current_lpips_distance
    avg_lpips_distance = total_lpips_distance / (total * num_samples)
    print(data_dir)
    print(f'lpips_distance: {avg_lpips_distance}')
    return avg_lpips_distance


# calc_LPIPS(data_dir="/media/x/disk/BBDM/results/evaluation_temp_dir/CelebAMaskHQ/cyclegan",
#            gt_dir="/media/x/disk/BBDM/results/evaluation_temp_dir/CelebAMaskHQ/faces",
#            num_samples=1)

@torch.no_grad()
def random_LPIPS(data_dir, gt_dir, num_samples=1):
    dir_list = os.listdir(data_dir)
    dir_list.sort()

    total = len(dir_list)
    total_lpips_distance = 0
    for i in tqdm(range(total), total=total, smoothing=0.01):
        gt_name = os.path.join(gt_dir, f'{str(i)}.png')
        gt_img = torch.load(gt_name).to(torch.device('cuda:0'))
        j = random.randint(0, num_samples-1)
        img_name = os.path.join(os.path.join(data_dir, str(i), f'output_{str(j)}.png'))
        img_calc = torch.load(img_name).to(torch.device('cuda:0'))
        current_lpips_distance = loss_fn.forward(gt_img, img_calc)
        total_lpips_distance = total_lpips_distance + current_lpips_distance
    avg_lpips_distance = total_lpips_distance / total
    return avg_lpips_distance


@torch.no_grad()
def find_max_min_LPIPS(data_dir, gt_dir, num_samples=1):
    max_LPIPS = 0
    min_LPIPS = 10

    for i in range(100):
        avg_LPIPS = random_LPIPS(data_dir, gt_dir, num_samples=num_samples)
        if avg_LPIPS > max_LPIPS:
            max_LPIPS = avg_LPIPS
        if avg_LPIPS < min_LPIPS:
            min_LPIPS = avg_LPIPS
        if i % 20 == 0:
            print(f"{i} current_LPIPS = {avg_LPIPS}, max_LPIPS = {max_LPIPS}, min_LPIPS = {min_LPIPS}")
    print(data_dir)
    print(f'max_LPIPS = {max_LPIPS}, min_LPIPS = {min_LPIPS}')
