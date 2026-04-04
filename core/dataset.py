import os
import json
import random

import cv2
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

from utils.file_client import FileClient
from utils.img_util import imfrombytes
from utils.flow_util import resize_flow, flowread
from core.utils import (create_random_shape_with_random_motion, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip,GroupRandomHorizontalFlowFlip)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict):
        self.args = args
        self.video_root = args['video_root']
        self.flow_root = args['flow_root']
        self.num_local_frames = args['num_local_frames']
        self.num_ref_frames = args['num_ref_frames']
        self.size = self.w, self.h = (args['w'], args['h'])

        args['load_flow'] = True
        self.load_flow = args['load_flow']
        if self.load_flow:
            assert os.path.exists(self.flow_root)
        
        #json_path = os.path.join('./datasets', args['name'], 'train.json')

        #with open(json_path, 'r') as f:
        #    self.video_train_dict = json.load(f)
        #self.video_names = sorted(list(self.video_train_dict.keys()))
        
        self.video_names = sorted(os.listdir(self.video_root))
        mask_root = "datasets\\youtube-vos\\test_masks\\"
        test_masks = set(os.listdir(mask_root))
        
        #dont train on test data
        self.video_names = [v for v in self.video_names if v not in test_masks]
        
        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            if v_len > self.num_local_frames + self.num_ref_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list
                

        self.video_names = list(self.video_dict.keys()) # update names

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def __getitem__(self, index):
        video_name = self.video_names[index]
        
        #Disable this for finetunning
        Use_Random_masks = True
        
        #20% of the time we use the actual traning data
        if random.random() < 1.0:#0.2:
            Use_Random_masks = False
            
        if Use_Random_masks:
            # create masks
            all_masks = create_random_shape_with_random_motion(
                self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # create sample index
        selected_index = self._sample_index(self.video_dict[video_name],
                                            self.num_local_frames,
                                            self.num_ref_frames)

        # read video frames
        frames = []
        masks = []
        scaled_influences = []
        flows_f, flows_b = [], []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img_bytes = self.file_client.get(img_path, 'img')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgnp = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(imgnp)

            frames.append(img)
            
            if Use_Random_masks:
                mask = all_masks[idx]
            else:
                if random.random() < 0.5:
                    mask_folder = "datasets\\youtube-vos\\left_masks\\"
                else:
                    mask_folder = "datasets\\youtube-vos\\right_masks\\"
                
                mask_path = os.path.join(mask_folder, video_name, frame_list[idx] + '.png')
                mask = Image.open(mask_path).resize(self.size, Image.NEAREST).convert('L')

                # origin: 0 indicates missing. now: 1 indicates missing
                mask = np.asarray(mask)
                m = np.array(mask > 0).astype(np.uint8)

                mask = Image.fromarray(m * 255)
            masks.append(mask)

            if True:
                distance = distance_transform_edt(mask)
                
                min_pixels = 8
                max_pixels = 20
                
                min_scale_val = 0.0
                max_scale_val = 0.99
                
                
                #0.99(max_scale_val) in the midle of holes where there should be blur, 0 where there should be no blur
                scaled_influence = 1 - np.clip(min_scale_val + (max_pixels - distance) / (max_pixels - min_pixels) * (max_scale_val - min_scale_val), min_scale_val, max_scale_val)
                
                imgnp = imgnp.astype(np.float32) / 255.0
                
                blurred_gt = gaussian_filter(imgnp, sigma=(3, 3, 0))

                # 2. Mixa original GT och blurred GT med din scaled_influence
                # 0 vid kanten -> Original GT
                # 1 i mitten -> Blurred GT
                
                mask_3d = scaled_influence[:, :, np.newaxis] 
                
                soft_real_img = (1 - mask_3d) * imgnp + mask_3d * blurred_gt

                

                # Vi plockar ut Batch 0, Frame 0
                # Vi ändrar ordningen från (C, H, W) till (H, W, C) för att kunna rita den
                #img_to_show = soft_real_img#.transpose(1, 2, 0)

                # Om dina bildvärden ligger mellan -1 och 1, skala om dem till 0-1
                #if img_to_show.min() < 0:
                #    img_to_show = (img_to_show + 1) / 2
                
                if False:
                    plt.figure(figsize=(15, 7))

                    # Bild 1: Original GT
                    plt.subplot(1, 2, 1)
                    plt.title("Original GT (Skarp)")
                    # Flytta kanaler till sist: (C, H, W) -> (H, W, C)
                    img1 = imgnp#.transpose(1, 2, 0)
                    #if img1.min() < 0: img1 = (img1 + 1) / 2
                    plt.imshow(imgnp)
                    plt.axis('off')

                    # Bild 2: Soft GT (Den du skickar till Discriminatorn)
                    plt.subplot(1, 2, 2)
                    plt.title("Soft GT (Suddig i mitten)")
                    img2 = soft_real_img#.transpose(1, 2, 0)
                    
                    plt.imshow(np.clip(img2, 0, 1))
                    plt.axis('off')

                    plt.show()
                    exit()
                    
                scaled_influences.append(Image.fromarray((soft_real_img*255).astype(np.uint8)))
                
            #masks.append(all_masks[idx])

            if len(frames) <= self.num_local_frames-1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx+1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, f'{next_n}_{current_n}_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            if len(frames) == self.num_local_frames: # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    masks.reverse()
                    scaled_influences.reverse()
                    if self.load_flow:
                        flows_f.reverse()
                        flows_b.reverse()
                        flows_ = flows_f
                        flows_f = flows_b
                        flows_b = flows_
        
        
        if self.load_flow:
            frames, scaled_influences, masks, flows_f, flows_b = GroupRandomHorizontalFlowFlip()(frames, scaled_influences, masks, flows_f, flows_b)
        else:
            frames = GroupRandomHorizontalFlip()(frames)

        # normalizate, to tensors
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        
        if True:
            #tensor_scaled_influences = np.stack(scaled_influences)
            #tensor_scaled_influences = torch.from_numpy(tensor_scaled_influences) * 2.0 - 1.0
            tensor_scaled_influences = self._to_tensors(scaled_influences) * 2.0 - 1.0
        else:
            tensor_scaled_influences = torch.tensor([])
        
        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()
        
        #print("tensor_scaled_influences:", tensor_scaled_influences.shape)
        # img [-1,1] mask [0,1]
        if self.load_flow:
            return frame_tensors, mask_tensors, tensor_scaled_influences, flows_f, flows_b, video_name
        else:
            return frame_tensors, mask_tensors, tensor_scaled_influences, 'None', 'None', video_name


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.size = self.w, self.h = args['size']

        self.video_root = args['video_root']
        self.mask_root = args['mask_root']
        self.flow_root = args['flow_root']
        
        args['load_flow'] = True
        self.load_flow = args['load_flow']
        if self.load_flow:
            assert os.path.exists(self.flow_root)
        self.video_names = sorted(os.listdir(self.mask_root))

        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            self.video_dict[v] = v_len
            self.frame_dict[v] = frame_list

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        selected_index = list(range(self.video_dict[video_name]))

        # read video frames
        frames = []
        masks = []
        flows_f, flows_b = [], []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            frame_path = os.path.join(self.video_root, video_name, frame_list[idx])

            img_bytes = self.file_client.get(frame_path, 'input')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            frames.append(img)

            mask_path = os.path.join(self.mask_root, video_name, frame_list[idx] + '.png')
            mask = Image.open(mask_path).resize(self.size, Image.NEAREST).convert('L')

            # origin: 0 indicates missing. now: 1 indicates missing
            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)

            #m = cv2.dilate(m,
            #               cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
            #               iterations=4)
            mask = Image.fromarray(m * 255)
            masks.append(mask)

            if len(frames) <= len(selected_index)-1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx+1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, f'{next_n}_{current_n}_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

        # normalizate, to tensors
        frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        
        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        if self.load_flow:
            return frame_tensors, mask_tensors, flows_f, flows_b, video_name, frames_PIL
        else:
            return frame_tensors, mask_tensors, 'None', 'None', video_name