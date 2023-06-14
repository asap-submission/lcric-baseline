import os, sys, glob

# import pickle
import torch
import random 
import math 
import time 
import json

import torchvision
import numpy as np 
import _pickle as cp
import os.path as osp
import cv2
import os
import random

from tqdm import tqdm

from PIL import Image
from torch.nn.utils.rnn import pad_sequence


class TQN_dataloader(object):
    def __init__(self, args, mode='train',transform=None,SUFB=False):


        self.root = args.root
        self.SUFB = SUFB
        self.mode = mode
        self.dataset = args.dataset
        self.img_dim = args.img_dim

        self.transform = transform
        self.clip_len = args.clip_len
        self.downsample = args.downsample
        self.max_length = args.max_length

        self.img_dim = args.img_dim

        random.seed(args.seed)

        # self.label2id = cp.load(open(osp.join('../annotations/',args.dataset+'_label2id.pkl'),'rb'))
        # self.id2label = cp.load(open(osp.join('../annotations/',args.dataset+'_id2label.pkl'),'rb'))

        # if 'diving' in args.dataset:

        #     if mode=='train':
        #         self.gts = json.load(open(osp.join(self.root,'Diving48_V2_train.json'),'rb'))
        #     else:
        #         self.gts = json.load(open(osp.join(self.root,'Diving48_V2_test.json'),'rb'))
           
        #     self.vocab = json.load(open('../annotations/diving48_vocab.json','rb'))

        #     class_tokens = []
        #     for a in self.vocab:
        #         gt_class = torch.tensor([self.label2id[i] for i in a])
        #         class_tokens.append(gt_class)

        #     self.class_tokens = torch.stack(class_tokens,0)

        # elif 'gym' in args.dataset:

        #     if mode =='train':
        #         self.gts = open(osp.join(self.root,'scripts',args.dataset+'_train_element_v1.1.txt'),'r').readlines()
        #     else:
        #         self.gts = open(osp.join(self.root,'scripts',args.dataset+'_val_element.txt'),'r').readlines()

        #     self.class_tokens = torch.stack([torch.tensor(i) for i in [*self.label2id.values()]],0)

        label2id = [[0, 0, 0, 0], [1, 0, 0, 4], [1, 0, 0, 1], [1, 0, 0, 2], 
                    [1, 0, 0, 6], [0, 1, 0, 0], [1, 0, 1, 2], [1, 0, 1, 1], 
                    [1, 0, 1, 5], [1, 0, 0, 3], [1, 0, 1, 3], [1, 1, 1, 5], 
                    [1, 1, 1, 1]]

        # label2id = [[0, 2, 4, 6], [1, 2, 4, 10], [1, 2, 4, 7], [1, 2, 4, 8], 
        #             [1, 2, 4, 12], [0, 3, 4, 6], [1, 2, 5, 8], [1, 2, 5, 7], 
        #             [1, 2, 5, 11], [1, 2, 4, 9], [1, 2, 5, 9], [1, 3, 5, 11], 
        #             [1, 3, 5, 7]]
        
        self.class_tokens = torch.tensor(label2id)

        self.videos = os.listdir(self.root + 'videos/')
        ann = []
        for i in self.videos:
            name = i.split('.')[0]
            ann.append(self.root + 'annotations/' + name + '/' + name + '.json')
        self.annotations = ann
        # print(ann)
       
        self.elements = self.preprocess(args.dataset)

        if mode=='train':
            self.elements = random.sample(self.elements, k=round(len(self.elements) * 0.75))
        else:
            self.elements = random.sample(self.elements, k=round(len(self.elements) * 0.25))
        # x=self.bn2(x)
        # x=self.relu(x)len(self.elements) * 0.25))

        if self.SUFB:
            # Use the Stochastically Updated Feature Bank
            self.K = args.K
            self.vid2id = cp.load(open(args.feature_file.replace('features','vid2id'),'rb'))


    def __getitem__(self, index):

        gt = self.elements[index]

        # if 'diving' in self.dataset:
        #     v_id = gt['vid_name']
        #     clabel = gt['label']
        #     frame_path = osp.join(self.root,'frames',v_id)
        #     total_frames = gt['end_frame'] - gt['start_frame']
        #     tokens = torch.tensor([self.label2id[i] for \
        #         i in self.vocab[clabel]])

        # elif 'gym' in self.dataset:
        #     v_id,clabel,cname = gt
        #     frame_path = osp.join(self.root,'frames',v_id)
        #     total_frames = len(os.listdir(frame_path)) 
        #     tokens = torch.tensor(self.label2id[int(clabel)])

        #######
        ####CHANGE ALL THESE THINGS LATER ON ONCE MORE DATA IS ADDED
        #######
        # label2id = [[0, 2, 4, 6], [1, 2, 4, 10], [1, 2, 4, 7], [1, 2, 4, 8], 
        #             [1, 2, 4, 12], [0, 3, 4, 6], [1, 2, 5, 8], [1, 2, 5, 7], 
        #             [1, 2, 5, 11], [1, 2, 4, 9], [1, 2, 5, 9], [1, 3, 5, 11], 
        #             [1, 3, 5, 7]]

        # label_dct = {'NOT-SCORED':0, 'SCORED':1, 'NOT-OUT':2, 'OUT':3, 'NOT-WIDE':4,
        #              'WIDE':5, 0:6, 1:7, 2:8, 3:9, 4:10, 5:11, 6:12}

        label2id = [[0, 0, 0, 0], [1, 0, 0, 4], [1, 0, 0, 1], [1, 0, 0, 2], 
                    [1, 0, 0, 6], [0, 1, 0, 0], [1, 0, 1, 2], [1, 0, 1, 1], 
                    [1, 0, 1, 5], [1, 0, 0, 3], [1, 0, 1, 3], [1, 1, 1, 5], 
                    [1, 1, 1, 1]]
        
        label_dct = {'NOT-SCORED':0, 'SCORED':1, 'NOT-OUT':0, 'OUT':1, 'NOT-WIDE':0,
                     'WIDE':1, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}

        v_id, overs, start_frame, end_frame, batter, bowler, runs, ques = gt
        # print(ques)
        total_frames = end_frame - start_frame
        # print(start_frame)
        vid_path = os.path.join(self.root, 'videos/' + v_id + '.mp4')
        tokens = torch.tensor([label_dct[i] for i in ques])
        clabel = label2id.index(tokens.tolist())

        downsample = self.set_downsample_rate(total_frames)

        # print(start_frame)
        if total_frames <=2:
            # skip broken samples
            return None,None,None,None

        elif self.mode != 'test':
            frames,ptr = self.sample_frames(total_frames,downsample)
            if len(frames) ==0:
                print(v_id,downsample,frames)

            #### Remember to use some sort of transformation (normalization and such)
            seq = self.load_images(vid_path,frames, start_frame, end_frame, total_frames)

        elif self.mode =='test':
            frames_list = self.sample_frames_test(total_frames,downsample)

            seq_list =[]
            for frames in frames_list:
                seq = self.load_images(vid_path,frames, start_frame, end_frame, total_frames)
                seq_list.append(seq)

            # align and stack seqs in the lists
            min_chunks = min([s.shape[0] for s in seq_list])
            seq_list = [s[:min_chunks,:] for s in seq_list]
            seq = torch.stack(seq_list,dim=0)

        clabel = torch.tensor(int(clabel))

        if self.SUFB:

            v_id = self.vid2id[v_id]
            assert seq.shape[0] == self.K
            return v_id, seq, clabel, ptr, tokens

        # print(clabel)
        # print(tokens)
        return v_id, seq, clabel ,tokens


    def load_images(self,frame_path,frames, start_frame, end_frame, total_frames):
        # load images and apply transformation 
        vc = cv2.VideoCapture(frame_path)
        vc.set(1, start_frame)
        seqs = []
        i = 0
        for i in range(total_frames):
            ret, frame = vc.read()
            frame = cv2.resize(frame, (self.img_dim, self.img_dim))
            if i in frames:
                seqs.append(frame)
            i+=1
        vc.release()

        seqs = self.pad_seq(seqs)

        # seq_names = [os.path.join(frame_path, 'image_%06d.jpg' % (i+1)) for i in frames]
        # seqs = [pil_loader(i) for i in seq_names]
        seqs = self.transform(seqs)   ### see the transform here
        seq = torch.stack(seqs, 1)

        # print(start_frame)
        # print(end_frame)
        # print(seq.shape)

        C,T,H,W = seq.shape # [NUM_CLIPS, C, CLIP_LEN, H, W]
        seq = seq.view(C,-1,self.clip_len,H,W).transpose(0,1) 
        return seq



    def sample_frames(self,total_frames,downsample):
        first_f = np.random.choice(np.arange(downsample+1))
        frames =  np.arange(first_f,total_frames,downsample).tolist()

        if self.SUFB:
            # randomly choose a start point in the video to sample K clips
            n_clips  = int(np.ceil(len(frames) / self.clip_len))
            ptr = np.random.choice(max(1,n_clips - self.K + 1))
            start = ptr * self.clip_len
            end = min([len(frames),(ptr + self.K) * self.clip_len])
            frames = frames[start:end]


            ### randomly remove some frames (CHECK THE HYPERPARAMETERS OF 0.05 HERE)
            # if self.mode == 'train':
            #     for _ in range(int(0.05*len(frames))+1):
            #         frames.remove(random.choice(frames))

            # pad the seq with the last frame to make the number of frames 
            # sampled equal to K * clip_len,
            # where K in the number of clips computed online 
            # in each iteration in the SUFB
            frames = self.pad_seq(frames)

        else:
            # temporal jittering
            # if self.mode == 'train':
            #     for _ in range(int(0.01*total_frames) + 1):
            #         frames.remove(random.choice(frames))

            # pad the seq with the last frame if the number of frames 
            # sampled is not divisiable by clip_len
            frames = self.pad_seq(frames)
            ptr = None

        return frames,ptr

    def sample_frames_test(self,total_frames,downsample):
        # temporal jittering for testing 
        frames = list(np.arange(0,total_frames,downsample))
        frames0 = self.pad_seq(frames)
        frames1 = self.pad_seq(self.drop_frames(frames))

        return [frames0,frames1]


    def pad_seq(self,frames):

        if not isinstance(frames,list):
            frames = frames.tolist()

        if self.SUFB:
            diff_T = self.clip_len * self.K - len(frames) 
        else:
            hanging_T = len(frames) % self.clip_len
            diff_T = 0
            if hanging_T !=0:
                diff_T = self.clip_len - hanging_T
            # print(diff_T)

        for i in range(diff_T):
            frames.append(frames[-1])
        return frames


    def preprocess(self,dataset):
        # Filter the videos by length for the 1st stage training
        elements= []
        # if 'diving' in dataset:
        #     for gt in self.gts:
        #         v_id, clabel, start_frame, end_frame = [*gt.values()]
        #         num_frames = start_frame - end_frame
        #         if num_frames < self.max_length:
        #             elements.append(gt)

        # elif 'gym' in dataset:
        #     self.dict =  cp.load(open(osp.join('../annotations',self.dataset+'_anno.pkl'),'rb'))
        #     for gt in self.gts:
        #         v_id,clabel = gt.split(' ')
        #         num_frames = int(self.dict[v_id]['num_frames'])
        #         cname = self.dict[v_id]['cname']
        #         if num_frames < self.max_length:
        #             elements.append((v_id,clabel,cname))

        for ann in self.annotations:
            with open(ann) as f:
                temp = json.load(f)
            for i in temp:
                elements.append(i)
        return elements    #### CHANGE IT


    def drop_frames(self,frames):
        total_frames = len(frames)
        new = frames.copy()
        for _ in range(int(0.02*total_frames)+1):
            new.remove(random.choice(new))
        return new 

    def set_downsample_rate(self,total_frames):
        downsample = self.downsample
        # print(downsample)
        while total_frames - downsample * self.clip_len < 1 and downsample > 1 :
            downsample -=1
        # print(downsample)
        return downsample 

    def __len__(self):
        return len(self.elements)

        
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def SUFB_collate(batch):

    ids = [b[0] for b in batch if b[0] is not None]
    if ids ==[]:
        return None,None,None,None
    else:
        seqs = [b[1] for b in batch if b[1] is not None]
        labels = [b[2] for b in batch if b[2] is not None]
        tokens = [b[-1] for b in batch if b[-1] is not None]

        seqs = torch.stack(seqs,dim=0)
        labels=torch.stack(labels,dim=0)
        tokens = pad_sequence(tokens,batch_first =True)

        if len(batch)>3:
            # train or val mode
            ptrs = torch.tensor([b[3] for b in batch if b[3] is not None])
            return torch.tensor(ids),seqs,labels,ptrs,tokens
        else:
            # test mode
            return  ids,seqs,labels,tokens


def collate(batch):
    ids = [b[0] for b in batch if b[0] is not None]
    seq = [b[1] for b in batch if b[1] is not None]
    label = [b[2] for b in batch if b[2] is not None]
    tokens = [b[-1] for b in batch if b[-1] is not None]

    if len(seq) ==0:
        return None,None,None,None,None
    else:
        Ks = [s.shape[0] for s in seq]
        seq = pad_sequence(seq,batch_first=True)
        label=torch.stack(label,dim=0)
        tokens = pad_sequence(tokens,batch_first =True)
        return ids,seq,label,Ks,tokens
