import torch
import time
import random
from tqdm import tqdm
import numpy as np
from utils.utils import *
from utils.plot_utils import *
from torch import nn
import pdb
import wandb

wandb.init(settings=wandb.Settings(start_method="fork"))
wandb.init(project="tqn-project", entity="aniket99")


def train_one_epoch(args,epoch,net,optimizer,trainset,train_loader,SUFB = False):
    np.random.seed(epoch)
    random.seed(epoch)
    net.train()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = [AverageMeter()]
    accuracy = [AverageMeter(),AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
    criterion = nn.CrossEntropyLoss(reduction='mean') 

    t0 = time.time()   

    for j, batch_samples in enumerate(train_loader):
        data_time.update(time.time() - t0)


        # cls_targets: action class labels 
        # att_targets: attribute labels 
        if not SUFB:
            v_ids, seq, cls_targets, n_clips_per_video, att_targets = batch_samples
            if seq is None:
                continue
            mask = tfm_mask(n_clips_per_video)
            preds1, preds2, preds3, preds4, cls_preds = net((seq,mask))
            # pdb.set_trace()
        else:
            # ptrs: clip pointers, where the online sampled clips start
            v_ids, seq, cls_targets, ptrs, att_targets = batch_samples
            preds1, preds2, preds3, preds4, cls_preds = net((seq,v_ids,ptrs))


        # pdb.set_trace()
        cls_targets = cls_targets.cuda()
        att_targets = att_targets.cuda()
        # print(att_targets.shape)
        # pdb.set_trace()
        match_acc1, match_acc2, match_acc3, match_acc4 = multihead_acc(preds1, preds2, preds3, preds4, cls_targets, att_targets, \
            trainset.class_tokens, Q = args.num_queries)

        # preds = preds.reshape(-1, args.attribute_set_size)
        cls_acc = calc_topk_accuracy(cls_preds, cls_targets, (1,))[0]

        # print('Class acc: ', cls_acc.item())
        # print('Match acc: ', match_acc.item())

        acc = [torch.stack([cls_acc, match_acc1, match_acc2, match_acc3, match_acc4], 0).unsqueeze(0)]
        cls_acc, match_acc1, match_acc2, match_acc3, match_acc4 = torch.cat(acc, 0).mean(0)
        # pdb.set_trace()

        loss = criterion(preds1, att_targets[:, 0])
        loss += criterion(preds2, att_targets[:, 1])
        loss += criterion(preds3, att_targets[:, 2])
        loss += criterion(preds4, att_targets[:, 3])
        loss += criterion(cls_preds, cls_targets)
        
        # print('Class acc: ', cls_acc.item())
        # print('Match acc: ', match_acc.item())

        accuracy[0].update(match_acc1.item(), args.batch_size)
        accuracy[1].update(match_acc2.item(), args.batch_size)
        accuracy[2].update(match_acc3.item(), args.batch_size)
        accuracy[3].update(match_acc4.item(), args.batch_size)
        accuracy[4].update(cls_acc.item(), args.batch_size)
        losses[0].update(loss.item(), args.batch_size)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_norm) 
        optimizer.step()

        torch.cuda.empty_cache()
        batch_time.update(time.time() - t0)
        t0 = time.time()

        if j % (args.print_iter) == 0:
            t1 = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss[0].val:.4f} Acc: {acc[0].val:.4f}\t'
              'T-data:{dt.val:.2f} T-batch:{bt.val:.2f}\t'.format(
              epoch, j, len(train_loader), 
              loss=losses, acc=accuracy, dt=data_time, bt=batch_time))

            # args.train_plotter.add_data('local/loss', losses[0].local_avg, epoch*len(train_loader)+j)
            # args.train_plotter.add_data('local/match_acc', accuracy[0].local_avg,epoch*len(train_loader)+j)
            # args.train_plotter.add_data('local/cls_acc', accuracy[1].local_avg, epoch*len(train_loader)+j)
            torch.cuda.empty_cache()

    if epoch % args.save_epoch == 0:
        print('Saving state, epoch: %d iter:%d'%(epoch, j))
        save_ckpt(net,optimizer,args.best_acc,epoch,args.save_folder,str(epoch),SUFB)

    save_ckpt(net,optimizer,args.best_acc,epoch,args.save_folder,'latest',SUFB)
    
    wandb.log({"loss": loss})

    train_acc = [i.avg for i in accuracy]
    print('q1 acc: ', train_acc[0])
    print('q2 acc: ', train_acc[1])
    print('q3 acc: ', train_acc[2])
    print('q4 acc: ', train_acc[3])
    print('global/cls_acc: ', train_acc[4])


    file1 = open(args.acc_txt + "myfile.txt","a")
    file1.write("q1 acc: "+ str(train_acc[0])+ " q2 acc: "+ str(train_acc[1])+ " q3 acc: "+ str(train_acc[2])+ " q4 acc: "+ str(train_acc[3]) + " \n")
    file1.close()

    # print('global/match_acc: ', train_acc[1])
    # args.train_plotter.add_data('global/loss', [i.avg for i in losses], epoch)        
    # args.train_plotter.add_data('global/match_acc', accuracy[0].local_avg, epoch)
    # args.train_plotter.add_data('global/cls_acc', accuracy[1].local_avg, epoch)




def eval_one_epoch(args,epoch,net,testset,test_loader,SUFB = False):
    net.eval()
    test_accuracy = [AverageMeter(),AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
    np.random.seed(epoch+1)
    random.seed(epoch+1)

    with torch.no_grad():
        for k, batch_samples in tqdm(enumerate(test_loader),total=len(test_loader)):

            # cls_targets: action class labels 
            # att_targets: attribute labels
            if not SUFB:
                v_ids,seq,cls_targets,n_clips_per_video,att_targets = batch_samples
                if seq is None:
                    continue
                mask = tfm_mask(n_clips_per_video)
                preds1, preds2, preds3, preds4, cls_preds = net((seq,mask))
            else:
 
                # ptrs: clip pointers, where the online sampled clips start
                v_ids,seq,cls_targets,ptrs,att_targets = batch_samples
                preds1, preds2, preds3, preds4, cls_preds = net((seq,v_ids,ptrs))

            cls_targets = cls_targets.cuda()
            att_targets = att_targets.cuda()
            match_acc1, match_acc2, match_acc3, match_acc4  = multihead_acc(preds1, preds2, preds3, preds4,cls_targets, att_targets, \
                testset.class_tokens, Q=args.num_queries)

            # preds = preds.reshape(-1,args.attribute_set_size)
            cls_acc = calc_topk_accuracy(cls_preds, cls_targets, (1,))[0]

            acc = [torch.stack([cls_acc, match_acc1, match_acc2, match_acc3, match_acc4], 0).unsqueeze(0)]
            cls_acc, match_acc1, match_acc2, match_acc3, match_acc4 = torch.cat(acc, 0).mean(0)

            test_accuracy[0].update(match_acc1.item(), args.batch_size)
            test_accuracy[1].update(match_acc2.item(), args.batch_size)
            test_accuracy[2].update(match_acc3.item(), args.batch_size)
            test_accuracy[3].update(match_acc4.item(), args.batch_size)
            test_accuracy[4].update(cls_acc.item(), args.batch_size)

            torch.cuda.empty_cache()

        test_acc = [i.avg for i in test_accuracy]
        print('q1 acc: ', test_acc[0])
        print('q2 acc: ', test_acc[1])
        print('q3 acc: ', test_acc[2])
        print('q4 acc: ', test_acc[3])
        print('global/cls_acc: ', test_acc[4])
        # args.val_plotter.add_data('global/cls_acc',test_acc[0], epoch)
        # args.val_plotter.add_data('global/match_acc',test_acc[1], epoch)


    if test_acc[4] > args.best_acc:
        args.best_acc = test_acc[4]
        torch.save({'model_state_dict': net.state_dict(),\
            'best_acc':test_acc[4]},\
            args.save_folder + '/'  +  'best.pth')



def save_ckpt(net,optimizer,best_acc,epoch,save_folder,name,SUFB):
    if SUFB:
        torch.save({'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 
            'queue':net.module.queue,
            'best_acc':best_acc,
            'epoch':epoch},
            save_folder + '/'  +  name+'.pth')

    else:
        torch.save({'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc':best_acc,
            'epoch':epoch},
            save_folder + '/'  +  name+'.pth')
