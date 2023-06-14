import torch
import os 
import os.path as osp
import pdb

def tfm_mask(seg_per_video,temporal_mutliplier=1):
    """
    Attention mask for padded sequence in the Transformer
    True: not allowed to attend to 
    """
    B = len(seg_per_video)
    L = max(seg_per_video) * temporal_mutliplier
    mask = torch.ones(B,L,dtype=torch.bool)
    for ind,l in enumerate(seg_per_video):
        mask[ind,:(l*temporal_mutliplier)] = False

    return mask



def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def multihead_acc(preds1, preds2, preds3, preds4, clabel,target,vocab,\
    Q=4,return_probs=False):
    """
    Args:
    preds: Predicted logits
    clabel: Class labels,
            List, [batch_size]
    target: Ground Truth attribute labels
            List, [batch_size,num_queries]
    vocab: The mapping between class index and attributes. 
           List, [num_classes,num_queries]
    Q: Number of queries, Int

    Output:
    prob_acc: match predicted attibutes to ground-truth attibutes of N classes,
              class with the highest similarity is the predicted class. 
    """
    # pdb.set_trace()
    # reshape the preds to (B,num_heads,num_classes)
    # if len(preds1.shape)==2:
    #     BQ,C = preds1.shape
    #     B = BQ//Q
    #     preds1 = preds1.view(-1,Q,C)
    # elif len(preds1.shape)==3:
    #     B,Q,C = preds1.shape

    batch_size = target.size(0)

    _, soft_1 = preds1.topk(1, 1, True, True)
    soft_1 = soft_1.t()
    correct_1 = soft_1.eq(target[:, 0].view(1, -1).expand_as(soft_1))

    _, soft_2 = preds2.topk(1, 1, True, True)
    soft_2 = soft_2.t()
    correct_2 = soft_2.eq(target[:, 1].view(1, -1).expand_as(soft_2))

    _, soft_3 = preds3.topk(1, 1, True, True)
    soft_3 = soft_3.t()
    correct_3 = soft_3.eq(target[:, 2].view(1, -1).expand_as(soft_3))

    _, soft_4 = preds4.topk(1, 1, True, True)
    soft_4 = soft_4.t()
    correct_4 = soft_4.eq(target[:, 3].view(1, -1).expand_as(soft_4))


    #### MAKE QUESTION-SPECIFIC ACC.
    # pdb.set_trace()
    # target = target.view(-1,Q)
    # vocab_onehot = one_hot(vocab,C)
    # # pdb.set_trace()

    # cls_logits =torch.einsum('bhc,ahc->ba', preds, vocab_onehot.cuda())
    # cls_pred = torch.argmax(cls_logits,dim=-1)
    # prob_acc =  (cls_pred == clabel).sum()*100.0 /B

    met1 = []
    met2 = []
    met3 = []
    met4 = []

    correct_1 = correct_1[:1].view(-1).float().sum(0)
    met1.append(correct_1.mul_(100.0 / batch_size))

    correct_2 = correct_2[:1].view(-1).float().sum(0)
    met2.append(correct_2.mul_(100.0 / batch_size))

    correct_3 = correct_3[:1].view(-1).float().sum(0)
    met3.append(correct_3.mul_(100.0 / batch_size))

    correct_4 = correct_4[:1].view(-1).float().sum(0)
    met4.append(correct_4.mul_(100.0 / batch_size))

    return met1[0], met2[0], met3[0], met4[0]
    


def one_hot(indices,depth):
    """
    make one hot vectors from indices
    """
    y = indices.unsqueeze(-1).long()
    y_onehot = torch.zeros(*indices.shape,depth)
    if indices.is_cuda:
        y_onehot = y_onehot.cuda()
    return y_onehot.scatter(-1,y,1)



def make_dirs(args):

    if osp.exists(args.save_folder) == False:
        os.mkdir(args.save_folder)
    args.save_folder = osp.join(args.save_folder ,args.name)
    if osp.exists(args.save_folder) == False:
        os.mkdir(args.save_folder)

    args.tbx_dir =osp.join(args.tbx_folder,args.name)
    if osp.exists(args.tbx_folder) == False:
        os.mkdir(args.tbx_folder)

    if osp.exists(args.tbx_dir) == False:
        os.mkdir(args.tbx_dir)

    result_dir = osp.join(args.tbx_dir,'results')
    if osp.exists(result_dir) == False:
        os.mkdir(result_dir)



def batch_denorm(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1):
    """
    De-normalization the images for viusalization
    """
    shape = [1]*tensor.dim(); shape[channel] = 3
    dtype = tensor.dtype 
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device).view(shape)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device).view(shape)
    output = tensor.mul(std).add(mean)
    return output
