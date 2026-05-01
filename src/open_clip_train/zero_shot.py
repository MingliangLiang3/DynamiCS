import logging
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, ZERO_SHOT_TEMPLATES, ZERO_SHOT_CLASSNAMES
from .precision import get_autocast


def accuracy(output, labels, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def kitti_accuracy(output, labels, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)
        # to float
        output = output.float()
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_per_class(outputs, labelss):
    pred = outputs.argmax(1)
    confusion_matrix = metrics.confusion_matrix(labelss, pred)
    per_classes = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

    return per_classes.mean()


def roc_auc(outputs, labelss):
    pos_score = outputs[:, 1] - outputs[:, 0]
    metric = metrics.roc_auc_score(labelss, pos_score)

    return metric


def run(model, classifier, dataloader, args, is_acc=True):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    all_outputs = []
    all_labelss = []

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, labels in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            labels = labels.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            if is_acc:
                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)
            else:
                all_outputs.append(logits.cpu())
                all_labelss.append(labels.cpu())
    if is_acc:
        top1 = (top1 / n)
        top5 = (top5 / n)
        return top1, top5
    else:
        return torch.cat(all_outputs), torch.cat(all_labelss)


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results


def zero_shot_eval_dataset(model, data, epoch, args, tokenizer=None):
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    with open(args.zero_shot_eval_dataset) as f:
        catalog = json.load(f)
    results = {}
    for d in catalog:
        logging.info(f'Starting zero-shot classification {d}')
        # val_dataset = get_downstream_dataset(catalog, name=d, is_train=False, transform=val_transform)
        val_dataset = data['zero-shot-eval'][d]

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, drop_last=False)

        autocast = get_autocast(args.precision)
        with autocast():
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=ZERO_SHOT_CLASSNAMES[d],
                templates=ZERO_SHOT_TEMPLATES[d],
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        is_acc = d not in ['aircraft', 'pets', 'caltech101', 'flowers', 'kitti_distance', 'kinetics700_frames',
                           'patch_camelyon', 'hateful_memes', 'rendered_sst2']
        acc_or_outputs = run(model, classifier, val_loader, args, is_acc)

        if d in ['aircraft', 'pets', 'caltech101', 'flowers']:
            metric = mean_per_class(*acc_or_outputs)
        elif d == 'kinetics700_frames':
            top1, top5 = kitti_accuracy(*acc_or_outputs, topk=(1, 5))
            metric = (top1 + top5) / 2
            metric = metric.item()
        elif d in ['kitti_distance', 'patch_camelyon', 'rendered_sst2']:
            top1 = kitti_accuracy(*acc_or_outputs, topk=(1,))
            metric = top1[0].item() / 100
        elif d == 'hateful_memes':
            metric = roc_auc(*acc_or_outputs)
        else:
            metric = acc_or_outputs[0]

        results[f'{d}-zeroshot-val-top1'] = metric
        logging.info(f'Finished zero-shot classification {d} and {metric}.')

    return results


def imagenet_linear_probe_eval(model, data, epoch, args):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting linear probe imagenet.')

    autocast = get_autocast(args.precision)
    dataloader = data['imagenet-val'].dataloader
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, labels in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            labels = labels.to(args.device)

            with autocast():
                model_out = model(images)
            acc1, acc5 = accuracy(model_out['image_features'], labels, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)   

    results = {}
    top1 = (top1 / n)
    top5 = (top5 / n)
    results['imagenet-linear-probe-val-top1'] = top1
    results['imagenet-linear-probe-val-top5'] = top5

    logging.info('Finished linear probe imagenet.')
    return results