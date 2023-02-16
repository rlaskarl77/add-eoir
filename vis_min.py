# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/visualize.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/visualize.py --weights yolov5s.pt           # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (MacOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy)
from utils.metrics import ap_per_class
from utils.plots import output_to_target, plot_images
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
# from resources import get_resources_path

SAVE_ROOT = ROOT


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        project=SAVE_ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        line_thickness=3,  # bounding box thickness (pixels)
        with_labels=False,  # draw bbox with labels
        with_conf=False,  # draw bbox with confidences
        find_iou_range=[0.3, 0.95], # IoU visualization range
        callbacks=Callbacks(),
        ):
    # Initialize/load model and set device
    device = select_device(device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, pt, jit, onnx, engine = model.stride, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    elif engine:
        batch_size = model.batch_size
    else:
        half = False
        batch_size = 1  # export.py models default to batch-size 1
        device = torch.device('cpu')
        LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

    # Data
    data = check_dataset(data)  # check

    # Configure
    model.eval()
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # iou vector from 0.3 to 0.95 by step of 0.05
    iouv = np.linspace(.3, 0.95, int(np.round((0.95 - .3) / .05)) + 1, endpoint=True)
    iouv = torch.from_numpy(iouv).to(device)
    niou = iouv.numel()

    # Dataloader
    model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    # dataloader = create_dataloader(data['val'], imgsz, batch_size, stride, single_cls, pad=0.0, rect=pt,
    #                                workers=workers, prefix=colorstr('val'))[0]
    dataloader = create_dataloader(data['test'], imgsz, batch_size, stride, single_cls, pad=0.0, rect=pt,
                                   workers=workers, prefix=colorstr('test'))[0]

    seen = 0
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    print(names)
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    jdict, stats, ap, ap_class = [], [], [], []

    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        t1 = time_sync()
        if pt or jit or engine:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, _ = model(im, augment=False)  # inference, loss outputs

        # NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        out = non_max_suppression(out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=single_cls)

        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Calculate maximum IoU threshold that each prediction is regarded as true positive
            max_thresh = [5 * (i - 1) + 30 if i > 0 else 0 for i in torch.sum(correct.cpu(), 1).numpy()]

            # If max thresh falls within the predefined range, visualize the image
            r1, r2 = int(find_iou_range[0]*100), int(find_iou_range[1]*100)
            if any(r1 <= th < r2 for th in max_thresh):
                # Resize the image to original dimensions
                im0 = im[si].clone().cpu().numpy()
                im0 = np.transpose(im0, (1, 2, 0))*255
                im0 = np.ascontiguousarray(im0)
                im0 = cv2.resize(im0, (shape[1], shape[0]))
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                #TODO find false negative option?

                # Draw targets : blue
                if nl:
                    for cls, *xyxy in reversed(labelsn):
                        c = int(cls)  # integer class
                        label = f'{names[c]}' if with_labels else ''
                        annotator.box_label(xyxy, label, color=(255, 0, 0))

                # Draw predictions
                # False positive (thresh = 0) : red
                for thresh, (*xyxy, conf, cls) in zip(reversed(max_thresh), reversed(predn)):
                    c = 0 if thresh == 0 else (1 if r1 <= thresh < r2 else 2)
                    if with_conf or thresh == 0:
                        label = f'{names[c]} {thresh} {conf:.3f}' if with_labels else f'{thresh} {conf:.3f}'
                    else:
                        label = f'{names[c]} {thresh}' if with_labels else f'{thresh}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

                im0 = annotator.result()

                save_path = os.path.join(str(save_dir), 'visualization')
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(f'{save_path}/{seen:05d}.jpg', im0)

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--project', default=SAVE_ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--with-labels', action='store_true', help='draw bbox with labels')
    parser.add_argument('--with-conf', action='store_true', help='draw bbox with confidences')
    parser.add_argument('--find-iou-range', nargs='+', type=float, default=[0.3, 0.95],
                        help='visualize images where at least one object`s max positive IoU threshold is within '
                             'this range. ex) --find-iou-range 0.3 0.5')
    opt = parser.parse_args()

    if len(opt.find_iou_range) != 2:
        print("the length of find_iou_range argument must be 2")
        sys.exit()

    opt.data = check_yaml(opt.data)  # check YAML
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)