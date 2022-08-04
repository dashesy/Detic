import sys
import cv2
import tempfile
from pathlib import Path
import torch
import torch.nn as nn

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder


cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
predictor = DefaultPredictor(cfg)

BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}
BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

vocabulary = 'lvis'
image = "desk.jpg"
image = cv2.imread(image)

metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
classifier = BUILDIN_CLASSIFIER[vocabulary]
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)

# outputs0 = predictor(image)

if predictor.input_format == "RGB":
    # whether the model expects BGR inputs or RGB
    image = image[:, :, ::-1]
height, width = image.shape[:2]
image_byte = predictor.aug.get_transform(image).apply_image(image).transpose(2, 0, 1)
image_byte = torch.as_tensor(image_byte).cuda()

class FasterRCNN(nn.Module):
    """Wrap FasterRCNN and return tensors
    """
    def __init__(self, net):
        super(FasterRCNN, self).__init__()
        self.model = net

    def forward(self, x, height, width):
        inputs = {"image": x.float(), "height": height, "width": width}
        predictions = self.model([inputs])[0]
        instances = predictions['instances']
        return instances.pred_boxes.tensor, instances.scores, instances.pred_classes

m = FasterRCNN(predictor.model)

boxes, scores, labels = m(image_byte, height, width)
gg
onnxfile = "/repos/output/detic.onnx"
targets = ["bbox", "scores", "labels"]
dynamic_axes = {'image': {1 : 'height', 2: 'width'}}
dynamic_axes.update({t: {0: 'i'} for t in targets})
torch.onnx.export(m, (image_byte, height, width), onnxfile,
                  verbose=True,
                  input_names=['image', 'height', 'width'],
                  dynamic_axes=dynamic_axes,
                  output_names=targets,
                  opset_version=14)

def optimize_graph(onnxfile, onnxfile_optimized=None, providers=None):
    if providers is None:
        providers = 'CPUExecutionProvider'
    import onnxruntime as rt

    if not onnxfile_optimized:
        onnxfile_optimized = onnxfile[:-5] + "_optimized.onnx"  # ONNX optimizer is broken, using ORT to optimzie
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = onnxfile_optimized
    _ = rt.InferenceSession(onnxfile, sess_options, providers=[providers])
    return onnxfile_optimized

optimize_graph(onnxfile)
