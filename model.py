import datetime
import cv2
from PIL import Image as Im
import io
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import base64
matplotlib.use('agg')


class Model():
    DISEASE_CLASSES = [
        'grape Black rot',
        'grape esca',
        'grape leaf blight',
        'potato early blight',
        'potato late blight'
    ]

    def __init__(self):
        register_coco_instances(
            "train_dataset", {}, "dataset/train/result.json", "dataset/train")
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "./model/model_final.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
        self.metadata = MetadataCatalog.get("train_dataset")
        self.predictor = DefaultPredictor(cfg)

    def predict(self, im):
        print(datetime.datetime.now())
        outputs = self.predictor(im)
        print(datetime.datetime.now())
        v = Visualizer(
            im[:, :, ::-1], {"thing_classes": self.DISEASE_CLASSES}, scale=1)
        visualizer = Visualizer(im, metadata=self.metadata)
        visualizer = v.draw_instance_predictions(
            outputs["instances"].to("cpu"))
        display_img = visualizer.get_image()  # BGR to RGB
        data = Im.fromarray(display_img)
        buffered = io.BytesIO()
        data.save(buffered, format="JPEG")
        print(outputs)
        response = {
            "image": base64.b64encode(buffered.getvalue()),
            "class": [
                self.DISEASE_CLASSES[i] for i in outputs["instances"].pred_classes
            ],
            "score": outputs["instances"].scores.tolist()
        }
        return response
