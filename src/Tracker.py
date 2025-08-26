import pika
import pickle
import yaml
import torch
import numpy as np
import threading
import time
import cv2

from ultralytics.utils import ops

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor

from src.Utils import write_partial

from src.Model import BoundingBox


class Tracker:
    def __init__(self, config):
        rabbit_config = config.get("rabbit", {})
        credentials = pika.PlainCredentials(rabbit_config.get("username"), rabbit_config.get("password"))
        params = pika.ConnectionParameters(
            host=rabbit_config.get("address"),
            virtual_host=rabbit_config.get("virtual-host"),
            credentials=credentials
        )
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
        print("[Tracker] Connected to RabbitMQ.")

        self.bbox_queue = "bbox_queue"
        self.ori_img_queue = "ori_img_queue"

        self.bbox_buffer = {}
        self.image_buffer = {}

        self.stop_event = threading.Event()
        self.image_stream_stopped = False
        self.bbox_stream_stopped = False

        self.fps = 30
        self.start_receive_bounding_box = 0
        self.start_receive_origin_image = 0

        self.orig_img_size = (0 , 0)

        self.dict_data = {
            "[T]totalTm" : 0
            ,"[T]totalFr" : 0
            ,"[T]TmRecv" : 0
            ,"[T]FRPS" : 0
        }

        self.start_time_received = 0
        self.total_frames = 0
        self.digits = 5



    def _declare_queues(self):
        self.channel.queue_declare(queue=self.bbox_queue, durable=False)
        self.channel.queue_declare(queue=self.ori_img_queue, durable=False)

    def _image_callback(self, ch, method, properties, body):
        try:
            message = pickle.loads(body)
            if message == 'STOP':
                print("[Tracker] STOP signal received from image queue.")
                print(f"[Origin Image][Time] {time.time() - self.start_receive_origin_image}")
                self.image_stream_stopped = True
                return

            frame_index = message.get("frame_index")
            frame = message.get("ori_img")
            total_frames = message.get("total_frames")

            if total_frames != -1 :
                self.total_frames = total_frames
            self.orig_img_size = message.get("orig_img_size")

            if frame_index == 0:
                self.start_receive_origin_image = time.time()

            # print(f"--- [Received Image] Frame Index: {frame_index} ---")
            self.image_buffer[frame_index] = frame

            if frame_index in self.bbox_buffer:
                self._process_pair(frame_index)
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def _bbox_callback(self, ch, method, properties, body):
        try:
            message = pickle.loads(body)
            if message == 'STOP':
                print("[Tracker] STOP signal received from bbox queue.")
                print(f"[Bouding Box][Time] {time.time() - self.start_receive_bounding_box}")
                self.bbox_stream_stopped = True
                return

            frame_index = message.get("frame_index")
            predictions = message.get("predictions")

            if frame_index == 0:
                self.start_receive_bounding_box = time.time()

            # print(f"--- [Received BBox] Frame Index: {frame_index} ---")
            # print(f"[Predictions] check type {type(predictions)}")
            self.bbox_buffer[frame_index] = predictions

            if frame_index in self.image_buffer:
                self._process_pair(frame_index)

        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def start_listening(self):
        self._declare_queues()
        self.channel.basic_consume(queue=self.ori_img_queue, on_message_callback=self._image_callback, auto_ack=False)
        self.channel.basic_consume(queue=self.bbox_queue, on_message_callback=self._bbox_callback, auto_ack=False)

        print("[Tracker] Listening for confirmation... Press Ctrl+C to exit.")
        start_time = time.time()

        while not (self.image_stream_stopped and self.bbox_stream_stopped):
            if self.stop_event.is_set():
                break
            self.connection.process_data_events(time_limit=1)

        total_time = time.time() - start_time
        print(f"[Tracker][Time] total time: {total_time:.2f}s")
        self.dict_data["[T]totalTm"] = round(total_time , self.digits)
        write_partial(self.dict_data)

        print("\n[Tracker] All streams stopped. Loop finished.")

    def run(self):
        try:
            self.start_listening()
        except KeyboardInterrupt:
            print("\n[Tracker] Interrupted by user.")
            self.stop_event.set()
        finally:
            self.cleanup()

    def cleanup(self):
        print("[Tracker] Cleaning up...")
        if self.connection and self.connection.is_open:
            self.connection.close()
            cv2.destroyAllWindows()
            print("[Tracker] Connection closed.")

    def _process_pair(self, frame_index):
        predictor = BoundingBox()
        self.dict_data["[T]totalFr"] = self.total_frames
        # print("[Frame] index " , frame_index)

        if frame_index == 1:
            self.start_time_received = time.time()
        elif frame_index == self.total_frames :
            total_real = time.time() - self.start_time_received
            self.dict_data["[T]TmRecv"] = round(total_real , self.digits)
            self.dict_data["[T]FRPS"] = round(self.total_frames / total_real , self.digits)
        # print(f"[Start time received] {self.start_time_received}")


        origin_frame_test = self.image_buffer[frame_index]
        raw_prediction_tensor = self.bbox_buffer[frame_index]

        origin_frame_shape = origin_frame_test.shape
        origin_frame_width , origin_frame_height = origin_frame_shape[:2]

        display = True
        if display:

            orig_imgs_list = [origin_frame_test]
            tensor = torch.zeros(origin_frame_width, origin_frame_height)

            results = predictor.postprocess(
                preds=raw_prediction_tensor,
                resized_shape=(640 , 640),
                orig_shape=origin_frame_shape[:2],
                orig_imgs=orig_imgs_list
            )

            if results:
                final_result = results[0]
                annotated_image = final_result.plot()
                annotated_image = annotated_image[0:self.orig_img_size[0] , 0 : self.orig_img_size[1]]
                # cv2.imshow("Visual Detection Output", annotated_image)
                # cv2.waitKey(int(1000 / self.fps))


