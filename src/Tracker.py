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

        print(f"[Tracker][Time] total time: {time.time() - start_time:.2f}s")
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

        origin_frame_test = self.image_buffer[frame_index]
        raw_prediction_tensor = self.bbox_buffer[frame_index]

        origin_frame_shape = origin_frame_test.shape
        origin_frame_width , origin_frame_height = origin_frame_shape[:2]

        # print(f"[Frame] : {frame_index}")
        # detections = self.process_yolo_output(raw_prediction_tensor)
        # print(detections[:5])

        # print(f"[BBox][type] : {type(raw_prediction_tensor)}")
        # print(f"[BBox][shape] : {raw_prediction_tensor.shape}")

        print(f"[Origin Image] type : {type(origin_frame_test)}")
        print(f"[Origin Image] shape : {origin_frame_shape[:2]}")
        # print(f"[Origin Image Width] :{origin_frame_width}")
        # print(f"[Origin Image Height]: {origin_frame_height}")

        display = True
        if display:

            orig_imgs_list = [origin_frame_test]
            tensor = torch.zeros(origin_frame_width, origin_frame_height)

            results = predictor.postprocess(
                preds=raw_prediction_tensor,
                resized_shape=(640 , 640),
                orig_shape=origin_frame_shape[:2],  #(480 , 852)
                orig_imgs=orig_imgs_list
            )

            if results:
                # print(f"[Result][type]{type(results)}")
                # print(f"[Result[0]][type]{type(results[0])}")
                # print("[Boxes]")
                # print(results[0].boxes)
                final_result = results[0]
                annotated_image = final_result.plot()
                annotated_image = annotated_image[0:self.orig_img_size[0] , 0 : self.orig_img_size[1]]
                cv2.imshow("Visual Detection Output", annotated_image)
                cv2.waitKey(int(1000 / self.fps))


