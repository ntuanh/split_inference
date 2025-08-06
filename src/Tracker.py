import cv2
import pika
import pickle
import yaml
import torch
import numpy as np
import time
import threading

from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from src.Model import SplitDetectionModel


class Tracker:
    def __init__(self, config):
        rabbit_config = config.get("rabbit", {})
        address = rabbit_config.get("address")
        username = rabbit_config.get("username")
        password = rabbit_config.get("password")
        virtual_host = rabbit_config.get("virtual-host")

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials)
        )
        self.channel = self.connection.channel()
        print("[Tracker] Connected to RabbitMQ.")

        self.bbox_queue = "bbox_queue"
        self.ori_img_queue = "ori_img_queue"

        self.bbox_buffer = {}
        self.image_buffer = {}

        self.stop_event = threading.Event()
        self.image_stream_stopped = False
        self.bbox_stream_stopped = False

        model_name = config.get('server', {}).get('model', 'yolov8n')
        try:
            # 1. Tải đối tượng YOLO đầy đủ
            print(f"[Tracker] Loading YOLO model '{model_name}.pt'...")
            yolo_model = YOLO(f"{model_name}.pt")

            # 2. Lấy predictor đã được tạo sẵn từ bên trong đối tượng YOLO.
            #    Đây là cách làm chính thống và an toàn nhất.
            self.predictor = yolo_model.predictor

            # 3. Nếu predictor chưa được tạo (do cơ chế lazy-loading),
            #    hãy trigger một phiên predict "giả" để buộc nó phải khởi tạo.
            if self.predictor is None:
                print("[Tracker] Predictor not initialized. Triggering dummy prediction...")
                _ = yolo_model
                predict
                source = np.zeros((1, 1, 3), dtype=np.uint8)  # Một ảnh đen 1x1
                self.predictor = yolo_model.predictor
                if self.predictor is None:
                    raise RuntimeError("Failed to initialize the predictor.")

            # 4. Lấy các thông tin cần thiết từ model và predictor
            self.class_names = self.predictor.model.names
            self.nms_args = self.predictor.args

            print("[Tracker] YOLO components loaded successfully.")
            print(f"[Tracker] Using NMS args: conf={self.nms_args.conf}, iou={self.nms_args.iou}")

        except Exception as e:
            raise RuntimeError(f"Could not load YOLO model or predictor '{model_name}.pt'. Error: {e}")

    def _declare_queues(self):
        self.channel.queue_declare(queue=self.bbox_queue, durable=False)
        self.channel.queue_declare(queue=self.ori_img_queue, durable=False)

    def _image_callback(self, ch, method, properties, body):
        try:
            if body is None :
                print("[Tracker] Received an empty image message. Skipping.")
                return

            message = pickle.loads(body)
            if message is None:
                print("[Tracker] Decoded image message is None. Skipping.")
                return

            self.process_image(message)
        except Exception as e:
            print(f"Error processing image message: {e}")
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def _bbox_callback(self, ch, method, properties, body):
        try:
            if body is None:
                print("[Tracker] Received an empty bbox message. Skipping.")
                return

            message = pickle.loads(body)
            if message is None:
                print("[Tracker] Decoded bbox message is None. Skipping.")
                return
            self.process_bbox(message)
        except Exception as e:
            print(f"Error processing bbox message: {e}")
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def process_bbox(self, message_obj):
        if message_obj == "STOP":
            print("[Tracker] STOP signal received from bbox queue.")
            self.bbox_stream_stopped = True
            return

        frame_index = message_obj.get("frame_index")
        predictions = message_obj.get("predictions")

        print(f"--- [Received BBox] Frame Index: {frame_index} ---")
        self.bbox_buffer[frame_index] = predictions

        if frame_index in self.image_buffer:
            self._process_pair(frame_index)

    def process_image(self, message):
        if message == "STOP":
            print("[Tracker] STOP signal received from image queue.")
            self.image_stream_stopped = True
            return

        frame_index = message.get("frame_index")
        frame = message.get("ori_img")

        print(f"--- [Received Image] Frame Index: {frame_index} ---")
        self.image_buffer[frame_index] = frame

        if frame_index in self.bbox_buffer:
            self._process_pair(frame_index)

    def _process_pair(self, frame_index):
        original_image = self.image_buffer.pop(frame_index, None)
        raw_predictions = self.bbox_buffer.pop(frame_index, None)

        if original_image is None or raw_predictions is None:
            print(f"--- [Error] Data missing for paired processing of frame {frame_index}. ---")
            return

        print(f"--- [Processing Paired] Frame Index: {frame_index} ---")

        annotated_frame = original_image

        try:
            device = raw_predictions.device
            dummy_img_tensor = torch.zeros(1, 3, 640, 640, device=device)

            # === THAY ĐỔI QUAN TRỌNG: Thiết lập self.batch thủ công ===
            # Đây là cách "đánh lừa" predictor một cách chính thống.
            # self.batch cần là một tuple chứa (path, img, orig_img, None)
            dummy_path = f"frame_{frame_index}.jpg"
            self.predictor.batch = ([dummy_path], dummy_img_tensor, [original_image], None)
            # =======================================================

            # Bây giờ gọi postprocess mà KHÔNG cần truyền path
            results_list = self.predictor.postprocess(
                preds=raw_predictions,
                img=dummy_img_tensor,
                orig_imgs=[original_image]
            )

            if results_list and len(results_list) > 0:
                final_results = results_list[0]

                if final_results.boxes and len(final_results.boxes) > 0:
                    annotated_frame = final_results.plot()
                    print(f"--- [Info] Plotted {len(final_results.boxes)} boxes for frame {frame_index}. ---")
                else:
                    print(f"--- [Info] No objects detected in frame {frame_index} after NMS. ---")

        except Exception as e:
            print(f"--- [ERROR] in _process_pair for frame {frame_index}: {e} ---")
            import traceback
            traceback.print_exc()

        cv2.imshow("Tracker Output", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_event.set()

    def start_listening(self):
        self._declare_queues()
        self.channel.basic_consume(queue=self.ori_img_queue, on_message_callback=self._image_callback, auto_ack=False)
        self.channel.basic_consume(queue=self.bbox_queue, on_message_callback=self._bbox_callback, auto_ack=False)

        print("[Tracker] Listening... Press Ctrl+C or 'q' on video window to exit.")
        start_time = time.time()

        while not (self.image_stream_stopped and self.bbox_stream_stopped):
            if self.stop_event.is_set():
                print("[Tracker] User interruption detected.")
                break
            self.connection.process_data_events(time_limit=0.5)

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
        cv2.destroyAllWindows()
        if self.connection and self.connection.is_open:
            self.connection.close()
            print("[Tracker] Connection closed.")

