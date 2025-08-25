import pickle
import time
from tqdm import tqdm
import torch
import cv2
from src.Model import SplitDetectionPredictor
from src.Model import BoundingBox
from ultralytics.engine.results import Results

class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.intermediate_queue = f"intermediate_queue_{self.layer_id}"
        self.channel.queue_declare(self.intermediate_queue, durable=False)

        self.bbox_queue = "bbox_queue"
        self.ori_img_queue = "ori_img_queue"


    def send_next_layer(self, intermediate_queue, data, logger):
        if data != 'STOP':
            data["layers_output"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in data["layers_output"]]
            message = pickle.dumps({
                "action": "OUTPUT",
                "data": data
            })

            self.channel.basic_publish(
                exchange='',
                routing_key=intermediate_queue,
                body=message,
            )
        else:
            message = pickle.dumps(data)
            self.channel.basic_publish(
                exchange='',
                routing_key=intermediate_queue,
                body=message,
            )
    def send_to_tracker(self ,tracker_queue ,  predictions , frame_index , logger ):
        # prepare data
        try :
            if predictions is not 'STOP':
                if not isinstance(predictions , (list , tuple)) or len(predictions) == 0 or not isinstance(predictions[0] , torch.Tensor):
                    logger.log_warning(
                        f"Frame {frame_index}: Invalid prediction format received. Skipping send to tracker.")
                    return

                prediction_tensor = predictions[0]
                prediction_tensor_cpu = prediction_tensor.cpu()

                message_to_tracker = {
                    "predictions": prediction_tensor_cpu ,
                    "frame_index": frame_index
                }
            else :
                message_to_tracker = 'STOP'

            message_bytes = pickle.dumps(message_to_tracker)
            self.channel.basic_publish(
                exchange='',
                routing_key=tracker_queue,
                body=message_bytes
            )
        except Exception as e:
            logger.log_error(f"Frame {frame_index}: Failed to send data to tracker. Error: {e}")

    def send_ori_img(self , tracker_queue ,  frame_to_send  , frame_index , orig_img_size):
        try :
            if frame_to_send is not 'STOP':
                message = {
                    "ori_img" : frame_to_send ,
                    "frame_index" : frame_index,
                    "orig_img_size" : orig_img_size
                }
            else :
                message = 'STOP'

            message_bytes = pickle.dumps(message)
            self.channel.basic_publish(
                exchange= '',
                routing_key=tracker_queue ,
                body=message_bytes
            )
        except Exception as e :
            logger.log_error(f"Frame {frame_index}: Failed to send data to tracker. Error: {e}")

    def first_layer(self, model, data, save_layers, batch_frame, logger):
        time_inference = 0
        input_image = []
        predictor = SplitDetectionPredictor(model ,overrides={"imgsz": 640})
        frame_index = 0

        self.channel.queue_declare(queue=self.ori_img_queue, durable=False)
        self.channel.basic_qos(prefetch_count=50)

        model.eval()
        model.to(self.device)
        video_path = data
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.log_error(f"Not open video")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.log_info(f"FPS input: {fps}")
        path = None
        pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        while True:
            start = time.time()
            ret, frame = cap.read()

            # send origin frame

            if not ret or frame is None:
                y = 'STOP'
                self.send_next_layer(self.intermediate_queue, y, logger)
                self.send_ori_img(self.ori_img_queue , y , frame_index, (0 , 0))
                break

            # make border
            h , w, c = frame.shape
            size = max(h , w)
            orig_img_size = (h , w)
            if h > w:
                border_size = h - w
                frame= cv2.copyMakeBorder(frame, 0, 0, 0, border_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                border_size = w - h
                frame= cv2.copyMakeBorder(frame, 0, border_size, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            self.send_ori_img(self.ori_img_queue, frame, frame_index , orig_img_size)

            frame = cv2.resize(frame, (640 , 640 ))
            # print(f"[Frame][Shape] : {frame.shape}")
            tensor = torch.from_numpy(frame).float().permute(2, 0, 1)  # shape: (3, 640, 640)
            tensor /= 255.0
            input_image.append(tensor)
            # # debug
            # frame = cv2.resize(frame, (750, 750))
            # self.send_ori_img(self.ori_img_queue , frame , frame_index)


            if len(input_image) == batch_frame:
                input_image = torch.stack(input_image)
                input_image = input_image.to(self.device)
                # Prepare data
                predictor.setup_source(input_image)
                for predictor.batch in predictor.dataset:
                    path, input_image, _ = predictor.batch

                # Preprocess
                preprocess_image = predictor.preprocess(input_image)

                # Head predict
                y = model.forward_head(preprocess_image, save_layers)
                y["img_shape"] = preprocess_image.shape[2:]
                y["orig_img_shape"] = input_image.shape[2:]

                # if save_output:
                #     y["img"] = preprocess_image
                #     y["orig_imgs"] = input_image
                #     y["path"] = path
                time_inference += (time.time() - start)
                # print(f"[img_shape]{y["img_shape"].size()}")
                # print(f"[orig_img_shape] {y["orig_img_shape"].shape}")
                self.send_next_layer(self.intermediate_queue, y, logger)
                input_image = []
                pbar.update(batch_frame)
                frame_index += batch_frame
            else:
                continue

        cap.release()
        pbar.close()
        logger.log_info(f"End Inference.")
        return time_inference

    def last_layer(self, model, batch_frame, logger):
        time_inference = 0
        frame_index = 0
        predictor = BoundingBox()
        # predictor = SplitDetectionPredictor(model , overrides={"imgsz": 640} )

        model.eval()
        model.to(self.device)
        last_queue = f"intermediate_queue_{self.layer_id - 1}"
        self.channel.queue_declare(queue=last_queue, durable=False)
        self.channel.basic_qos(prefetch_count=50)

        self.channel.queue_declare(queue=self.bbox_queue, durable=False)
        self.channel.basic_qos(prefetch_count=50)

        pbar = tqdm(desc="Processing video (while loop)", unit="frame")

        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=last_queue, auto_ack=True)
            if method_frame and body:

                received_data = pickle.loads(body)
                if received_data != 'STOP':
                    y = received_data["data"]
                    y["layers_output"] = [t.to(self.device) if t is not None else None for t in y["layers_output"]]
                    start = time.time()
                    # Tail predict
                    predictions = model.forward_tail(y)
                    # print(f"[Prediction[0]][Shape]{predictions[0].shape}")


                    self.send_to_tracker(self.bbox_queue , predictions , frame_index , logger)

                    display = False
                    if display :
                        raw_prediction_tensor = predictions[0]

                        orig_imgs_list = [origin_frame_test]

                        tensor = torch.zeros(1024, 576)

                        results = predictor.postprocess(
                            preds=raw_prediction_tensor,
                            img_shape=tensor.shape,
                            orig_shape= tensor.shape ,
                            orig_imgs=orig_imgs_list
                        )

                        if results:
                            final_result = results[0]
                            annotated_image = final_result.plot()
                            cv2.imshow("Test Postprocess", annotated_image)
                            cv2.waitKey(int(1000/20))

                    time_inference += (time.time() - start)
                    frame_index += batch_frame
                    pbar.update(batch_frame)
                else:
                    cv2.destroyAllWindows()
                    self.send_to_tracker(self.bbox_queue , 'STOP' , frame_index , logger)
                    break
            else:
                continue
        pbar.close()
        logger.log_info(f"End Inference.")
        return time_inference

    def middle_layer(self, model):
        pass

    def inference_func(self, model, data, num_layers, save_layers, batch_frame, logger):
        time_inference = 0
        if self.layer_id == 1:
            time_inference = self.first_layer(model, data, save_layers, batch_frame, logger)
        elif self.layer_id == num_layers:
            time_inference = self.last_layer(model, batch_frame, logger)
        else:
            self.middle_layer(model)
        return time_inference