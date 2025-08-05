import pika
import pickle
import yaml
import torch

def main():
    try:
        # Đọc cấu hình RabbitMQ từ config.yaml
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: 'config.yaml' not found. Please run this script from the project root.")
        return

    rabbit_config = config.get("rabbit", {})
    credentials = pika.PlainCredentials(rabbit_config.get("username"), rabbit_config.get("password"))
    params = pika.ConnectionParameters(
        host=rabbit_config.get("address"),
        virtual_host=rabbit_config.get("virtual-host"),
        credentials=credentials
    )

    try:
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
    except pika.exceptions.AMQPConnectionError as e:
        print(f"Error connecting to RabbitMQ: {e}")
        print("Please ensure RabbitMQ is running and 'config.yaml' is correct.")
        return

    tracker_queue = "tracker"
    channel.queue_declare(queue=tracker_queue, durable=False)

    print(f"[Tracker] Waiting for messages on queue '{tracker_queue}'. Press CTRL+C to exit.")

    message_count = 0
    for method_frame, properties, body in channel.consume(tracker_queue):
        if body:
            message_obj = pickle.loads(body)

            # Acknowledge the message was received and processed
            channel.basic_ack(method_frame.delivery_tag)

            if message_obj == "STOP":
                print("\n[Tracker] Received STOP signal. Shutting down.")
                break

            message_count += 1
            print(f"\n---------- Message #{message_count} Received ----------")

            if isinstance(message_obj, dict):
                # Lấy ra các key
                frame_index = message_obj.get("frame_index")
                predictions = message_obj.get("predictions")

                print(f"  - Frame Index: {frame_index}")

                # Kiểm tra predictions
                if predictions is not None:
                    print(f"  - Type of 'predictions': {type(predictions)}")
                    if isinstance(predictions, torch.Tensor):
                        print(f"  - Shape of 'predictions': {predictions.shape}")
                        print(f"  - Device of 'predictions': {predictions.device}")
                    else:
                        print(f"  - 'predictions' is NOT a Tensor.")
                else:
                    print("  - 'predictions' key not found or is None.")
            else:
                print(f"  - Received message of unexpected type: {type(message_obj)}")

            print("----------------------------------------")

    print("\n[Tracker] Cleaning up...")
    connection.close()
    print("[Tracker] Connection closed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Tracker] Interrupted by user. Exiting.")