import pika
from requests.auth import HTTPBasicAuth
import requests
import pandas as pd


def delete_old_queues(address, username, password, virtual_host):
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "result") or queue_name.startswith("rpc_queue"):

                http_channel.queue_delete(queue=queue_name)

            else:
                http_channel.queue_purge(queue=queue_name)

        connection.close()
        return True
    else:
        return False

def write_to_csv(partial_data):

    cols = ["FPS_Input" , "" , "num Frame" , "" ,
            "[1] All time", "" , "[1] Inference time" , "" , "[1] Utilization" , "" ,
            "[2] All time", "" , "[2] Inference time" , "" , "[2] Utilization" , "" ,
            "display time", "" , "non-display time"
            ]
    """
    df = pd.DataFrame(columns=cols)
    df.to_csv('output.csv', index=False)
    """
    file_path = 'output.csv'
    partial_df = pd.DataFrame([partial_data])

    # ensure all columns are present
    for col in cols :
        if col not in partial_df.columns:
            partial_df[col] =''

    # reoder
    partial_df = partial_df[expected_columns]

    if not os.path.exists(file_path):
        partial_df.to_csv(file_path, index=False)
    else:
        partial_df.to_csv(file_path, mode='a', index=False, header=False)


