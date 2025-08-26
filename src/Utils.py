import os
import pika
from requests.auth import HTTPBasicAuth
import requests
import pandas as pd
import csv


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

""" write to csv file """

cols = [
        "[T]totalTm","[T]totalFr","[T]Frme1st"
        #"[1]totalTm", "[1]unitiTm",
        #"[2]totalTm", "[2]unitiTm",
        ]


file_path = "output.csv"

row_buffer = {}

def write_partial(partial_data, flush=False):
    global row_buffer, cols

    # don't run if not exist csv file
    update_csv_header(file_path ,cols)

    # update buffer with new data
    row_buffer.update(partial_data)

    new_cols = [c for c in partial_data.keys() if c not in cols]
    if new_cols:
        cols.extend(new_cols)  # add new columns
        # reload CSV with new header
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for c in new_cols:
                df[c] = ""  # add empty column for past rows
            df.to_csv(file_path, index=False)

    if all(col in row_buffer for col in cols):
        flush = True

    if flush:
        # build dataframe with all current columns
        row_df = pd.DataFrame([row_buffer], columns=cols)

        if not os.path.exists(file_path):
            row_df.to_csv(file_path, index=False)
        else:
            row_df.to_csv(file_path, mode='a', index=False, header=False)

        row_buffer = {}
        print("[CSV] write csv successfully !")

def update_csv_header(filename, new_headers):
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    if not reader:
        raise ValueError("CSV file is empty!")

    # Replace only the first row
    reader[0] = new_headers

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(reader)
