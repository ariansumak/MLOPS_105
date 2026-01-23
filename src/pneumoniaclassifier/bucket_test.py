from google.cloud import storage

def upload_file(bucket_name, source_file_path, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_path)

    print(f"Uploaded {source_file_path} to gs://{bucket_name}/{destination_blob_name}")

if __name__ == "__main__":
    upload_file(
        bucket_name="pneumonia-classifier-drift",
        source_file_path="./data/chest_xray/test/NORMAL/IM-0001-0001.jpeg",
        destination_blob_name="from_script/IM-0001-0001.jpeg",
    )
