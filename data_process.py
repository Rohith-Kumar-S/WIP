
import cmd
import subprocess
from tqdm import tqdm
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
import argparse
import os 
import sys
sys.path.append(os.path.abspath(".."))

from interaction_predicition.data_process import DataProcess



class DataProcessv1(DataProcess):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def merge_sensors_with_scenario(self, shard_dataset, shard_id):
        os.makedirs("/content/data/lidar_and_camera", exist_ok=True)
        os.makedirs(self.args.save_path, exist_ok=True)
        output_path = f"{self.args.save_path}/merged_shard-{shard_id}.tfrecord"

        writer = tf.io.TFRecordWriter(output_path)
        scenario_ids = []
        for data in shard_dataset:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(data.numpy())
            scenario_ids.append(scenario.scenario_id)
            
        # Build download paths for all scenarios in the shard
        paths = [
            f"gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/lidar_and_camera/training/{sid}.tfrecord"
            for sid in scenario_ids
        ]
        
        # Write paths to a temporary file for batch downloading
        with open("paths.txt", "w") as f:
            for p in paths:
                f.write(p + "\n")
        
        # Use gsutil to download all lidar and camera data for the scenarios in the shard
        subprocess.run([
            "gsutil", "-m", "cp", "-I", "/content/data/lidar_and_camera/"
        ], stdin=open("paths.txt", "r"))
        
        pbar = tqdm(shard_dataset, desc="Merging sensor data to scenarios", unit="file")

        for data in pbar:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(data.numpy())
            scenario_id = scenario.scenario_id
            pbar.set_postfix(id=scenario_id[:8])
            lidar_cam_path = f"/content/data/lidar_and_camera/{scenario_id}.tfrecord"
            if os.path.exists(lidar_cam_path):
                lidar_dataset = tf.data.TFRecordDataset(lidar_cam_path)
                lidar_data = next(iter(lidar_dataset))
                lidar_cam = scenario_pb2.Scenario()
                lidar_cam.ParseFromString(lidar_data.numpy())
                scenario.compressed_frame_laser_data.MergeFrom(lidar_cam.compressed_frame_laser_data)
                scenario.frame_camera_tokens.MergeFrom(lidar_cam.frame_camera_tokens)
                writer.write(scenario.SerializeToString())
        writer.close()
        print('Merged sensor data with shards')
  
  
def main():
    parser = argparse.ArgumentParser(description='Data Processing Interaction Predictions')
    parser.add_argument('--shards_path', type=str, help='path to save processed data')
    parser.add_argument('--save_path', type=str, help='path to save processed data')
    parser.add_argument('--load_all_shards', type=bool, help='whether to load all shards or just one')
    args = parser.parse_args()
    
    if not args.load_all_shards:
        cmd = [
            "gsutil", "-m", "cp",
            "gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training/training.tfrecord-00000-of-01000",
            args.shards_path
        ]
        subprocess.run(cmd, check=True)
        filenames = tf.io.matching_files(args.shards_path)
        train_dataset = tf.data.TFRecordDataset(filenames, compression_type='')
    
        processor = DataProcessv1(args)
        processor.merge_sensors_with_scenario(train_dataset, "00000-of-01000")
    
  
if __name__ == "__main__":
    main()