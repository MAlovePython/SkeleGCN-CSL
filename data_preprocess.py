import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def extract_keyframes(data, num_keyframes):
    total_frames = data.shape[0]
    if total_frames < num_keyframes:
        # 如果帧数不足，通过重复来填充
        repeats = num_keyframes // total_frames + 1
        data = np.tile(data, (repeats, 1))[:num_keyframes]
    elif total_frames > num_keyframes:
        # 如果帧数过多，均匀采样
        indices = np.linspace(0, total_frames - 1, num_keyframes, dtype=int)
        data = data[indices]
    return data


def abs2rel(coords, crop_size):
    return coords / crop_size


def preprocess_data(raw_data_path, processed_data_path, num_keyframes=36, crop_size=256):
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    data = []
    labels = []

    # 指定需要的关节点索引
    needed_joints = [3, 2, 20, 1, 0, 4, 5, 6, 7, 21, 22, 8, 9, 10, 11, 23, 24]

    class_folders = sorted([f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))])
    for class_idx, class_folder in enumerate(tqdm(class_folders, desc="Processing classes")):
        class_path = os.path.join(raw_data_path, class_folder)

        for sample_file in os.listdir(class_path):
            if sample_file.endswith('.txt'):
                sample_path = os.path.join(class_path, sample_file)
                try:
                    # 读取txt文件
                    with open(sample_path, 'r') as f:
                        lines = f.readlines()

                    # 解析数据
                    sample_data = []
                    for line in lines:
                        values = list(map(float, line.strip().split()))
                        sample_data.append(values)

                    sample_data = np.array(sample_data)

                    # 提取关键帧
                    keyframes = extract_keyframes(sample_data, num_keyframes)

                    # 只保留指定的关节点数据
                    keyframes = keyframes[:, needed_joints]

                    # 将绝对坐标转换为相对坐标
                    keyframes = abs2rel(keyframes, crop_size)

                    # 重塑数据为 (关键帧数, 关节点数*2)
                    keyframes = keyframes.reshape(num_keyframes, -1)

                    data.append(keyframes)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing file {sample_file}: {str(e)}")
                    continue  # 跳过这个文件，继续处理下一个

    data = np.array(data)
    labels = np.array(labels)

    np.save(os.path.join(processed_data_path, 'csl500_data.npy'), data)
    np.save(os.path.join(processed_data_path, 'csl500_labels.npy'), labels)

    print(f"Processed data shape: {data.shape}")
    print(f"Processed labels shape: {labels.shape}")


def validate_preprocessed_data(processed_data_path):
    data = np.load(os.path.join(processed_data_path, 'csl500_data.npy'))
    labels = np.load(os.path.join(processed_data_path, 'csl500_labels.npy'))

    print(f"Total samples: {len(data)}")
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")

    # 打印每个类别的样本数量
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nSamples per class:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} samples")

    # 可视化一些随机样本
    num_samples = 5
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 5 * num_samples))
    for i in range(num_samples):
        idx = np.random.randint(0, len(data))
        sample = data[idx]
        label = labels[idx]

        print(f"\nSample {i}:")
        print(f"  Shape: {sample.shape}")
        print(f"  Label: {label}")
        print(f"  Data snippet:\n{sample[:2, :4]}")  # 只显示前两帧的前四个值

        # 可视化样本的第一帧
        ax = axes[i]
        im = ax.imshow(sample.T, aspect='auto', cmap='viridis')
        ax.set_title(f"Sample {i}, Label: {label}")
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Features")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(processed_data_path, 'sample_visualization.png'))
    print(f"Sample visualization saved to {os.path.join(processed_data_path, 'sample_visualization.png')}")


if __name__ == "__main__":
    raw_data_path = 'data/CSL-500/raw/xf500_body_depth_txt'  # 修改为txt文件的路径
    processed_data_path = 'data/CSL-500/processed'
    preprocess_data(raw_data_path, processed_data_path)
    validate_preprocessed_data(processed_data_path)