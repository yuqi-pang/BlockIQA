import time
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from model import BaseModel, ImageQualityAggregator, CombinedModel
from folders import IQADataset
from function import train

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "../IQA-Dataset/data"
    metadata_files = {
        # "CSIQ": ("../IQA-Dataset/data/CSIQ/CSIQ.txt", (0, 1)),
        # "LIVE": ("../IQA-Dataset/data/LIVE/LIVE.txt", (0, 100)),
        # "TID2013": ("../IQA-Dataset/data/TID2013/TID2013.txt", (0, 9)),
        # "KADID-10k": ("../IQA-Dataset/data/KADID-10k/KADID-10k.txt", (1, 5)),
        # "KonIQ-10k": ("../IQA-Dataset/data/KonIQ-10k/KonIQ-10k.txt", (1, 5)),
        # "CID2013": ("../IQA-Dataset/data/CID2013/CID2013.txt",(0,100)),
        "LIVE_Challenge": ("../IQA-Dataset/data/LIVE_Challenge/LIVE_Challenge.txt", (0, 100)),
        # "SPAQ": ("../IQA-Dataset/data/SPAQ/SPAQ.txt", (0, 100))
    }

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    n_blocks = 2
    batch_size = 16

    basemodel = BaseModel()
    image_quality_aggregator = ImageQualityAggregator(n_blocks=n_blocks)
    model = CombinedModel(basemodel, image_quality_aggregator)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # 针对每个数据集进行训练和评估
    for dataset_name, (metadata_file, score_range) in metadata_files.items():
        start_time = time.time()

        print(f"{dataset_name} dataset is training...")

        dataset = IQADataset(data_dir, metadata_file, transform=transform_train, score_range=score_range)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # 为测试数据集应用测试变换
        test_dataset.transform = transform_test

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        best_srcc, best_plcc = train(model, train_loader, test_loader, criterion, optimizer, n_blocks,
                                     min_epochs=10)
        print(f"Training of {dataset_name} dataset finished.")
        print(f"代码运行时间为: {time.time() - start_time} 秒")
