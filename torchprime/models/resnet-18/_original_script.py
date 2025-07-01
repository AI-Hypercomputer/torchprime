# import argparse
# import glob
# import logging
# import multiprocessing
# import os
# import sys
# from datetime import datetime

# if "PJRT_DEVICE" in os.environ and os.environ["PJRT_DEVICE"] == "TPU":
#     # Enable bfloat16 conversion with TPUs
#     os.environ["XLA_USE_BF16"] = "1"
#     os.environ["TPU_SKIP_MDS_QUERY"] = "True"
#     os.environ["TPU_ACCELERATOR_TYPE"] = "v5e-8"
#     os.environ["ACCELERATOR_TYPE"] = "v5e-8"
#     os.environ["TPU_WORKER_ID"] = "0"
#     os.environ["TPU_WORKER_HOSTNAMES"] = "localhost"
# else:
#     # Enable cuBLAS deterministic mode
#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
# import random
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Subset, Dataset, DataLoader, default_collate, random_split

# from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall

# from torchvision.models import resnet18
# from torchvision.datasets import ImageFolder
# from torchvision.transforms import v2 as transforms
# from PIL import Image

# if "PJRT_DEVICE" in os.environ and os.environ["PJRT_DEVICE"] == "TPU":
#     import torch_xla as xla
#     import torch_xla.core.xla_model as xm
    
# from datasets import load_dataset, load_from_disk

# from google.cloud import aiplatform
# import hypertune

# # Inspect available CPUs
# CPUS = multiprocessing.cpu_count()
# logging.info(f"Found {CPUS} CPU cores.")

# # Init Hyperparameter search reporting
# hpt = hypertune.HyperTune()    


# class FID_Open_Dataset(Dataset):
#     """
#     Custom PyTorch Dataset that returns:
#         - Image: as a Tensor
#         - Label: as an Integer
#         - Metadata: a dictionary containing Participant ID and File ID
#     """

#     def __init__(self, dataset_paths, transform=None):
#         """
#         Initialize dataset.

#         Args:
#             dataset_path (str): Path to the dataset directory.
#             transform (callable, optional): Transformations to apply to the images.
#         """
#         # Get all image file paths
#         self.image_paths = dataset_paths
        
#         if len(self.image_paths) == 0:
#             raise RuntimeError(f"No images found in {dataset_paths}")

#         logging.info("Dataset found. Creating Torch Dataset...")

#         # Extract class names from folder structure
#         class_names = list(set([os.path.basename(os.path.dirname(path)) for path in self.image_paths]))
#         class_names.sort()  # Ensure consistent label ordering
        
#         # Create a mapping from class name to label index
#         self.label_dict = {class_name: i for i, class_name in enumerate(class_names)}
        
#         # Define transformations
#         self.transform = transform if transform else transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.Resize((224, 224)),  # Resize images to a fixed size
#             transforms.ToTensor(),         # Convert images to tensors
#         ])
#         self.df=pd.DataFrame(self.image_paths,columns=['image_path'])
#         self.df['Participant_ID']=self.df['image_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
#         self.df['file_info_id']=self.df['image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
#         self.df['label']=self.df['Participant_ID'].apply(lambda x: self.label_dict[x])
#         logging.info("Torch Dataset created successfully.")
        

#     def __len__(self):
#         """Return the total number of images."""
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         """
#         Get an image, label, and metadata.

#         Args:
#             idx (int): Index of the image.

#         Returns:
#             tuple: (image, label, metadata)
#         """
#         image_path = self.image_paths[idx]
#         class_name = os.path.basename(os.path.dirname(image_path))  # Extract class name
#         label = self.label_dict[class_name]  # Convert class name to integer label

#         # Load and transform image
#         image = Image.open(image_path).convert("RGB")
#         image = self.transform(image)

#         # Extract metadata (Participant ID and File ID)
#         file_name = os.path.basename(image_path)  # Extract filename
#         participant_id = class_name  # Assuming class folder represents participant ID
#         file_id = os.path.splitext(file_name)[0]  # Remove extension for File ID

#         metadata = {
#             "Participant_ID": participant_id,
#             "file_info_id": file_id
#         }

#         return image, label, metadata

# class EarlyStopper:
#     def __init__(self, patience=1, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.min_loss = float("inf")

#     def early_stop(self, loss):
#         if loss < self.min_loss:
#             self.min_loss = loss
#             self.counter = 0
#         elif loss > (self.min_loss + self.min_delta):
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False

# def _init(params):
#     logging.info("Setting seeds...")
#     # Set a seed
#     random.seed(params.seed)
#     np.random.seed(params.seed)
#     torch.manual_seed(params.seed)
#     #torch.use_deterministic_algorithms(True)
#     #torch.set_float32_matmul_precision("medium")
#     if "xm" in sys.modules:
#         xm.set_rng_state(params.seed)

# def _report_metric(tag, value, global_step):
#     aiplatform.log_time_series_metrics({tag: value}, step=global_step)
#     hpt.report_hyperparameter_tuning_metric(
#         hyperparameter_metric_tag=tag,
#         metric_value=value,
#         global_step=global_step)

# def _split_paths(dataset_path):
#     if not os.path.exists(dataset_path):
#         raise FileNotFoundError(f"Directory not found: {dataset_path}")
    
#     paths = glob.glob(os.path.join(dataset_path, "*", "*"))
#     logging.info(f"Detected {len(paths)} paths (e.g. {paths[0]}).")
    
#     train_fraction = round(len(paths) * 0.8)
#     train_paths, valid_paths = paths[:train_fraction], paths[train_fraction:]
    
#     return train_paths, valid_paths

# def _seed_worker(worker_id):
#     import numpy as np
#     import random
#     import torch
#     worker_seed = torch.initial_seed() % 2**32 + worker_id
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
#     torch.manual_seed(worker_seed)

# def _prepare_dataloader(dataset, params):
#     logging.info("Preparing dataset loader...")
#     return DataLoader(dataset, 
#                       batch_size=params.batch_size,
#                       num_workers=0, #4,
#                       shuffle=True,
#                       #prefetch_factor=2,
#                       #worker_init_fn=_seed_worker,
#                       #generator=torch.Generator(),
#                       #multiprocessing_context="fork",
#                       drop_last=True,
#                       #persistent_workers=False
#                      )

# def _train(params):
#     if params.dataset == "hf-celeba":
#         logging.info("Loading HF datasets from GCS...")
#         train_dataset = load_from_disk("gs://sandbox-michael-menzel-training-us-central1/trainings/mercedes-cmcs/data-cache/train/")
#         test_dataset = load_from_disk("gs://sandbox-michael-menzel-training-us-central1/trainings/mercedes-cmcs/data-cache/test/")
        
#         CLASSES = list(set(train_dataset.unique('celeb_id')) | set(test_dataset.unique('celeb_id')))
#         NUM_CLASSES = len(CLASSES)
#         NUM_CHANNELS = 3
#     elif params.dataset == "mb-custom":
#         data_paths = ["/gcs/sandbox-michael-menzel-data-us-central1/FaceID-550",
#                       "/gcs/sandbox-michael-menzel-data-us-central1/vggface"]
#         train_paths=[]
#         valid_paths=[]
#         for path in data_paths:
#             train_paths_split, valid_paths_split = _split_paths(path)
#             train_paths.extend(train_paths_split)
#             valid_paths.extend(valid_paths_split)

#         train_dataset = FID_Open_Dataset(train_paths)
#         test_dataset = FID_Open_Dataset(valid_paths)
        
#         NUM_CLASSES = len(list(set(train_dataset.label_dict.keys()) | set(test_dataset.label_dict.keys())))
#         NUM_CHANNELS = 1
#     elif params.dataset == "mb-torchvision":
#         data_path = "/gcs/sandbox-michael-menzel-data-us-central1/FaceID-550+vggface"

#         img_transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.Resize((224, 224)),
#             transforms.ToImage(), 
#             transforms.ToDtype(torch.float32, scale=True)
#         ])
        
#         folder_dataset = ImageFolder(root=data_path, transform=img_transform)
#         train_idx, test_idx = train_test_split(np.arange(len(folder_dataset.targets)),
#                                                test_size=0.2,
#                                                shuffle=True,
#                                                stratify=folder_dataset.targets)
#         train_dataset, test_dataset = Subset(folder_dataset, train_idx), Subset(folder_dataset, test_idx)
        
#         NUM_CLASSES = len(set(folder_dataset.targets))
#         NUM_CHANNELS = 1
#     else:
#         logging.error("No dataset specified!")
#         raise RuntimeError("No valid dataset parameter provided.")

#     train_loader = _prepare_dataloader(train_dataset, params)
#     test_loader = _prepare_dataloader(test_dataset, params)

#     aiplatform.log_params({"dataset_train_num_example": len(train_dataset), 
#                            "dataset_test_num_example": len(test_dataset)})
    
#     logging.info("Downloading pretrained model...")
#     # Load a pretrained ResNet18 model
#     model = resnet18(pretrained=True)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
#     model.conv1 = nn.Conv2d(NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     model.bn1 = nn.BatchNorm2d(64)
    
#     aiplatform.log_params({"model_num_classes": NUM_CLASSES, 
#                            "model_num_channels": NUM_CHANNELS})

#     # Set XLA device
#     logging.info(f"Available TPU devices: {xm.get_xla_supported_devices('TPU')}")
#     device = torch.device("cuda" if torch.cuda.is_available() else (xm.xla_device() if "torch_xla" in sys.modules.keys() else "cpu"))
#     logging.info(f"Using device {device}.")
    
#     logging.info("Moving model to device...")
#     # Move the model to the GPU/TPU if available
#     model.to(device)

#     # Define optimizer & loss function
#     loss_fct = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=params.learning_rate)
#     metrics = {
#         "accuracy_top1": MulticlassAccuracy(num_classes=NUM_CLASSES, k=1),
#         "accuracy_top5": MulticlassAccuracy(num_classes=NUM_CLASSES, k=5),
#         "precision": MulticlassPrecision(num_classes=NUM_CLASSES),
#         "recall": MulticlassRecall(num_classes=NUM_CLASSES)
#     }
    
#     global_step = 0
#     early_stopper = EarlyStopper(patience=20)
    
#     logging.info("Starting training loop...")
#     model.train()
#     for epoch in range(params.num_epochs):
#         epoch_losses = []
#         epoch_accuracies = []
#         for batch_idx, batch in enumerate(train_loader):
#             optimizer.zero_grad()

#             data, target = (batch[0], batch[1])
#             data, target = (data.to(device), target.to(device))

#             output = model(data)
#             loss = loss_fct(output, target)
#             if torch.isnan(loss).item():
#                 break
            
#             loss.backward()
#             optimizer.step()
            
#             if device.type == "xla":
#                 xm.mark_step()
          
#             loss_val = loss.cpu().item()
#             epoch_losses.append(loss_val)
#             _report_metric("step_loss", loss_val, global_step)
            
#             for name, metric in metrics.items():
#                 metric.update(output, target)
#                 _report_metric("step_" + name, metric.compute(), global_step)
#             global_step += 1
         
#         epoch_loss = np.mean(epoch_losses)
        
#         _report_metric("epoch_loss", epoch_loss, epoch)
#         for name, metric in metrics.items():
#             _report_metric("epoch_" + name, metric.compute(), epoch)
#             metric.reset()
 
#         if early_stopper.early_stop(epoch_loss):
#             break

#     logging.info("Training done.")

#     logging.info("Starting evaluation loop...")
#     model.eval()  
#     test_losses = []
#     test_accuracies = []
#     for batch_idx, batch in enumerate(test_loader):
#         data, target = (batch["image"], batch["label"])
#         data, target = (data.to(device), target.to(device))

#         output = model(data)
#         loss = loss_fct(output, target)
#         if torch.isnan(loss).item():
#             break

#         if device.type == "xla":
#             xm.mark_step()

#         loss_val = loss.cpu().item()
#         test_losses.append(loss_val)
#         accuracy_val = (torch.sum(torch.argmax(output, dim=-1) == target) / data.size(dim=0)).cpu().item()
#         test_accuracies.append(accuracy_val)

#         _report_metric("test_step_loss", loss_val, batch_idx)
#         _report_metric("test_step_accuracy", accuracy_val, batch_idx)
        
#     test_loss = np.mean(test_losses)
#     test_accuracy = np.mean(test_accuracies)
#     _report_metric("test_loss", epoch_loss, epoch)
#     _report_metric("test_accuracy", test_accuracy, epoch)
    
#     logging.info("Evaluation done.")

# def _get_args():
#     """Argument parser.
#     Returns:
#     Dictionary of arguments.
#     """
#     cloud_ml_job_id = os.environ["CLOUD_ML_JOB_ID"]
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--experiment",
#         type=str,
#         default=f"experiment-{cloud_ml_job_id}",
#         help="experiment name to log metrics and checkpoints, " +
#              "default=experiment-[CLOUD_ML_JOB_ID]")
#     parser.add_argument(
#         "--dataset",
#         type=str,
#         default=f"hf-celeba",
#         help="dataset to use, e.g. hf-celeba, mb-custom, mb-torchvision")
#     parser.add_argument(
#         "--num-epochs",
#         type=int,
#         default=200,
#         help="number of times to go through the data, default=5")
#     parser.add_argument(
#         "--batch-size",
#         default=128,
#         type=int,
#         help="number of records to read during each training step, default=128")
#     parser.add_argument(
#         "--learning-rate",
#         default=.01,
#         type=float,
#         help="learning rate for optimizer, default=.01")
#     parser.add_argument(
#         "--seed",
#         default=42,
#         type=int,
#         help="seed to initialize RNGs in the training program, default=42")
#     parser.add_argument(
#         "--verbosity",
#         choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
#         default="DEBUG")
#     return parser.parse_args()

# if __name__ == "__main__":   
#     params = _get_args()
    
#     if params:
#         logging.basicConfig(level=logging.getLevelName(params.verbosity))
        
#         aiplatform.init(project=os.environ["CLOUD_ML_PROJECT_ID"],
#                         location=os.environ["CLOUD_ML_REGION"],
#                         experiment=params.experiment)
        
#         datetime_now_fmt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
#         device_type_str = "cuda" if torch.cuda.is_available() else ("xla" if "torch_xla" in sys.modules.keys() else "cpu")
#         aiplatform.start_run(
#             run=f"resnet18-trainer-{device_type_str}-{datetime_now_fmt}")
#         aiplatform.log_params(vars(params))
                
#         try:
#             _init(params)
#             _train(params)
#         except Exception as e:
#             logging.error(f"Could not complete training job. Error: {e}")
#             print(e)
#         except subprocess.CalledProcessError as e:
#             logging.error(f"Could not complete training job due to a subprocess. Error: {e}")
#             print(e.output)
#         finally:
#             aiplatform.end_run()
#     else:
#         logging.error("Could not parse parameters and configuration.")
