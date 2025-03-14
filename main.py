##-- Import libraries --##
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_modules import data_setup, engine, utils
from tqdm.auto import tqdm
from timeit import default_timer as timer

##-- Setup device agnostic code --##
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device.index)  # Ensure the device is set correctly
print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {device}")


##-- Setting up hyperparamters
BATCH_SIZE = 32
RANDOM_STATE = 42
EPOCHS = 100
LEARNING_RATE = 1e-03

print(f"Learning Rate: {LEARNING_RATE}")

##-- Setup directory paths --##
from pathlib import Path
image_data_path = Path("chest_xray")
train_dir = image_data_path / "train"
val_dir = image_data_path / "val"


##-- Get a set of pretrained model weights --##
weights_b_16 =torchvision.models.ViT_B_16_Weights.DEFAULT # 'DEFAULT' = best availabel weight

##-- Get the transforms used to create the pretrained weights
auto_transforms = weights_b_16.transforms()

##-- Create DataLoaders using auto_transforms --##
train_dataloader, val_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                              val_dir=val_dir,
                                                                              transform=auto_transforms,
                                                                              batch_size=BATCH_SIZE)


weights_b_16 = torchvision.models.ViT_B_16_Weights.DEFAULT # "DEFAULT" = get the best available weights
model_b_16 = torchvision.models.vit_b_16(weights=weights_b_16).to(device)


##-- Freeze all the base layers in EffNetB7 --##
for params in model_b_16.features.parameters():
    params.requires_grad = False # won't updated the weights


##-- Update the classifier head of our model to suite our problem --##
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)

model_b_16.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=768, # features vector coming in from the forzen layers
              out_features=len(class_names)).to(device))

torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)

##-- Define Loss & Optimizer --##
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_b_16.parameters(), lr=LEARNING_RATE)

start_time = timer()

##-- Setup training and save the results --##
model_b_16_results = engine.train(model=model_b_16,
                               train_dataloader=train_dataloader,
                               val_dataloader=val_dataloader,
                               optimizer=optimizer,
                               loss_fn=loss_fn,
                               epochs=EPOCHS,
                               device=device,
                               log_file=f"./log_files/b_16_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}.txt")
end_time = timer()

##- Save the model --##
from pytorch_modules import utils
utils.save_model(model=model_b_16,
                 target_dir="./saved_models",
                 model_name=f"b_16_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}.pth")

print(f"[INFO]: Total training time: {(end_time-start_time)/60:.2f} minutes.")


##-- Plot loss curves --##
epochs = [i for i in range(1, EPOCHS+1)]
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, model_b_16_results["train_loss"], color="blue", label="train_loss")
plt.plot(epochs, model_b_16_results["val_loss"], color="red", label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, model_b_16_results["train_acc"], color="blue", label="train_acc")
plt.plot(epochs, model_b_16_results["val_acc"], color="green", label="val_acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.suptitle("ViT_B_16 Results")
plt.savefig(f"./plots/b_16_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}.png")
plt.show()

