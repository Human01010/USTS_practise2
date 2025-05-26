import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt # Import plt once
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

# cudnn.benchmark = True
# plt.ion() # interactive mode - Keep this commented out for script mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Global variables for dataset and dataloaders (will be populated in __main__)
image_datasets = None
dataloaders = None
dataset_sizes = None
class_names = None
device = None # Device will be set in __main__

def imshow(ax, inp, title=None):
    """Display image for Tensor on a given axes."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp)
    if title is not None:
        ax.set_title(title)
    ax.axis('off') # Turn off axis for cleaner image display
    # Removed plt.pause(0.001)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    losses = []
    eval_losses = []

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        # Save initial model state (before training) - useful if training fails immediately
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]: # Use global dataloaders
                    inputs = inputs.to(device) # Use global device
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase] # Use global dataset_sizes
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    losses.append(epoch_loss)
                else:
                    eval_losses.append(epoch_loss)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    return model, losses, eval_losses

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0

    # Create a figure and a grid of axes for visualization
    fig, axes = plt.subplots(num_images // 2, 2, figsize=(8, 8))
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']): # Use global dataloaders
            inputs = inputs.to(device) # Use global device
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if images_so_far < num_images:
                    ax = axes[images_so_far]
                    ax.set_title(f'predicted: {class_names[preds[j]]}') # Use global class_names
                    imshow(ax, inputs.cpu().data[j]) # Pass the specific axis
                    images_so_far += 1
                else:
                    # Done visualizing
                    model.train(mode=was_training)
                    fig.tight_layout() # Adjust layout
                    return fig # Return the figure object

        # If dataset has fewer than num_images, return after iterating through all
        model.train(mode=was_training)
        fig.tight_layout() # Adjust layout
        return fig # Return the figure object

def visualize_model_predictions(model, img_path):
    was_training = model.training
    model.eval()

    # Create a figure and axis for this single prediction
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    try:
        img = Image.open(img_path)
        img = data_transforms['val'](img) # Use global data_transforms
        img = img.unsqueeze(0) # Add batch dimension
        img = img.to(device) # Use global device

        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)

            ax.set_title(f'Predicted: {class_names[preds[0]]}') # Use global class_names
            imshow(ax, img.cpu().data[0]) # Pass the specific axis

    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}")
        ax.set_title("Image Not Found")
        ax.text(0.5, 0.5, "File not found", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('on') # Keep axis on to show text

    model.train(mode=was_training)
    fig.tight_layout() # Adjust layout
    return fig # Return the figure object


if __name__ == '__main__':
    # Initialize global variables
    data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x]) # Use global data_transforms
                    for x in ['train', 'val']}
    # It's often recommended to set num_workers=0 or 1 for debugging,
    # or be mindful of platform differences (especially Windows) with multiprocessing.
    # Keeping num_workers=4 as per original, but be aware if issues persist.
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # --- Plotting the initial batch ---
    # Create a new figure for the initial batch visualization
    fig_initial_batch, ax_initial_batch = plt.subplots(1, 1, figsize=(8, 8))
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train'])) # Use global dataloaders
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    # Use the modified imshow that takes an axes object
    imshow(ax_initial_batch, out, title=[class_names[x] for x in classes]) # Use global class_names

    # --- Train the model ---
    # model_ft = models.resnet18(weights='IMAGENET1K_V1')
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 2)
    # model_ft = model_ft.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # model_ft, train_losses_ft, eval_losses_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

    model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv, train_losses, eval_losses = train_model(model_conv, criterion, optimizer_conv,
                            exp_lr_scheduler, num_epochs=25)

    # --- Plotting loss curves ---
    # Create a new figure for the loss plot
    fig_losses, ax_losses = plt.subplots(1, 1, figsize=(8, 6))
    ax_losses.plot(train_losses, label='train')
    ax_losses.plot(eval_losses, label='eval')
    ax_losses.legend()
    ax_losses.set_title('Losses')
    ax_losses.set_xlabel('Epochs')
    ax_losses.set_ylabel('Loss')

    # --- Visualizing model performance on validation set ---
    # visualize_model function now returns a figure
    fig_val_viz = visualize_model(model_conv)
    fig_val_viz.suptitle('Validation Predictions') # Add a super title

    # --- Visualizing a single prediction ---
    # visualize_model_predictions function now returns a figure
    fig_single_pred = visualize_model_predictions(
        model_conv,
        img_path='data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg'
    )
    fig_single_pred.suptitle('Single Image Prediction') # Add a super title


    # --- Show all created figures at once ---
    # Remove all intermediate plt.show() and plt.ioff() calls
    print("Displaying figures. Close plot windows to exit.")
    plt.show() # This single call displays all figures and blocks until they are closed.