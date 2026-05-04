import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report



# 1. Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# 2. Load MNIST dataset
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))



# 3. Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # 28x28 -> 26x26
        x = F.relu(self.conv2(x))       # 26x26 -> 24x24
        x = F.max_pool2d(x, 2)          # 24x24 -> 12x12
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN().to(device)



# 4. Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3

print("\nTraining started...")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

print("Training finished.")



# 5. Evaluate clean accuracy
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1)

            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total


clean_accuracy = evaluate(model, test_loader)
print("\nClean accuracy:", clean_accuracy)



# 6. C&W-like L2 attack
def cw_l2_attack(model, images, labels, c=1.0, kappa=0, steps=100, lr=0.01):
    """
    Simplified C&W L2 attack.
    Untargeted adversarial attack.

    Objective:
    minimise L2 perturbation + c * misclassification loss
    """

    model.eval()

    images = images.to(device)
    labels = labels.to(device)

    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    optimizer = optim.Adam([adv_images], lr=lr)

    for step in range(steps):
        outputs = model(adv_images)

        one_hot_labels = F.one_hot(labels, num_classes=10).float().to(device)

        real = torch.sum(one_hot_labels * outputs, dim=1)
        other = torch.max(
            (1 - one_hot_labels) * outputs - one_hot_labels * 10000,
            dim=1
        )[0]

        # Untargeted C&W-style loss:
        # reduce the confidence of the true class until another class becomes stronger
        f_loss = torch.clamp(real - other + kappa, min=0)

        l2_loss = torch.sum((adv_images - images) ** 2, dim=[1, 2, 3])

        loss = torch.mean(l2_loss + c * f_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        adv_images.data = torch.clamp(adv_images.data, 0, 1)

    return adv_images.detach()



# 7. Run C&W attack
# C&W is slow on CPU.
# Use 200 first. Increase to 500 later if needed.
attack_sample_size = 200

test_subset = Subset(test_dataset, range(attack_sample_size))
attack_loader = DataLoader(test_subset, batch_size=50, shuffle=False)

total = 0
clean_correct = 0
adv_correct = 0
successful_attacks = 0
total_l2 = 0

all_original = []
all_adversarial = []
all_labels = []
all_clean_preds = []
all_adv_preds = []

print("\nRunning C&W attack...")

for batch_idx, (data, target) in enumerate(attack_loader):
    data, target = data.to(device), target.to(device)

    clean_output = model(data)
    clean_pred = clean_output.argmax(dim=1)

    adv_data = cw_l2_attack(
        model=model,
        images=data,
        labels=target,
        c=1.0,
        kappa=0,
        steps=100,
        lr=0.01
    )

    adv_output = model(adv_data)
    adv_pred = adv_output.argmax(dim=1)

    clean_correct += (clean_pred == target).sum().item()
    adv_correct += (adv_pred == target).sum().item()

    successful = (clean_pred == target) & (adv_pred != target)
    successful_attacks += successful.sum().item()

    l2 = torch.sqrt(torch.sum((adv_data - data) ** 2, dim=[1, 2, 3]))
    total_l2 += l2.sum().item()

    total += target.size(0)

    all_original.append(data.cpu())
    all_adversarial.append(adv_data.cpu())
    all_labels.append(target.cpu())
    all_clean_preds.append(clean_pred.cpu())
    all_adv_preds.append(adv_pred.cpu())

    print(f"Batch {batch_idx + 1} done")


clean_acc_subset = clean_correct / total
adv_acc = adv_correct / total
attack_success_rate = successful_attacks / clean_correct if clean_correct > 0 else 0
avg_l2 = total_l2 / total

print("\n========== C&W Attack Results ==========")
print("Samples attacked:", total)
print("Clean accuracy on attacked subset:", clean_acc_subset)
print("Adversarial accuracy:", adv_acc)
print("Attack success rate:", attack_success_rate)
print("Average L2 perturbation:", avg_l2)



# 8. Show original vs adversarial examples
original_images = torch.cat(all_original)
adversarial_images = torch.cat(all_adversarial)
labels = torch.cat(all_labels)
clean_preds = torch.cat(all_clean_preds)
adv_preds = torch.cat(all_adv_preds)

num_examples = 10

plt.figure(figsize=(12, 4))

for i in range(num_examples):
    plt.subplot(2, num_examples, i + 1)
    plt.imshow(original_images[i].squeeze(), cmap="gray")
    plt.title(f"Orig\nT:{labels[i].item()}\nP:{clean_preds[i].item()}")
    plt.axis("off")

    plt.subplot(2, num_examples, i + 1 + num_examples)
    plt.imshow(adversarial_images[i].squeeze(), cmap="gray")
    plt.title(f"Adv\nT:{labels[i].item()}\nP:{adv_preds[i].item()}")
    plt.axis("off")

plt.tight_layout()
plt.show()



# 9. Classification report and confusion matrix
y_true = labels.numpy()
y_pred_adv = adv_preds.numpy()

print("\nClassification Report - C&W Adversarial Examples:")
print(classification_report(y_true, y_pred_adv, zero_division=0))

cm = confusion_matrix(y_true, y_pred_adv)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - C&W Adversarial Predictions")
plt.show()