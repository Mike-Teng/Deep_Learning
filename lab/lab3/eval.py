# import EEGNet
from EEGNet import EEGNet, evaluate
from dataloader import read_bci_data
from torch.utils.data import DataLoader, TensorDataset
import torch

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_data, train_label, test_data, test_label = read_bci_data()
    dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
    test_dataset = DataLoader(dataset, batch_size=64, shuffle=False)

    # ReLU model evaluation 
    model = EEGNet(torch.nn.ReLU())
    model.to(device)
    model.load_state_dict(torch.load("./weight/ReLU_model_egg_2.pt"))
    acc = evaluate(model, test_dataset, device)
    print(f"ReLU      EEG accuracy: {acc:.2f} %")
    
    # LeakyReLU model evaluation
    model = EEGNet(torch.nn.LeakyReLU())
    model.to(device)
    model.load_state_dict(torch.load("./weight/LeakyReLU_model_egg_2.pt"))
    acc = evaluate(model, test_dataset, device)
    print(f"LeakyReLU EEG accuracy: {acc:.2f} %")

    # ELU model evaluation
    model = EEGNet(torch.nn.ELU())
    model.to(device)
    model.load_state_dict(torch.load("./weight/ELU_model_egg_2.pt"))
    acc = evaluate(model, test_dataset, device)
    print(f"ELU       EEG accuracy: {acc:.2f} %")

    