import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

def train(dataset, model, device):
    data_induce = np.arange(0, dataset.__len__())
    kf = KFold(n_splits=10)
    # here we define some Hyperparameters
    theta = 0.80
    criterion_a = torch.nn.CrossEntropyLoss()
    criterion_b = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    final_loss_train = 0
    final_acc_a_train = 0
    final_acc_b_train = 0
    final_loss_val = 0
    final_acc_a_val = 0
    final_acc_b_val = 0

    i = 0
    for train_index, val_index in kf.split(data_induce):
        train_db = torch.utils.data.dataset.Subset(dataset, train_index)
        val_db = torch.utils.data.dataset.Subset(dataset, val_index)
        train_dataloader = DataLoader(train_db, batch_size=4, shuffle=True)
        val_dataloader = DataLoader(val_db, batch_size=2, shuffle=False)

        model.train()

        total_node_num = 0
        total_acc_1_train = 0
        total_acc_2_train = 0
        total_loss_train = 0

        for batchdata in tqdm(train_dataloader):
            batchdata = batchdata.to(device)
            total_node_num += batchdata.num_nodes
            optimizer.zero_grad()
            linear2_output, mean_pool_output = model(batchdata.x, batchdata.edge_index, batchdata.batch)
            loss_a = criterion_a(linear2_output, batchdata.y_a) / batchdata.num_nodes
            loss_b = criterion_b(mean_pool_output, batchdata.y_b)
            loss = theta * loss_a + (1 - theta) * loss_b
            predict_1 = linear2_output.argmax(dim=1)
            predict_2 = mean_pool_output.argmax(dim=1)
            acc_1 = ((predict_1 == batchdata.y_a).sum()).item()
            acc_2 = ((predict_2 == batchdata.y_b).sum()).item()
            total_loss_train += loss
            total_acc_1_train += acc_1
            total_acc_2_train += acc_2
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f'Epochs: {i + 1} | Train Loss: {total_loss_train / len(train_db): .3f} \
            | Task A Accuracy: {total_acc_1_train / total_node_num: .3f} \
            | Task B Accuracy: {total_acc_2_train / len(train_db): .3f}')
        final_loss_train += (total_loss_train / len(train_db))
        final_acc_a_train += (total_acc_1_train / total_node_num)
        final_acc_b_train += (total_acc_2_train / len(train_db))
        model.eval()
        total_node_num = 0
        total_acc_1_val = 0
        total_acc_2_val = 0
        total_loss_val = 0
        with torch.no_grad():
            for batchdata in tqdm(val_dataloader):
                batchdata = batchdata.to(device)
                total_node_num += batchdata.num_nodes
                linear2_output, mean_pool_output = model(batchdata.x, batchdata.edge_index, batchdata.batch)
                loss_a = criterion_a(linear2_output, batchdata.y_a) / batchdata.num_nodes
                loss_b = criterion_b(mean_pool_output, batchdata.y_b)
                loss = theta * loss_a + (1 - theta) * loss_b
                predict_1 = linear2_output.argmax(dim=1)
                predict_2 = (mean_pool_output).argmax(dim=1)
                acc_1 = ((predict_1 == batchdata.y_a).sum()).item()
                acc_2 = ((predict_2 == batchdata.y_b).sum()).item()
                total_loss_val += loss
                total_acc_1_val += acc_1
                total_acc_2_val += acc_2
        print(f'Epochs: {i + 1} | Val Loss: {total_loss_val / len(val_db): .3f} \
            | Task A Accuracy: {total_acc_1_val / total_node_num: .3f} \
            | Task B Accuracy: {total_acc_2_val / len(val_db): .3f}')
        final_loss_val += (total_loss_val / len(val_db))
        final_acc_a_val += (total_acc_1_val / total_node_num)
        final_acc_b_val += (total_acc_2_val / len(val_db))
        
        theta = theta - 0.10 if theta > 0.20 else theta

        i += 1

    print("\n")
    print(f'Final Train Loss: {final_loss_train / 10: .3f} \
        | Task A Accuracy: {final_acc_a_train / 10: .3f} \
        | Task B Accuracy: {final_acc_b_train / 10: .3f}')
    print(f'Final Val Loss: {final_loss_val / 10: .3f} \
        | Task A Accuracy: {final_acc_a_val / 10: .3f} \
        | Task B Accuracy: {final_acc_b_val / 10: .3f}')
