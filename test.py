from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from math import sqrt

def test(dataset, model, device):
    model.eval()
    theta = 0.50
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)   
    criterion = torch.nn.CrossEntropyLoss()
    y_a_true = None
    y_a_pred = None
    y_b_true = None
    y_b_pred = None
    with torch.no_grad():
        total_node_num = 0
        total_acc_1_test = 0
        total_acc_2_test = 0
        total_loss_test = 0
        for batchdata in tqdm(dataloader):
            batchdata = batchdata.to(device)
            total_node_num += batchdata.num_nodes
            linear2_output, mean_pool_output = model(batchdata.x, batchdata.edge_index, batchdata.batch)
            loss_a = criterion(linear2_output, batchdata.y_a) / batchdata.num_nodes
            loss_b = criterion(mean_pool_output, batchdata.y_b)
            loss = theta * loss_a + (1 - theta) * loss_b
            predict_a = linear2_output.argmax(dim=1)
            predict_b = mean_pool_output.argmax(dim=1)
            
            if y_a_true is None:
                y_a_true = batchdata.y_a.to(torch.device('cpu'))
            else:
                y_a_true = torch.cat((y_a_true, batchdata.y_a.to(torch.device('cpu'))))
            if y_a_pred is None:
                y_a_pred = predict_a.to(torch.device('cpu'))
            else:
                y_a_pred = torch.cat((y_a_pred, predict_a.to(torch.device('cpu'))))
            
            if y_b_true is None:
                y_b_true = batchdata.y_b.to(torch.device('cpu'))
            else:
                y_b_true = torch.cat((y_b_true, batchdata.y_b.to(torch.device('cpu'))))
            if y_b_pred is None:
                y_b_pred = predict_b.to(torch.device('cpu'))
            else:
                y_b_pred = torch.cat((y_b_pred, predict_b.to(torch.device('cpu'))))
                
            acc_1 = ((predict_a == batchdata.y_a).sum()).item()
            acc_2 = ((predict_b == batchdata.y_b).sum()).item()
            total_loss_test += loss
            total_acc_1_test += acc_1
            total_acc_2_test += acc_2
        
        print(f'Train Loss: {total_loss_test / len(dataset): .3f} \
            | Task A Accuracy: {total_acc_1_test / total_node_num: .3f} \
            | Task B Accuracy: {total_acc_2_test / len(dataset): .3f}')
        b_f1_score = f1_score(y_true=y_b_true, y_pred=y_b_pred, average='macro', zero_division=1)
        print(f'Task B F1 Macro Score is: {b_f1_score: .4f}')
        b_rmse = sqrt(mean_squared_error(y_true=y_b_true, y_pred=y_b_pred, multioutput='uniform_average'))
        print(f'Task B RMSE is: {b_rmse: .4f}')
        a_f1_score = f1_score(y_true=y_a_true, y_pred=y_a_pred, average='macro', zero_division=1)
        print(f'Task A F1 Macro Score is: {a_f1_score: .4f}')
