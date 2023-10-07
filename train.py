import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import DeepJIT
import torch
from tqdm import tqdm
from utils import mini_batches_train, save
import torch.nn as nn
import matplotlib.pyplot as plt
import os, datetime


def train_model(data, params):
    data_pad_msg, data_pad_code, data_labels, dict_msg, dict_code = data

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    # params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)

    if len(data_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = data_labels.shape[1]
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("device:", torch.device)
    print(torch.cuda.is_available())

    # create and train the defect model
    model = DeepJIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)

    criterion = nn.BCELoss()
    loss_df = pd.DataFrame(columns=['Epoch', 'Loss'])
    epoch_losses = []
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        # building batches for training model
        batches = mini_batches_train(X_msg=data_pad_msg, X_code=data_pad_code, Y=data_labels,
                                     mini_batch_size=params.batch_size)
        for i, (batch) in enumerate(tqdm(batches)):
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(labels)
            else:
                pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg, pad_code)
            loss = criterion(predict, labels)
            total_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
        total_loss /= (i + 1)
        print('Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))
        # loss_df = loss_df.append({'Epoch': epoch, 'Loss': total_loss.detach().item()}, ignore_index=True)
        epoch_losses.append(total_loss)
        save(model, params.save_dir, 'epoch', epoch)
    # loss_df.to_excel(os.path.join(params.save_dir, 'loss.xlsx'), index=False)

    plt.switch_backend('Agg')  # 后端设置'Agg' 参考：https://cloud.tencent.com/developer/article/1559466

    plt.figure()  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
    plt.plot(epoch_losses, 'b', label='loss')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()  # 个性化图例（颜色、形状等）
    plt.savefig(os.path.join("./snapshot/", "1_recon_loss.jpg"))  # 保存图片 路径：/imgPath/