import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
df1 = pd.DataFrame()
df2 = pd.DataFrame()

with open(file='./models/840_resnet50_with_pretrain.csv', mode='r') as f:
    df1 = pd.read_csv(f)

with open(file='./models/resnet50_wo_pretrain.csv', mode='r') as f:
    df2 = pd.read_csv(f)

plt.title('Result Comparison(ResNet50)', fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.xlabel("Epoch", fontsize=18)
plt.plot(range(1, 21), df1['acc_train'], label='train(with pretraining)')
plt.plot(range(1, 21), df1['acc_test'], label='test(with pretraining)')
plt.plot(range(1, 21), df2['acc_train'], label='train(w/o pretraining)c')
plt.plot(range(1, 21), df2['acc_test'], label='test(w/o pretraining)')
plt.legend()

plt.savefig('./fig/comp_resnet50.png')
plt.show()