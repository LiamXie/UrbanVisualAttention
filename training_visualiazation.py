import os 
import numpy as np

# read log.csv from each folder in runs
folders=os.listdir("./runs")
train_loss=[[] for i in range(8)]
val_loss=[[] for i in range(8)]
for folder in folders:
    path="./runs/"+folder+"/log.csv"
    with open(path) as f:
        lines=f.readlines()
        for i in range(1,9):
            data=lines[i].split(",")
            train_loss[i-1].append(float(data[1]))
            val_loss[i-1].append(float(data[2]))

train_mean=[]
train_std=[]

val_mean=[]
val_std=[]
for i in range(8):
    train_mean.append(np.mean(train_loss[i]))
    train_std.append(np.std(train_loss[i]))
    val_mean.append(np.mean(val_loss[i]))
    val_std.append(np.std(val_loss[i]))

print(val_mean)
print(val_std)