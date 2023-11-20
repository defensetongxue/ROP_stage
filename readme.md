# dataset condition
Split_name: 1, train, {0: 3199, 1: 465, 2: 133, 3: 68}
Split_name: 1, val, {0: 1757, 1: 278, 2: 76, 3: 54}
Split_name: 1, test, {0: 1712, 1: 253, 2: 74, 3: 15}
Split_name: 2, train, {0: 3411, 1: 463, 2: 161, 3: 50}
Split_name: 2, val, {0: 1500, 1: 248, 2: 46, 3: 33}
Split_name: 2, test, {0: 1757, 1: 285, 2: 76, 3: 54}
Split_name: 3, train, {0: 3486, 1: 538, 2: 150, 3: 69}
Split_name: 3, val, {0: 1682, 1: 210, 2: 87, 3: 35}
Split_name: 3, test, {0: 1500, 1: 248, 2: 46, 3: 33}
Split_name: 4, train, {0: 3212, 1: 501, 2: 120, 3: 48}
Split_name: 4, val, {0: 1757, 1: 285, 2: 76, 3: 54}
Split_name: 4, test, {0: 1699, 1: 210, 2: 87, 3: 35}

// "before_cleansing":
Last Epoch Train Loss: 0.0943, Val Loss: 0.3907, Acc: 0.8941, Auc: 0.9594, Recall (Pos): 1.0000, Recall (Class 1): 0.4105, Recall (Class 2): 0.8816, Recall (Class 3): 0.0185, 
Best Epoch Acc: 0.9424, Auc: 0.9648, Recall (Pos): 0.9976, Recall (Class 1): 0.7860, Recall (Class 2): 0.6053, Recall (Class 3): 0.3704, 
"cleansing"
Last Epoch Train Loss: 0.1311, Val Loss: 0.3067, Acc: 0.9010, Auc: 0.9300, Recall (Pos): 1.0000, Recall (Class 1): 0.4526, Recall (Class 2): 0.7500, Recall (Class 3): 0.2593, 
Best Epoch Acc: 0.9369, Auc: 0.9613, Recall (Pos): 1.0000, Recall (Class 1): 0.8035, Recall (Class 2): 0.4868, Recall (Class 3): 0.2222, 

now all the result is not padding

add aux for the embed after transformer

Aux: Acc: 0.8656 Auc: 0.9293 RecallPos: 0.7036 Recall1: 0.4982 Recall2: 0.5526 Recall3: 0.1852 
Last Epoch Train Loss: 0.1644, Val Loss: 0.2667, Acc: 0.9438, Auc: 0.9711, Recall (Pos): 0.9904, Recall (Class 1): 0.8351, Recall (Class 2): 0.5789, Recall (Class 3): 0.2037, 
Aux: Acc: 0.8545 Auc: 0.9348 RecallPos: 0.7470 Recall1: 0.4561 Recall2: 0.4342 Recall3: 0.4630 
Best Epoch Acc: 0.9420, Auc: 0.9758, Recall (Pos): 0.9904, Recall (Class 1): 0.8351, Recall (Class 2): 0.4079, Recall (Class 3): 0.3704, 

Aux: Acc: 0.8135 Auc: 0.9072 RecallPos: 0.8169 Recall1: 0.3614 Recall2: 0.6974 Recall3: 0.2963 
Last Epoch Train Loss: 0.0917, Val Loss: 0.3034, Acc: 0.9466, Auc: 0.9749, Recall (Pos): 0.9711, Recall (Class 1): 0.9158, Recall (Class 2): 0.3553, Recall (Class 3): 0.2037, 
Aux: Acc: 0.7311 Auc: 0.8827 RecallPos: 0.8289 Recall1: 0.0912 Recall2: 0.5921 Recall3: 0.5370 
Best Epoch Acc: 0.9448, Auc: 0.9807, Recall (Pos): 0.9759, Recall (Class 1): 0.7965, Recall (Class 2): 0.5658, Recall (Class 3): 0.4630, 