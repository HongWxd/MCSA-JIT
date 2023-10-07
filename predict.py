import subprocess

for epoch in range(1, 151):
    # 构造命令字符串
    command = f"python main.py -predict -pred_data D:\Program\ISSTA21-JIT-DP-master\DeepJIT\data\openstack_test.pkl -dictionary_data D:\Program\ISSTA21-JIT-DP-master\DeepJIT\data\openstack_dict.pkl -load_model D:\Program\ISSTA21-JIT-DP-master\DeepJIT\snapshot\epoch_{epoch}.pt"

    # command = f"python main.py -predict -pred_data D:\Program\ISSTA21-JIT-DP-master\DeepJIT\data\qt_test.pkl -dictionary_data D:\Program\ISSTA21-JIT-DP-master\DeepJIT\data\qt_dict.pkl -load_model D:\Program\ISSTA21-JIT-DP-master\DeepJIT\snapshot\epoch_{epoch}.pt"

    # 执行命令并输出结果
    output = subprocess.check_output(command, shell=True).decode()
    print(f"Epoch {epoch} evaluation result:\n{output}")

    # command = f"python main.py -predict -pred_data D:\Program\ISSTA21-JIT-DP-master\DeepJIT\data\openstack_test.pkl -dictionary_data D:\Program\ISSTA21-JIT-DP-master\DeepJIT\data\openstack_dict.pkl -load_model D:\Program\ISSTA21-JIT-DP-master\DeepJIT\snapshot\epoch_{epoch}.pt"