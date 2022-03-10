# 使用说明

## 运行环境

- paddle
- paddlenlp

## 文件说明

- baseline.py 是从notebook上移植下来的完整代码，耦合度高，未作拆分，可以以后修改的时候参考来用。
- train.py 训练模型的文件，当需要训练模型的时候，`python train.py`即可。
- predict.py 预测推理的文件，**注意**：里面目前采用的是单句预测，考虑到我们的数据集还没有完成，并没有实现输入一整个文档进行逐句预测。如果有这个需求，也可以进行更改。当需要进行预测的时候，`python predict.py`即可
- data.py 是对训练数据的封装等的一些操作
- utils文件夹内是数据操作的一些函数
- checkpoint文件夹内保存了当前最好的模型，在predict的时候加载使用。

## 备注

- 由于直接从notebook进行拆分，所以代码比较简易丑陋
- 参数的设置只能在文件里进行修改，尚未实现`python train.py -parameters parameter`的形式。没有进行`arg_parser.add_argument`的封装。