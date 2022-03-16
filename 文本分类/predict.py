import os
from multiprocessing import cpu_count
import numpy as np
import shutil
import paddle
import paddle.fluid as fluid



# 用训练好的模型进行预测并输出预测结果
# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program()) # paddle版本问题会报错,考虑降低paddle版本



save_path = './infer_model/'

# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)


# 获取数据
def get_data(sentence):
    # 读取数据字典
    with open('./datasets/dict_txt.txt', 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    dict_txt = dict(dict_txt)
    # 把字符串数据转换成列表数据
    keys = dict_txt.keys()
    data = []
    for s in sentence:
        # 判断是否存在未知字符
        if not s in keys:
            s = '<unk>'
        data.append(int(dict_txt[s]))
    return data


data = []
# 获取预测数据
# data1 = get_data('北碚养老院')
# data2 = get_data('旗山湖公园')
# data.append(data1)
# data.append(data2)
data_tmp = ['养老院', '旗山湖公园', '光大餐厨处理厂', '三号厂房', '四川天欣药业有限公司', 
'吉林大学第三医院分院', '馨康花园17号楼', '杭州 象山大厦店', '法库县人民政府', '小蚌埠派出所']
for i in data_tmp:
    data.append(get_data(i))

# 获取每句话的单词数量
base_shape = [[len(c) for c in data]]

# 生成预测数据
tensor_words = fluid.create_lod_tensor(data, base_shape, place)

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: tensor_words},
                 fetch_list=target_var)

# 分类名称
# names = [ '医院', '娱乐', '体育', '财经','房产', '汽车', '教育', '科技', '国际', '证券']
names = ['企业_制造业办公/工厂', '企业_小企业办公', '别墅家居_别墅家居', '医院_乡镇卫生院', '医院_其他医院', 
'医院_区县对公医院', '医院_私立医院', '商业服务_KTV', '商业服务_商业其他', '商业服务_商业地产', '商业服务_商业大厦/写字楼', 
'商业服务_商业综合体', '商业服务_景区', '商业服务_网吧', '商业服务_餐饮', '商业连锁_连锁其他', '商业连锁_餐饮连锁', '政府_区县政府', '政府_政府其他', 
'教育_中学', '教育_其他学校', '教育_培训机构', '教育_大学/学院', '教育_小学', '教育_幼儿园', '教育_职业学校', 
'酒店_公寓群租房', '酒店_星级酒店', '酒店_民宿', '酒店_经济型单体酒店', '酒店_连锁酒店', '金融_证券保险', '金融_银行']

# 获取结果概率最大的label
for i in range(len(data)):
    lab = np.argsort(result)[0][i][-1]
    print('预测结果标签为：%d， 名称为：%s， 概率为：%f' % (lab, names[lab], result[0][i][lab]))