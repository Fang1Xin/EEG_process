# 脑电数据预处理---念通智能、Ant设备版

自行整理python-mne的脑电数据预处理流程，欢迎随时讨论交流

代码针对念通智能64导脑电设备采集得到的脑电信号bdf文件，其他设备见后续文件

代码中使用mne包进行处理，官网：https://mne.tools/stable/documentation/index.html

## 代码和数据导入

建议在新建工程文件夹下再新建两个文件夹code和data分别存放代码和脑电数据

![image-20250409115234669](./pic/image-20250409115234669.png)

将preprocess.py和analysis.py代码文件放置在code文件夹里

## preprocess代码解析

脑电的预处理过程并非代码全自动进行，仍需实验人员对处理过程的准确性进行把关，因为代码使用jupter notebook的方式运行，每一块代码顺序运行

### 导入包

```python
# 导入包
import numpy as np
import mne
import os
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
%matplotlib qt 
```

### 脑电数据导入

念通智能设备数据导入

```python
input_data_path = r"D:\Documents\EEG\standard-EEG\data\静息3min_20250408095252.bdf"  # 修改：脑电数据位置
raw = mne.io.read_raw_bdf(input_data_path, preload=True)  # EEG为读入进来的脑电数据
```

Ant设备数据导入

```python
# 数据载入与（Ant设备版）
input_data_path = r"D:\Documents\EEG\split_EEG\data\fazhan_ban_2025-01-08_19-24-01.cnt"  # 修改：脑电数据位置
raw = read_raw_ant(input_data_path, eog=r"EOG", preload=True)
raw.pick("eeg").set_montage("standard_1005")
raw.plot_sensors(kind='topomap', show_names=True)  # 查看电极位置（可注释）
```

按照数据的格式修改对应的read_raw_bdf，mne支持多种脑电格式

- [`read_raw_ant`](https://mne.tools/stable/generated/mne.io.read_raw_ant.html#mne.io.read_raw_ant)
- [`read_raw_artemis123`](https://mne.tools/stable/generated/mne.io.read_raw_artemis123.html#mne.io.read_raw_artemis123)
- [`read_raw_bdf`](https://mne.tools/stable/generated/mne.io.read_raw_bdf.html#mne.io.read_raw_bdf)
- [`read_raw_boxy`](https://mne.tools/stable/generated/mne.io.read_raw_boxy.html#mne.io.read_raw_boxy)
- [`read_raw_brainvision`](https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html#mne.io.read_raw_brainvision)
- [`read_raw_cnt`](https://mne.tools/stable/generated/mne.io.read_raw_cnt.html#mne.io.read_raw_cnt)
- [`read_raw_ctf`](https://mne.tools/stable/generated/mne.io.read_raw_ctf.html#mne.io.read_raw_ctf)
- [`read_raw_curry`](https://mne.tools/stable/generated/mne.io.read_raw_curry.html#mne.io.read_raw_curry)
- [`read_raw_edf`](https://mne.tools/stable/generated/mne.io.read_raw_edf.html#mne.io.read_raw_edf)
- [`read_raw_eeglab`](https://mne.tools/stable/generated/mne.io.read_raw_eeglab.html#mne.io.read_raw_eeglab)
- [`read_raw_egi`](https://mne.tools/stable/generated/mne.io.read_raw_egi.html#mne.io.read_raw_egi)
- [`read_raw_eximia`](https://mne.tools/stable/generated/mne.io.read_raw_eximia.html#mne.io.read_raw_eximia)
- [`read_raw_eyelink`](https://mne.tools/stable/generated/mne.io.read_raw_eyelink.html#mne.io.read_raw_eyelink)
- [`read_raw_fieldtrip`](https://mne.tools/stable/generated/mne.io.read_raw_fieldtrip.html#mne.io.read_raw_fieldtrip)
- [`read_raw_fif`](https://mne.tools/stable/generated/mne.io.read_raw_fif.html#mne.io.read_raw_fif)
- [`read_raw_fil`](https://mne.tools/stable/generated/mne.io.read_raw_fil.html#mne.io.read_raw_fil)
- [`read_raw_gdf`](https://mne.tools/stable/generated/mne.io.read_raw_gdf.html#mne.io.read_raw_gdf)
- [`read_raw_kit`](https://mne.tools/stable/generated/mne.io.read_raw_kit.html#mne.io.read_raw_kit)
- [`read_raw_nedf`](https://mne.tools/stable/generated/mne.io.read_raw_nedf.html#mne.io.read_raw_nedf)
- [`read_raw_nicolet`](https://mne.tools/stable/generated/mne.io.read_raw_nicolet.html#mne.io.read_raw_nicolet)
- [`read_raw_nihon`](https://mne.tools/stable/generated/mne.io.read_raw_nihon.html#mne.io.read_raw_nihon)
- [`read_raw_nirx`](https://mne.tools/stable/generated/mne.io.read_raw_nirx.html#mne.io.read_raw_nirx)
- [`read_raw_nsx`](https://mne.tools/stable/generated/mne.io.read_raw_nsx.html#mne.io.read_raw_nsx)
- [`read_raw_persyst`](https://mne.tools/stable/generated/mne.io.read_raw_persyst.html#mne.io.read_raw_persyst)
- [`read_raw_snirf`](https://mne.tools/stable/generated/mne.io.read_raw_snirf.html#mne.io.read_raw_snirf)

### 对脑电数据的电极位置进行定位

```python
channel_montage = mne.channels.read_custom_montage(r"D:\Documents\EEG\ERP analysis\standard_1005.elc")  # 修改： .elc为电极定位文件（念通智能64导），需根据实际情况修改该文件位置
raw.drop_channels(['I0'])  # 去除眼睛下方的电极
raw.set_montage(channel_montage)  # 将电极位置坐标赋予脑电数据各通道
raw.plot_sensors(kind='topomap', show_names=True)  # 查看电极位置（可注释）
```

代码最后一句一般仅在第一次载入电极定位文件时使用，用于查看位置是否正确，之后可注释掉该句

#### 查看脑电数据并先删除实验开始前后不稳定的数据

注：这一步非必须，对于没有marker、后续不进行分段的脑电数据，可进行这一步，如果后续仅分析分段的脑电数据，这一步不用进行！

在实验最开始记录时，以及快要结束时，往往会因为被试移动等多种因素导致数据不稳定，同时这两段脑电数据并非实验需要，因此将其截取掉，只保留中间的数据

```python
fig = raw.plot(block=True)  # 可视化脑电信号
# --------------分割线----------------（分为两个代码块运行）
EEG = raw.copy().crop(tmin=11.314, tmax=raw.times[-1])  # 修改：脑电数据截取时段
fig_after = EEG.plot(block=True)  # 可视化脑电信号
```

首先查看输入的脑电数据，可视化的界面中鼠标点击数据背景可放置垂直指示线，显示当前时间点，记录需要保留的中间段开始时刻，再放置垂直指示线至数据后段记录需要保留的中间段结束时刻，对应修改tmin和tmax，示例中保留从11.314s到数据结束这一时段的数据

代码最后一句再次查看截取后保留的脑电数据，做进一步检查

弹出的可视化交互界面中，可进行的操作如下

![image-20250410105519340](./pic/image-20250410105519340.png)

### 脑电赋值

如果不进行上述删除首尾两端的数据的操作，需要进行这一步

```python
EEG = raw
```



### 滤波

```python
EEG.notch_filter(50)  # 陷波滤波，去除工频干扰50Hz
EEG.filter(l_freq=0.1, h_freq=None)  # 修改：高通滤波，保留大于0.1Hz的波
EEG.filter(l_freq=None, h_freq=100)  # 修改：低通滤波，保留小于100Hz的波
```

去除工频干扰并将脑电信号滤波至0.1-100Hz（自行设置）



### 剔除坏电极

```python
bad_channels = mne.preprocessing.find_bad_channels_lof(EEG, threshold=2)  # 设置threshold，超参数，数值越小越严格
# ------- 分割线 -------（先运行第一个代码块，可从输出界面查看坏导，并根据要求对应修改阈值）
EEG.info['bads'] = bad_channels
EEG = EEG.interpolate_bads(reset_bads=True)  # 插值坏导
```

采用局部离群因子算法自动识别坏电极，如果一个通道的统计特征在附近通道中格格不入就会被标为离群值，threshold是离群值阈值，大于1表即开始离群，mne默认阈值设置为1.5，数值越小越严格，这个值是一个超参数，自行设置

识别得到坏导后进行插值得到新的数据



### 降采样

```python
EEG.resample(sfreq=500)  # 降采样500HZ
```

可选，不需要降采样可注释该句



### 重参考

一般将重参考设置为双侧乳突的均值、全局平均参考、零电位参考，在原参考未被记录的情况下，可用前两种参考方法，当原参考电极数值可恢复时，三种方法均可使用，并建议使用零电位参考

念通智能设备使用的参考电极为FCz，位于头顶，且参考电极数据未被记录，将其重参考至双侧乳突的均值或进行全局平均参考，首先需要查看双侧乳突电极的数据质量，如果数据质量较差且电极通道数>=64，使用平均参考；质量较好时依据电极通道数进行选择，电极数<32不要使用平均参考

```python
fig = EEG.plot(block=True)
EEG.set_eeg_reference(ref_channels='average')  # 修改：ref_channels
```

```python
EEG.set_eeg_reference(ref_channels=['TP9', 'TP10'])   # 采用双侧乳突平均值时 
EEG.set_eeg_reference(ref_channels='REST')  # 采用零电位参考时 
```

### 分段+剔除坏段（如果无marker则直接进行到整体去伪迹ICA小节）

```python
# 剔除坏段:通过annotation对数据进行分段，并根据峰峰值拒绝某些段
# 查看数据中的marker
events, event_id = mne.events_from_annotations(EEG)
print(events)
print(event_id)
```

首先查看脑电数据中都有哪些marker并筛选出需要的

```python
# 删除无用marker
del_arr = ['boundary', 'S 10', 'S 11', 'S 12']
for i in del_arr:
    event_id.pop(i, None)
```

删除无用，这一步没有就不运行了

```python
# 按剩余marker分段并进行基线校准，基线范围可设定baseline，分段时间可设定（基于marker时刻），根据峰峰值200uV拒绝坏段
epochs = mne.Epochs(EEG, events, event_id, preload=True, tmin=-0.5, tmax=2, baseline=[-0.2, 0], reject=dict(eeg=200e-6))
```

开始分段，拒绝坏段

### 分段脑电 去伪迹 ICA

此处的代码与后一部分整体脑电ICA仅输入数据形式（整体/分段）不同，代码细微差异，不详细说明，请参考后面整体ICA部分的说明

```python
# 去伪迹 ICA
ica = ICA(n_components=30,  max_iter='auto')
ica.fit(epochs)
ica.plot_sources(epochs, show=True)
ica.plot_components()
```

```python
# ICA 自动寻找肌电
muscle_idx_auto, scores = ica.find_bads_muscle(EEG)
ica.plot_scores(scores, exclude=muscle_idx_auto)  # 输出每一个成分被判断为肌电的分数
print(f"Automatically found muscle artifact ICA components: {muscle_idx_auto}")  # 输出肌电伪迹的成分号
```

```python
# ICA 自动寻找眼电
eog_indices, eog_scores = ica.find_bads_eog(EEG, ch_name='Fpz')  # 设置伪眼电通道 ‘FPz’
print(f"Automatically found eye artifact ICA components: {eog_indices}")
```

```python
# 剔除肌电、眼电、心电成分
ica.exclude = [0, 1, 5, 6, 10, 12, 14, 16, 19, 25]
ica.apply(epochs)
epochs.plot(show=True)
```

```python
# 输出预处理后的脑电信号段
output_data_path = r"D:\Documents\EEG\standard-EEG\data\preocessed_epo.fif"
epochs.save(output_data_path)
```

注意此处修改保存地址，预处理完成

-------------------------------------- 如果数据不进行分段直接进行这一部分的ICA，注意不要运行成上面分段后的ICA，会报错（ica.fit()里的内容不一样）---------------------------------------------

### 整体脑电 去伪迹 ICA

首先，ICA分解为n个成分，此处设置为30，ICA函数的运行逻辑是先将数据PCA降维后再进行ICA分解，为了ICA算法的稳定性和收敛性，设置其成分数一般为15-30；

```python
ica = ICA(n_components=30,  max_iter='auto')
ica.fit(EEG)
ica.plot_sources(EEG, show=True)
ica.plot_components()
```

得到两类图，源+成分，从两种图手动观察眼电、肌电、心电

初步判断：

肌电成分：17、21

![image-20250411154450847](./pic/image-20250411154450847.png)

![image-20250411154526842](./pic/image-20250411154526842.png)

肌电成分的主要特征参见mne：https://mne.tools/stable/auto_examples/preprocessing/muscle_ica.html    成分波形为很密集的棘波，地形图主要集中在颞叶区/某一个位置

![image-20250411154720359](./pic/image-20250411154720359.png)



眼电成分：1、6

以下两种均为比较典型的眼电成分，两类眼电成分也有可能以红蓝颜色互换的形式呈现

![image-20250411154616451](./pic/image-20250411154616451.png)

心电成分：参见https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html

![image-20250411155138940](./pic/image-20250411155138940.png)

使用算法自动寻找肌电伪迹，mne所采取的判断方法如下，再得到算法判断的成分之后，结合实验人员的判断综合考虑（需要实验人员double check，不能全部依赖算法实现）

![image-20250411155509270](./pic/image-20250411155509270.png)

算法同时输出每一个成分判断为肌电的得分

```python
muscle_idx_auto, scores = ica.find_bads_muscle(EEG)
ica.plot_scores(scores, exclude=muscle_idx_auto)  # 输出每一个成分被判断为肌电的分数
print(f"Automatically found muscle artifact ICA components: {muscle_idx_auto}") # 输出肌电伪迹的成分号
```

```python
# 再次查看综合考虑后所判断的肌电成分，进行查看，这一步可以不进行
ica.plot_properties(raw, picks=[17, 19])  # 结合自行判断+程序判断，查看对应的肌电成分
```

使用算法自动寻找眼电、心电伪迹时，需要匹配眼电、心电通道的数据，但是实验采用的设备基本上都是仅脑电，因此可以设置一个伪眼电，即ch_name设置为靠近眼睛的某一电极；无法设置伪心电，因此需要实验人员判断

```python
eog_indices, eog_scores = ica.find_bads_eog(EEG, ch_name='Fpz')  # 设置伪眼电通道 ‘FPz’
print(f"Automatically found eye artifact ICA components: {eog_indices}")
```

程序自动判断出的眼电成分同样与实验人员判断出的眼电进行综合考虑，毕竟此处设置的‘FPz’是伪眼电

将最终判断得到的肌电、眼电、心电成分剔除

```python
# 剔除肌电、眼电、心电成分
ica.exclude = [1, 6, 17, 19]
ica.apply(EEG)
EEG.plot(show=True)  # 查看去伪迹后的数据，可注释
```

保存预处理后的脑电信号

```python
# 输出预处理后的脑电信号
output_data_path = r"D:\Documents\EEG\standard-EEG\data\preocessed_eeg.fif"
EEG.save(output_data_path)
```
指定输出路径，保存fif格式，fif为mne脑电格式，若后续需要用eeglab处理，可使用EEG.export()替换，并将path里数据格式变为.set文件进行保存

