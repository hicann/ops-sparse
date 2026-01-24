# ops-sparse

## 一、什么是ops-sparse
### ops-sparse介绍
ops-sparse 是基于昇腾芯片的高性能稀疏算子库，提供常用的稀疏矩阵基础运算。  
### 软件架构
待补充


### ops-sparse仓介绍

ops-sparse库的目录结构如下：

```
ops-sparse
├── build          //可存放构建生成的文件
├── docs           //文档文件
├── example        //算子调用示例代码，包含可直接运行的Demo
├── include        //存放公共头文件
├── scripts        //脚本文件存放目录
├── src            //主体源代码目录
│   ├── common     //初始化、上下文管理等通用代码
│   ├── spmv       //spmv算子实现
│   ├──  ...       //其他算子实现
│   └── CMakeLists.txt
├── tests          //测试代码
```

## 二、环境构建
### 快速安装CANN软件
本节提供快速安装CANN软件的示例命令，更多安装步骤请参考[详细安装指南](#cann详细安装指南)。

#### 安装前准备
在线安装和离线安装时，需确保已具备Python环境及pip3，当前CANN支持Python3.7.x至3.11.4版本。
离线安装时，请单击[获取链接](https://www.hiascend.com/developer/download/community/result?module=cann)下载CANN软件包，并上传到安装环境任意路径。
#### 安装CANN
```shell
chmod +x Ascend-cann-toolkit_8.5.RC1_linux-$(arch).run
./Ascend-cann-toolkit_8.5.RC1_linux-$(arch).run --install
```
#### 安装后配置
配置环境变量脚本set_env.sh，当前安装路径以${HOME}/Ascend为例。
```
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
```  

### CANN详细安装指南 
开发者可访问[昇腾文档-昇腾社区](https://www.hiascend.com/document)->CANN社区版->软件安装，查看CANN软件安装引导，根据机器环境、操作系统和业务场景选择后阅读详细安装步骤。

### 基础工具版本要求与安装

安装CANN之后，您可安装一些工具方便后续开发，参见以下内容：

* [CANN依赖列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0045.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)
* [CANN安装后操作](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0094.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)

## 三、快速上手
### ops-sparse编译
 - 加速库下载
    ```sh
    git clone https://gitcode.com/cann/ops-sparse.git
    ```
   您可自行选择需要的分支。
 - 加速库编译  
    编译加速库，设置加速库环境变量：
    ```sh
    cd ops-sparse
    bash scripts/build.sh
    source output/ops-sparse/set_env.sh
    ```
    
### 调用示例说明
本节示例代码分别展示了如何通过C++调用算子。
待补充

## 四、参与贡献
 
1.  fork仓库
2.  修改并提交代码
3.  新建 Pull-Request

## 六、学习资源

## 七、参考文档
**[CANN社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/index/index.html)**  