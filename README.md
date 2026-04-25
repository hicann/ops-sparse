# ops-sparse

## 🔥Latest News

- [2026/05] ops-sparse项目上线，提供稀疏矩阵计算的API以及优化的稀疏矩阵向量乘法实现。

## 🚀概述

ops-sparse是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供高性能稀疏矩阵计算的算子库，专注于优化稀疏矩阵的计算效率。

## ⚡️快速入门

若您希望**从零到一快速体验**项目能力，请访问下述简易教程。

1. [环境部署](docs/zh/install/quick_install.md)：介绍基础环境搭建，包括软件包和三方依赖的获取和安装、源码下载等。

   > **说明**：本步骤是QuickStart和各类教程的操作前提，请先完成基础环境搭建。
2. [QuickStart](docs/QUICKSTART.md)：提供快速上手本项目能力的指南，包括编译部署、算子调用/开发/调试等核心能力。

## 📖学习教程

若您已学习**环境部署和QuickStart**，对本项目有一定认知，并希望**深入了解和体验项目**，请访问下述详细教程。

1. [接口列表](docs/zh/api_list.md)：提供全量API信息，方便您查看aclsparse接口的分类和功能。

## 🔍目录结构

ops-sparse仓关键目录结构如下。

```txt
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

## 💬相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-linear-algebra)

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/ops-sparse/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/ops-sparse/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/ops-sparse/wiki)获取技术文章，如系列化教程、优秀实践等。
