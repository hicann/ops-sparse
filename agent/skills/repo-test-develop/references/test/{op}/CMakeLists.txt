# TEMPLATE: CMakeLists.txt 模板（新算子 GTest + CSV 模式）
# 使用 ops_sparse_add_gtest_tests 宏注册算子
# 宏位于 cmake/test.cmake，自动处理：
#   - 链接 GTest + test/frame/test_main.cpp（共享 main 入口）
#   - 包含 test/frame/ 作为 include 路径
#   - 自动查找 test/{op_name}/archXX/{op_name}_test.cpp（按 SOC_ARCH_DIRS 匹配）
#   - 自动拷贝 test/{op_name}/archXX/{op_name}_test.csv 到 build 目录
#   - 默认启用 Eigen（可通过 TEST_USE_EIGEN=OFF 关闭）

ops_sparse_add_gtest_tests({op_name} ${OPS_SPARSE})

# TEMPLATE: 老算子（仅 spmv/spmm，自定义 main + TestRegistry）使用：
# ops_sparse_add_test({op_name} ${OPS_SPARSE})
# 不链接 GTest，不链接共享 test_main.cpp
