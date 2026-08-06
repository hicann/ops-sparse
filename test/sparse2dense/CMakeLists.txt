# SparseToDense GTest + CSV tests for arch35.
ops_sparse_add_gtest_tests(sparse2dense ${OPS_SPARSE}
    EXTRA_INCLUDES
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        ${CMAKE_SOURCE_DIR}/sparse/sparse2dense/arch35
)

add_custom_command(TARGET sparse2dense_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/arch35/sparse2dense_l2_cases.csv
        $<TARGET_FILE_DIR:sparse2dense_test>/sparse2dense_l2_cases.csv
    COMMENT "Copying SparseToDense L2 CSV to build dir"
)
