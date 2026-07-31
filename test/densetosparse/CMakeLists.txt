# DenseToSparse GTest + CSV tests for arch35.
ops_sparse_add_gtest_tests(densetosparse ${OPS_SPARSE} EIGEN
    EXTRA_INCLUDES
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        ${CMAKE_SOURCE_DIR}/sparse/densetosparse/arch35
)

add_custom_command(TARGET densetosparse_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/arch35/densetosparse_l2_cases.csv
        $<TARGET_FILE_DIR:densetosparse_test>/densetosparse_l2_cases.csv
    COMMENT "Copying DenseToSparse L2 CSV to build dir"
)
