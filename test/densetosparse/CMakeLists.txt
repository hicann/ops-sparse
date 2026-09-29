# DenseToSparse GTest tests. Source resolved per-SOC by the shared function:
#   ascend910b* -> test/densetosparse/arch22/densetosparse_test.cpp (arch22 UT)
#   ascend950*  -> test/densetosparse/arch35/densetosparse_test.cpp (L2 cases)
ops_sparse_add_gtest_tests(densetosparse ${OPS_SPARSE} EIGEN
    EXTRA_INCLUDES
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
)

if(TARGET densetosparse_test AND SOC_ARCH_DIRS MATCHES "arch35")
    add_custom_command(TARGET densetosparse_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/arch35/densetosparse_l2_cases.csv
            $<TARGET_FILE_DIR:densetosparse_test>/densetosparse_l2_cases.csv
        COMMENT "Copying DenseToSparse L2 CSV to build dir"
    )
endif()

if(TARGET densetosparse_test AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/arch22/densetosparse_perf.cpp)
    add_executable(densetosparse_perf arch22/densetosparse_perf.cpp)
    target_compile_features(densetosparse_perf PRIVATE cxx_std_17)
    target_include_directories(densetosparse_perf PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/sparse/common
        ${CMAKE_SOURCE_DIR}/sparse/densetosparse/arch22
        ${CMAKE_SOURCE_DIR}/test/frame
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        $ENV{LINUX_INCLUDE_PATH})
    target_link_libraries(densetosparse_perf PRIVATE
        ${OPS_SPARSE}
        ${_ops_sparse_ascendcl_lib}
        ${_ops_sparse_libc_sec})
endif()
