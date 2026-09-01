ops_sparse_add_gtest_tests(spgemm ${OPS_SPARSE} WARN_ON_MISSING_SRC)
if(TARGET spgemm_test)
    target_compile_features(spgemm_test PRIVATE cxx_std_17)
endif()
