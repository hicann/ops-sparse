/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef CANN_OPS_SPARSE_COMMON_H
#define CANN_OPS_SPARSE_COMMON_H

#include <iostream>
#include <acl/acl.h>
#include "cann_ops_sparse.h"

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

/*
 * 描述符内部结构体（aclsparseSpMatDescr / aclsparseDnVecDescr /
 * aclsparseDnMatDescr）已迁移到内部头 src/common/aclsparse_descr_internal.h，
 * 对外只暴露不透明的 struct* 句柄，此处不再定义。
 */

#endif