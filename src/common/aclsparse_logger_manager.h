/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclsparse_logger_manager.h
 * \brief ops-sparse log配置管理模块（不对外暴露）。
 */

#ifndef ACLSPARSE_LOGGER_MANAGER_H_
#define ACLSPARSE_LOGGER_MANAGER_H_

#include "cann_ops_sparse.h"
#include "log/log.h"

namespace AclSparse {

typedef enum aclsparseLogLevel {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO = 1,
    LOG_LEVEL_ERROR = 2,
} aclsparseLogLevel_t;

AclSparseStatus aclsparseLoggerSetLevel(aclsparseLogLevel_t logLevel);

}

#endif // ACLBLAS_LOGGER_MANAGER_H_