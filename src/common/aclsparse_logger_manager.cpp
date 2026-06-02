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
 * \file aclsparse_logger.cpp
 * \brief ops-sparse log配置管理模块
 */

#include "aclsparse_logger_manager.h"

namespace AclSparse {

AclSparseStatus aclsparseLoggerSetLevel(aclsparseLogLevel_t logLevel)
{
    switch (logLevel) {
        case aclsparseLogLevel_t::LOG_LEVEL_INFO:
            dlog_setlevel(OP, DLOG_INFO, 1);
            break;
        case aclsparseLogLevel_t::LOG_LEVEL_ERROR:
            dlog_setlevel(OP, DLOG_ERROR, 1);
            break;
        case aclsparseLogLevel_t::LOG_LEVEL_DEBUG:
            dlog_setlevel(OP, DLOG_DEBUG, 1);
            break;
        default:
            dlog_setlevel(OP, DLOG_INFO, 1);
            break;
    }
    return AclSparseStatus::ACL_SPARSE_STATUS_SUCCESS;
}


}