/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file prune_kernel.h
 * \brief Prune kernel launcher declarations
 *        (ASC side defines, CXX side includes).
 *
 * host.cpp (CXX) includes this header and calls launchers as plain C functions.
 * kernel.cpp (ASC) defines the launchers and wraps the <<<>>> invocation.
 * host.cpp MUST NOT write <<<>>> directly.
 */

#ifndef SPLT_PRUNE_KERNEL_H
#define SPLT_PRUNE_KERNEL_H

#include <cstdint>
// AclsparseltTilingData is now passed by value through the kernel
// launch args block (no device buffer allocation). The header is shared by
// both CXX (host) and ASC (kernel) TUs — the host-only section is guarded by
// __CCE_AICORE__ inside the header, so including it here is safe on both sides.
#include "shared/aclsparselt_internal.h"

// Define GM_ADDR for the CXX (host) TU so the launcher
// declaration can use GM_ADDR uniformly. On the ASC (kernel) TU, GM_ADDR is
// already defined by the compiler — the #ifndef guard skips redefinition.
// This mirrors the spsm_kernel.h pattern and eliminates the (GM_ADDR) casts in
// <<<>>> calls that triggered the "cce_global attribute ignored" warning.
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Prune A (m,k) -> A_pruned (m,k) keeping top-2 abs values per 4 elements.
// Tiling is passed by value via the kernel launch args block instead
// of a separately aclrtMalloc'd device buffer. This removes the extra device
// storage (D5: cuSPARSELt "requires no extra storage") and the
// aclrtSynchronizeStream before aclrtFree (D4: "supports asynchronous
// execution with respect to stream"). The args block is managed by the ACL
// runtime and is valid for the lifetime of the launched task.
extern void splt_prune_kernel_launch(
    GM_ADDR aGm, GM_ADDR aPrunedGm, const AclsparseltTilingData *tiling,
    int32_t dataType, uint32_t blockDim, void *stream);

#ifdef __cplusplus
}
#endif

#endif // SPLT_PRUNE_KERNEL_H
