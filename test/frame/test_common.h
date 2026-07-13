/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_TEST_COMMON_H_
#define TEST_FRAME_TEST_COMMON_H_

// Common includes shared by all new-operator GTest + CSV tests.
// Each test file typically adds its operator-specific golden/param/wrapper headers.

#include "acl/acl.h"
#include "cann_ops_sparse.h"

#include "sparse_test.h"
#include "csv_loader.h"
#include "descriptor_manager.h"
#include "fill.h"
#include "verify.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#endif  // TEST_FRAME_TEST_COMMON_H_
