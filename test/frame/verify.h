/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_FRAME_VERIFY_H_
#define TEST_FRAME_VERIFY_H_

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "types.h"

namespace sparse_test {

class VerifyStrategy {
public:
    virtual ~VerifyStrategy() = default;

    bool verify(const float* output, const float* golden, size_t count, int64_t stride,
                const std::string& caseId) {
        printHead(output, count, stride, "Output", caseId);
        printHead(golden, count, stride, "Golden", caseId);

        size_t skippedCount = 0;
        for (size_t i = 0; i < count; i++) {
            float outVal = output[static_cast<int64_t>(i) * stride];
            float goldVal = golden[static_cast<int64_t>(i) * stride];
            if (shouldSkip(outVal, goldVal)) {
                skippedCount++;
                continue;
            }
            processElement(outVal, goldVal);
        }

        return reportResult(count, skippedCount, caseId);
    }

protected:
    virtual bool shouldSkip(float outVal, float goldVal) {
        if (outVal == goldVal) return true;
        if (std::isnan(outVal) && std::isnan(goldVal)) return true;
        return false;
    }

    virtual void processElement(float outVal, float goldVal) = 0;
    virtual bool reportResult(size_t count, size_t skippedCount, const std::string& caseId) = 0;

    static void printHead(const float* data, size_t count, int64_t stride,
                          const std::string& label, const std::string& caseId) {
        std::cout << std::fixed << std::setprecision(6);
        constexpr size_t kMaxPrint = 10;
        std::cout << "[" << caseId << "] " << label << ": ";
        for (size_t i = 0; i < count && i < kMaxPrint; i++) {
            std::cout << data[static_cast<int64_t>(i) * stride] << " ";
        }
        if (count > kMaxPrint) std::cout << "...";
        std::cout << std::endl;
    }
};

class AbsStrategy : public VerifyStrategy {
public:
    explicit AbsStrategy(double absTol) : absTol_(absTol) {}
protected:
    void processElement(float outVal, float goldVal) override {
        if (std::abs(outVal - goldVal) > absTol_) failCount_++;
    }
    bool reportResult(size_t count, size_t /*skippedCount*/, const std::string& caseId) override {
        bool pass = (failCount_ == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (absTol=" << absTol_ << ", " << failCount_ << "/" << count << " failures)" << std::endl;
        return pass;
    }
private:
    double absTol_;
    size_t failCount_ = 0;
};

class RelStrategy : public VerifyStrategy {
public:
    RelStrategy(double relTol, double eps) : relTol_(relTol), eps_(eps) {}
protected:
    void processElement(float outVal, float goldVal) override {
        double relErr = std::abs(outVal - goldVal) / (std::abs(goldVal) + eps_);
        if (relErr > maxRelErr_) maxRelErr_ = relErr;
    }
    bool reportResult(size_t /*count*/, size_t /*skippedCount*/, const std::string& caseId) override {
        bool pass = (maxRelErr_ < relTol_);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (maxRelErr=" << maxRelErr_ << ", relTol=" << relTol_ << ")" << std::endl;
        return pass;
    }
private:
    double relTol_;
    double eps_;
    double maxRelErr_ = 0.0;
};

class MereMareStrategy : public VerifyStrategy {
public:
    MereMareStrategy(double threshold, double multiplier)
        : threshold_(threshold), multiplier_(multiplier), outlierLimit_(multiplier * threshold) {}
protected:
    bool shouldSkip(float outVal, float goldVal) override {
        if (VerifyStrategy::shouldSkip(outVal, goldVal)) return true;
        if (std::isinf(outVal) || std::isinf(goldVal)) return true;
        return false;
    }
    void processElement(float outVal, float goldVal) override {
        double relErr = std::abs(outVal - goldVal) / (std::abs(goldVal) + kEpsilon);
        sumRelErr_ += relErr;
        if (relErr > maxRelErr_) maxRelErr_ = relErr;
        if (relErr > outlierLimit_) outlierCount_++;
    }
    bool reportResult(size_t count, size_t skippedCount, const std::string& caseId) override {
        size_t validCount = count - skippedCount;
        double mere = (validCount > 0) ? sumRelErr_ / static_cast<double>(validCount) : 0.0;

        std::cout << "[" << caseId << "] MERE=" << mere << " MARE=" << maxRelErr_
                  << " (threshold=" << threshold_ << ", outlier_limit=" << outlierLimit_;
        if (skippedCount > 0) std::cout << ", skipped " << skippedCount << " elements";
        std::cout << ")" << std::endl;

        bool pass = (mere < threshold_) && (maxRelErr_ < outlierLimit_);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (MERE < threshold && MARE < " << multiplier_ << "*threshold, "
                  << outlierCount_ << " outliers out of " << count << " elements)" << std::endl;
        return pass;
    }
private:
    static constexpr double kEpsilon = 0.00006103515625;
    double threshold_;
    double multiplier_;
    double outlierLimit_;
    double sumRelErr_ = 0.0;
    double maxRelErr_ = 0.0;
    size_t outlierCount_ = 0;
};

class ExactStrategy : public VerifyStrategy {
protected:
    void processElement(float /*outVal*/, float /*goldVal*/) override { failCount_++; }
    bool reportResult(size_t count, size_t /*skippedCount*/, const std::string& caseId) override {
        bool pass = (failCount_ == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (exact match, " << failCount_ << "/" << count << " mismatches)" << std::endl;
        return pass;
    }
private:
    size_t failCount_ = 0;
};

class IntegerStrategy : public VerifyStrategy {
protected:
    bool shouldSkip(float /*outVal*/, float /*goldVal*/) override { return false; }
    void processElement(float outVal, float goldVal) override {
        if (static_cast<int64_t>(outVal) != static_cast<int64_t>(goldVal)) failCount_++;
    }
    bool reportResult(size_t count, size_t /*skippedCount*/, const std::string& caseId) override {
        bool pass = (failCount_ == 0);
        std::cout << "[" << caseId << "] " << (pass ? "PASSED" : "FAILED")
                  << " (integer match, " << failCount_ << "/" << count << " mismatches)" << std::endl;
        return pass;
    }
private:
    size_t failCount_ = 0;
};

class Verifier {
public:
    static bool verifyVector(const std::vector<float>& output, const std::vector<float>& golden,
                             const VerifyConfig& cfg, const std::string& caseId) {
        if (output.size() != golden.size()) {
            std::cout << "[" << caseId << "] FAILED: size mismatch, output=" << output.size()
                      << " golden=" << golden.size() << std::endl;
            return false;
        }
        return verifyVector(output.data(), golden.data(), output.size(), 1, cfg, caseId);
    }

    static bool verifyVector(const float* output, const float* golden, size_t count, int64_t stride,
                             const VerifyConfig& cfg, const std::string& caseId) {
        auto strategy = createStrategy(cfg);
        return strategy->verify(output, golden, count, stride, caseId);
    }

    static bool verifyScalar(float output, float golden, const VerifyConfig& cfg, const std::string& caseId) {
        std::cout << "[" << caseId << "] Output: " << output << " Golden: " << golden << std::endl;
        if (output == golden) return true;
        if (std::isnan(output) && std::isnan(golden)) return true;
        return std::abs(output - golden) < cfg.absTol;
    }

    template <typename T>
    static std::vector<float> toFloat(const std::vector<T>& v) {
        std::vector<float> out(v.size());
        for (size_t i = 0; i < v.size(); i++) out[i] = static_cast<float>(v[i]);
        return out;
    }

private:
    static std::unique_ptr<VerifyStrategy> createStrategy(const VerifyConfig& cfg) {
        switch (cfg.mode) {
            case PrecisionMode::ABS:
                return std::make_unique<AbsStrategy>(cfg.absTol);
            case PrecisionMode::REL:
                return std::make_unique<RelStrategy>(cfg.relTol, cfg.epsilonForRel);
            case PrecisionMode::MERE_MARE:
                return std::make_unique<MereMareStrategy>(cfg.mereThreshold, cfg.mareMultiplier);
            case PrecisionMode::EXACT:
                return std::make_unique<ExactStrategy>();
            case PrecisionMode::INTEGER:
                return std::make_unique<IntegerStrategy>();
            case PrecisionMode::COMBINED:
            default:
                return std::make_unique<MereMareStrategy>(cfg.mereThreshold, cfg.mareMultiplier);
        }
    }
};

}

#endif
