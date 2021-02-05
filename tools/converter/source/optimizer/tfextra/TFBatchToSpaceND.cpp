//
//  TFBatchToSpaceND.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "TFExtraManager.hpp"

namespace MNN {
namespace Express {
static bool copyInfo(SpaceBatchT* dst, std::vector<VARP> inputs) {
    MNN_ASSERT(inputs.size() == 3);
    {
        auto blockShape = inputs[1];

        auto info = blockShape->getInfo();
        auto ptr = blockShape->readMap<int>();
        if (info == nullptr) {
            MNN_ERROR("Not Cost blockShape\n");
            return false;
        }
        if (halide_type_int != info->type.code || 32 != info->type.bits) {
            MNN_ERROR("Not int type blockShape\n");
            return false;
        }
        dst->blockShape.reset(new BlobT);
        auto block        = dst->blockShape.get();
        block->dataFormat = MNN_DATA_FORMAT_NHWC;
        block->dataType = DataType_DT_INT32;
        block->dims = info->dim;
        if (ptr != nullptr) {
            block->int32s.resize(info->size);
            ::memcpy(block->int32s.data(), ptr, info->size * sizeof(int32_t));
        }
    }
    {
        auto padding = inputs[2];
        auto info = padding->getInfo();
        auto ptr = padding->readMap<int>();
        if (info == nullptr) {
            MNN_ERROR("Not Cost paddingShape\n");
            return false;
        }
        if (halide_type_int != info->type.code || 32 != info->type.bits) {
            MNN_ERROR("Not int type paddingShape\n");
            return false;
        }
        dst->padding.reset(new BlobT);
        auto block        = dst->padding.get();
        block->dataFormat = MNN_DATA_FORMAT_NHWC;
        block->dataType = DataType_DT_INT32;
        block->dims = info->dim;
        if (ptr != nullptr) {
            block->int32s.resize(info->size);
            ::memcpy(block->int32s.data(), ptr, info->size * sizeof(int32_t));
        }
    }
    return true;
}

class BatchToSpaceNDTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto type   = op->main_as_Extra()->type()->str();
        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() == 3);
        std::unique_ptr<OpT> bsND(new OpT);
        bsND->name       = expr->name();
        bsND->type       = OpType_BatchToSpaceND;
        bsND->main.type  = OpParameter_SpaceBatch;
        bsND->main.value = new SpaceBatchT;
        if (!copyInfo(bsND->main.AsSpaceBatch(), inputs)) {
            return nullptr;
        }
        auto newExpr = Expr::create(bsND.get(), inputs, expr->outputSize());
        return newExpr;
    }
};
class SpaceToBatchNDTransform : public TFExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto type   = op->main_as_Extra()->type()->str();
        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() == 3);
        std::unique_ptr<OpT> bsND(new OpT);
        bsND->name       = expr->name();
        bsND->type       = OpType_SpaceToBatchND;
        bsND->main.type  = OpParameter_SpaceBatch;
        bsND->main.value = new SpaceBatchT;
        if (!copyInfo(bsND->main.AsSpaceBatch(), inputs)) {
            return nullptr;
        }
        auto newExpr = Expr::create(bsND.get(), inputs, expr->outputSize());
        return newExpr;
    }
};
static auto gRegister = []() {
    TFExtraManager::get()->insert("BatchToSpaceND",
                                  std::shared_ptr<TFExtraManager::Transform>(new BatchToSpaceNDTransform));
    TFExtraManager::get()->insert("SpaceToBatchND",
                                  std::shared_ptr<TFExtraManager::Transform>(new SpaceToBatchNDTransform));
    return true;
}();
} // namespace Express
} // namespace MNN
