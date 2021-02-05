//
//  Program.hpp
//  MNNConverter
//
//  Created by MNN on 2019/09/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Program_hpp
#define Program_hpp
#include <fstream>
#include <map>
#include <unordered_map>
#include <sstream>
#include <string>
#include <MNN/expr/Expr.hpp>
namespace MNN {
namespace Express {

struct Frame;
class Program {
public:
    void emit(std::ostream& output);
    void emitPython(std::ostream& output);
    void emitUtils(std::ostream& output);
    static std::shared_ptr<Program> create(const MNN::NetT* net, bool supportExtra);
    std::vector<VARP> outputs() const {
        return mOutputs;
    }
    bool needGenerateCode() const;

    void removeDeadNodes();

    void input(const std::unordered_map<std::string, VARP>& inputs);

private:
    Program() {
    }
    std::vector<std::shared_ptr<Frame>> mFrames;
    std::map<int, VARP> mVars;
    std::vector<VARP> mOutputs;
};
} // namespace Express
}; // namespace MNN

#endif
