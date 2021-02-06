//
//  main.cpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "OpInJs.h"
#include "include/jerryscript.h"
#include "include/jerryscript-ext/handler.h"
using namespace MNN::Express;

bool run();

int main(int argc, char* argv[]) { 
    printf("更新！！\n");
    printf("is %d \n", argc);
    generate_input_data();
    if (argc > 2) {
        auto type = (MNNForwardType)atoi(argv[2]);
        printf("the type is %d \n", type);
        FUNC_PRINT(type);
        MNN::BackendConfig config;
        if (argc > 3) {
            auto precision   = atoi(argv[3]);
            config.precision = (MNN::BackendConfig::PrecisionMode)precision;
        } else {
            config.precision = MNN::BackendConfig::Precision_High;
        }
        printf("here\n");
        MNN::Express::Executor::getGlobalExecutor()->setGlobalExecutorConfig(type, config, 1);
    }
    // if (argc > 1) {
    //     auto name = argv[1];
    //     MNNTestSuite::run(name);
    // } else {
    //     //run();
    //   MNNTestSuite::runAll();
    //  }


    jerry_init (JERRY_INIT_EMPTY);

    /* Register 'print' function from the extensions */
    jerryx_handler_register_global ((const jerry_char_t *) "print",
                                    jerryx_handler_print);
    
    register_js_op();

    const jerry_char_t my_js_scripts[] = "\
        var input = convInput(); \
        var filter = convFilter(); \
        var bias  = convBias(); \
        input = Convert(input, 'NCHW'); \
        var start = getCurrentTime(); \
        print(start); \
        for(var i = 0; i < 1000; ++i) \
        { \
            Conv(filter, bias, input);  \
        } \
        var end = getCurrentTime(); \
        print(end); \
        print(end - start); \
        ";
    
    jerry_value_t val = jerry_eval(my_js_scripts,
                                    sizeof(my_js_scripts) - 1,
                                    JERRY_PARSE_NO_OPTS);

    jerry_release_value(val);
    jerry_cleanup ();
                                
    return 0;
}
