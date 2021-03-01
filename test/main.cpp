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
#include <map>
#include <vector>
using namespace MNN;
using namespace MNN::Express;
bool run();
static inline std::string forwardType(MNNForwardType type) {
    switch (type) {
        case MNN_FORWARD_CPU:
            return "CPU";
        case MNN_FORWARD_VULKAN:
            return "Vulkan";
        case MNN_FORWARD_OPENCL:
            return "OpenCL";
        case MNN_FORWARD_METAL:
            return "Metal";
        default:
            break;
    }
    return "N/A";
}


int main(int argc, char* argv[]) {
    std::vector<std::string> model_files{
        "ResNet_50.js", "ResNet_18.js", "ResNet_34.js",
        "ResNet_101.js", "ResNet_152.js",
        "MobileNetV1_100_1.0_224.js", "SqueezeNet.js", "GoogLeNet.js",
        "MobileNetV1_100_1.0_128.js", "MobileNetV1_100_1.0_192.js"
    };
    std::cout << "MNN Expr Models benchmark" << std::endl;

    size_t loop = 10;
    MNNForwardType forward = MNN_FORWARD_OPENCL;
    size_t numThread = 2;

    if (argc >= 2) {
        loop = atoi(argv[2]);
    }
    if (argc >= 3) {
        forward = static_cast<MNNForwardType>(atoi(argv[3]));
    }
    if (argc >= 4) {
        numThread = atoi(argv[3]);
    }

    std::cout << "Forward type: **" << forwardType(forward) << "** thread=" << numThread << std::endl;
    ScheduleConfig config;
    config.type = forward;
    config.numThread = numThread;
    BackendConfig bnConfig;
    bnConfig.precision = BackendConfig::Precision_Low;
    bnConfig.power = BackendConfig::Power_High;
    config.backendConfig = &bnConfig;


    jerry_init (JERRY_INIT_EMPTY);

    /* Register 'print' function from the extensions */
    jerryx_handler_register_global ((const jerry_char_t *) "print",
                                    jerryx_handler_print);
    

    //create a js object : scheduleconfig
    jerry_value_t schedule_config_jerry = jerry_create_object();
    jerry_value_t schedule_name = jerry_create_string((const jerry_char_t *)"scheduleconfig");
    jerry_value_t schedule_ptr_value = jerry_create_number((double)((unsigned long)(&config)));
    jerry_release_value(jerry_set_property(schedule_config_jerry, schedule_name, schedule_ptr_value));
    jerry_release_value(schedule_name);
    jerry_release_value(schedule_ptr_value);


    //
    jerry_value_t global_object = jerry_get_global_object();
    jerry_value_t config_name_global_object = jerry_create_string((const jerry_char_t*)"config");
    jerry_release_value(jerry_set_property(global_object, config_name_global_object, schedule_config_jerry));
    jerry_release_value(config_name_global_object);
    
    register_js_op();
    int script_length = 0;
    for(auto file: model_files){
        const jerry_char_t *my_js_scripts = (jerry_char_t*)readJSFile(file.c_str(), &script_length);
        jerry_value_t val = jerry_eval(my_js_scripts,
                                        script_length,
                                        JERRY_PARSE_NO_OPTS);

        jerry_release_value(val);
        delete my_js_scripts;
    }

    jerry_cleanup ();
    return 0;
}
