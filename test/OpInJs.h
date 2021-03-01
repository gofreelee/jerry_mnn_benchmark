#ifndef OPINJS_H_
#define OPINJS_H_
#include <sys/time.h>
#include <include/jerryscript.h>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNN_generated.h"
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>

void register_js_op(); //js 算子绑定

void bind_function(jerry_value_t obj, const char* function_name, jerry_external_handler_t handler_p);

void generate_input_data();


jerry_value_t get_current_time(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t generate_input(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t test_ptr_convert(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t Relu_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);



jerry_value_t Convert_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t Conv_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);


jerry_value_t Maxpool_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);


jerry_value_t ResidualBlock_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t AvePool_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t Softmax_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t runnet_prepare(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t runnet(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t bottleNeckBlock_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t convBlock_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t fireMoudle_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t getVARPInfo(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t inception_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);
char* readJSFile(const char* file_path, int *file_length);

#endif