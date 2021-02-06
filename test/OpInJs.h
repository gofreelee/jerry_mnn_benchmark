#ifndef OPINJS_H_
#define OPINJS_H_
#include <sys/time.h>
#include <include/jerryscript.h>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>

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

jerry_value_t generate_conv_input(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);

jerry_value_t generate_conv_filter(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt);
            
jerry_value_t generate_conv_bias(const jerry_value_t func_value, /**< function object */
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

#endif