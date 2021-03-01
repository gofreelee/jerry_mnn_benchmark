#include "OpInJs.h"
#include <string.h>
#include <vector>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <map>
using namespace MNN::Express;
#define BUFFER_SIZE 8

#define GET_VARP_PTR(tensor, input_ptr) \
        { \
        jerry_value_t tensor_name = jerry_create_string((const jerry_char_t *)"tensor"); \
        jerry_value_t tensor_v = jerry_get_property(tensor, tensor_name); \
        jerry_value_t jerry_num = jerry_value_to_number(tensor_v); \
        input_ptr = ( MNN::Express::VARP*)((unsigned long)jerry_get_number_value(jerry_num)); \
        jerry_release_value(jerry_num); \
        jerry_release_value(tensor_name);\
        }

#define GET_SCHEDULECONFIG_PTR(config, scheduleconfig_ptr) \
        {\
        jerry_value_t config_name = jerry_create_string((const jerry_char_t *)"scheduleconfig"); \
        jerry_value_t config_v = jerry_get_property(config, config_name); \
        jerry_value_t jerry_num = jerry_value_to_number(config_v); \
        scheduleconfig_ptr = ( MNN::ScheduleConfig*)((unsigned long)jerry_get_number_value(jerry_num)); \
        jerry_release_value(jerry_num); \
        jerry_release_value(config_name);\
        }

#define GET_PTR_FROM_JS_OBJECT(js_object, ptr_type, prop_name, ptr)\
        {\
        jerry_value_t prop_name = jerry_create_string((const jerry_char_t *)#prop_name); \
        jerry_value_t ptr_v = jerry_get_property(js_object, prop_name); \
        jerry_value_t jerry_num = jerry_value_to_number(ptr_v); \
        ptr = (ptr_type*)((unsigned long)jerry_get_number_value(jerry_num)); \
        jerry_release_value(jerry_num); \
        jerry_release_value(prop_name);\
        }


#define REGISTER_PTR_IN_JERRY(tensor, ptr, prop_name) \
        {\
        jerry_value_t prop_tensor = jerry_create_string((const jerry_char_t *)#prop_name); \
        jerry_value_t ptr_value = jerry_create_number((double)((unsigned long)ptr)); \
        jerry_release_value(jerry_set_property(tensor, prop_tensor, ptr_value)); \
        jerry_release_value(prop_tensor); \
        jerry_release_value(ptr_value);\
        }

#define JUDGE_DIMENSION_FORMAT(format, dimension_format) \
    if(strcmp((const char*)format, "NHWC") == 0) \
    { \
        dimension_format = MNN::Express::NHWC; \
    } \
    else if(strcmp((const char*)format, "NC4HW4") == 0) \
    { \
        dimension_format = MNN::Express::NC4HW4; \
    }\
    else if(strcmp((const char*)format, "NCHW") == 0)\
    {\
        dimension_format = MNN::Express::NCHW;\
    }\
    else\
    {\
        printf("format is error !\n");\
        return jerry_create_undefined();\
    } \

#define JSArray_TO_VECTOR(data_type, length, vector, shape) \
        for(int i = 0; i < length; ++i) \
        { \
            jerry_value_t array_value = jerry_get_property_by_index(shape, i);\
            jerry_value_t value2num = jerry_value_to_number(array_value);\
            vector.push_back((data_type)jerry_get_number_value(value2num));\
            jerry_release_value(array_value);\
            jerry_release_value(value2num);\
        }\

#define JSArray_TO_INTS(length, vector, shape) \
        JSArray_TO_VECTOR(int, length, vector, shape) 

static inline uint64_t getTimeInUs() {
    uint64_t time;
#if defined(_MSC_VER)
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    uint64_t sec = now.QuadPart / freq.QuadPart;
    uint64_t usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    time = sec * 1000000 + usec;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
    return time;
}


static MNN::Express::VARP residual(MNN::Express::VARP x, MNN::Express::INTS channels, int stride) {
    using namespace MNN;
    using namespace MNN::Express;
    int inputChannel = x->getInfo()->dim[1], outputChannel = channels[1];
    auto y = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {3, 3}, MNN::Express::SAME, {stride, stride}, {1, 1}, 1);
    y = _Conv(0.0f, 0.0f, y, {outputChannel, outputChannel}, {3, 3}, MNN::Express::SAME, {1, 1}, {1, 1}, 1);
    if (inputChannel != outputChannel || stride != 1) {
        x = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {1, 1}, MNN::Express::SAME, {stride, stride}, {1, 1}, 1);
    }
    y = _Add(x, y);
    return y;
}

static MNN::Express::VARP bottleNeck(MNN::Express::VARP x, MNN::Express::INTS channels, int stride) {
    using namespace MNN;
    using namespace MNN::Express;
    int inputChannel = x->getInfo()->dim[1], narrowChannel = channels[1], outputChannel = channels[2];
    auto y = _Conv(0.0f, 0.0f, x, {inputChannel, narrowChannel}, {1, 1}, SAME, {stride, stride}, {1, 1}, 1);
    y = _Conv(0.0f, 0.0f, y, {narrowChannel, narrowChannel}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
    y = _Conv(0.0f, 0.0f, y, {narrowChannel, outputChannel}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    if (inputChannel != outputChannel || stride != 1) {
        x = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {1, 1}, SAME, {stride, stride}, {1, 1}, 1);
    }
    y = _Add(x, y);
    return y;
}

static VARP convBlock(VARP x, INTS channels, int stride) {
    int inputChannel = channels[0], outputChannel = channels[1];
    int group = inputChannel;
    x = _Conv(0.0f, 0.0f, x, {inputChannel, inputChannel}, {3, 3}, SAME, {stride, stride}, {1, 1}, group);
    x = _Conv(0.0f, 0.0f, x, {inputChannel, outputChannel}, {1, 1}, SAME, {1, 1}, {1, 1}, 1);
    return x;
}

static VARP fireMoudle(VARP x, int inputChannel, int squeeze_1x1,
                       int expand_1x1, int expand_3x3) {
    x = _Conv(0.0f, 0.0f, x, {inputChannel, squeeze_1x1}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    auto y1 = _Conv(0.0f, 0.0f, x, {squeeze_1x1, expand_1x1}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    auto y2 = _Conv(0.0f, 0.0f, x, {squeeze_1x1, expand_3x3}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
    return _Concat({y1, y2}, 1); // concat on channel axis (NCHW)
}

static VARP inception(VARP x, int inputChannelSet, int channel_1x1,
                      int channel_3x3_reduce, int channel_3x3,
                      int channel_5x5_reduce, int channel_5x5,
                      int channel_pool) {
    auto inputChannel = x->getInfo()->dim[1];
    auto y1 = _Conv(0.0f, 0.0f, x, {inputChannel, channel_1x1}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    auto y2 = _Conv(0.0f, 0.0f, x, {inputChannel, channel_3x3_reduce}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    y2 = _Conv(0.0f, 0.0f, y2, {channel_3x3_reduce, channel_3x3}, {3, 3}, SAME, {1, 1}, {1, 1}, 1);
    auto y3 = _Conv(0.0f, 0.0f, x, {inputChannel, channel_5x5_reduce}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    y3 = _Conv(0.0f, 0.0f, y3, {channel_5x5_reduce, channel_5x5}, {5, 5}, SAME, {1, 1}, {1, 1}, 1);
    auto y4 = _MaxPool(x, {3, 3}, {1, 1}, SAME);
    y4 = _Conv(0.0f, 0.0f, y4, {inputChannel, channel_pool}, {1, 1}, VALID, {1, 1}, {1, 1}, 1);
    return _Concat({y1, y2, y3, y4}, 1); // concat on channel axis (NCHW)
}

void register_js_op()
{
    jerry_value_t global_object = jerry_get_global_object();
    bind_function(global_object, "Input", generate_input);
    bind_function(global_object, "test", test_ptr_convert);
    bind_function(global_object, "Relu", Relu_js); //op
    bind_function(global_object, "getCurrentTime", get_current_time);
    bind_function(global_object, "Convert", Convert_js); //op
    bind_function(global_object, "Conv", Conv_js);
    bind_function(global_object, "MaxPool", Maxpool_js);
    bind_function(global_object, "ResiduBlock", ResidualBlock_js);
    bind_function(global_object, "AvePool", AvePool_js);
    bind_function(global_object, "SoftMax", Softmax_js);
    bind_function(global_object, "BottleNeckBlock", bottleNeckBlock_js);
    bind_function(global_object, "prepare", runnet_prepare);
    bind_function(global_object, "runnet", runnet);
    bind_function(global_object, "getVARPInfo", getVARPInfo);
    bind_function(global_object, "convBlock", convBlock_js);
    bind_function(global_object, "fireMoudle", fireMoudle_js);
    bind_function(global_object, "inception", inception_js);
    return;
}


//给obj 绑定一个函数
void bind_function(jerry_value_t obj, const char* function_name, jerry_external_handler_t handler_p)
{
    jerry_value_t func_name = jerry_create_string((const jerry_char_t *)function_name);
    jerry_value_t function_value = jerry_create_external_function(handler_p);
    jerry_release_value(jerry_set_property(obj, func_name, function_value));
    jerry_release_value(func_name);
    jerry_release_value(function_value);
}

jerry_value_t generate_input(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    /*参数应该有： array(表示type) ， dataformat*/
    if(args_cnt != 2)
    {
        printf("paramater is not enough ! \n");
        return jerry_create_undefined();
    }

    jerry_value_t shape = args_p[0];   //获取array对象， 表示输入的shape
    jerry_value_t data_format = jerry_value_to_string(args_p[1]); 

    int shape_length = jerry_get_array_length(shape);

    //_input()需要的两个参数，先构造出vector<int>
    std::vector<int> shape_vector;
    
    for(int i = 0; i < shape_length; ++i)
    {
        jerry_value_t array_value = jerry_get_property_by_index(shape, i);
        jerry_value_t value2num = jerry_value_to_number(array_value);
        shape_vector.push_back((int)jerry_get_number_value(value2num));
        jerry_release_value(array_value);
        jerry_release_value(value2num);
    }
    
    //构造出format字符串
    jerry_char_t format[BUFFER_SIZE];
    jerry_size_t copied_bytes = jerry_string_to_utf8_char_buffer(data_format, format, sizeof(format) - 1);
    format[copied_bytes] = '\0';

    MNN::Express::Dimensionformat dimension_format;
    if(strcmp((const char*)format, "NHWC") == 0)
    {
        dimension_format = MNN::Express::NHWC;
    }
    else if(strcmp((const char*)format, "NC4HW4") == 0)
    {
        dimension_format = MNN::Express::NC4HW4;

    }
    else if(strcmp((const char*)format, "NCHW") == 0)
    {
        dimension_format = MNN::Express::NCHW;
    }
    else
    {
        printf("format is error !\n");
        return jerry_create_undefined();
    }

    MNN::Express::VARP* input_ptr = new MNN::Express::VARP(MNN::Express::_Input(shape_vector, dimension_format));
    
    // printf("%d\n", (unsigned long)input_ptr);
    jerry_value_t input = jerry_create_object();
    jerry_value_t prop_tensor = jerry_create_string((const jerry_char_t *)"tensor");
    jerry_value_t ptr_value = jerry_create_number((double)((unsigned long)input_ptr));

    jerry_release_value(jerry_set_property(input, prop_tensor, ptr_value));
    jerry_release_value(prop_tensor);
    jerry_release_value(ptr_value);
    jerry_release_value(data_format);

    return input;
}

jerry_value_t test_ptr_convert(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    jerry_value_t jerry_num = jerry_value_to_number(args_p[0]);  
    int num =  (unsigned long)jerry_get_number_value(args_p[0]);
    printf("%d\n", num);
   // jerry_release_value(jerry_num);
    return jerry_create_undefined();
}

jerry_value_t Relu_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* input_ptr = 0;
    GET_VARP_PTR(args_p[0],input_ptr);
    jerry_value_t slope_jerry_num = jerry_value_to_number(args_p[1]);
    double slope = jerry_get_number_value(slope_jerry_num);
    // printf("the slope is %f\n", slope);
    // printf("the ptr is %d \n", (unsigned long)input_ptr);
    MNN::Express::VARP* output_ptr = new MNN::Express::VARP(MNN::Express::_Relu(*input_ptr, slope));
    
    // printf("%d\n", (unsigned long)output_ptr);
    jerry_value_t output = jerry_create_object();
    jerry_value_t prop_tensor = jerry_create_string((const jerry_char_t *)"tensor");
    jerry_value_t ptr_value = jerry_create_number((double)((unsigned long)output_ptr));

    jerry_release_value(jerry_set_property(output, prop_tensor, ptr_value));
    jerry_release_value(prop_tensor);
    jerry_release_value(ptr_value);
    jerry_release_value(slope_jerry_num);
    return output;
}

jerry_value_t get_current_time(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long curr_time =  static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
    // printf("%ld time is current time \n", curr_time);
    return jerry_create_number(curr_time);
}

jerry_value_t Convert_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* input_ptr = 0;
    GET_VARP_PTR(args_p[0], input_ptr)
    
    jerry_value_t data_format = jerry_value_to_string(args_p[1]); 
    jerry_char_t format[BUFFER_SIZE];
    jerry_size_t copied_bytes = jerry_string_to_utf8_char_buffer(data_format, format, sizeof(format) - 1);
    format[copied_bytes] = '\0';
    MNN::Express::Dimensionformat dimension_format;
    JUDGE_DIMENSION_FORMAT(format, dimension_format)
    
    MNN::Express::VARP* output_ptr = new MNN::Express::VARP
    (MNN::Express::_Convert(*input_ptr, dimension_format));
    jerry_value_t output_tensor = jerry_create_object();
    REGISTER_PTR_IN_JERRY(output_tensor, output_ptr,tensor)
    return output_tensor;
}


//float weight, float bias, MNN::Express::VARP x, 
//MNN::Express::INTS channel, MNN::Express::INTS kernelSize, 
//MNN::Express::PaddingMode pad = MNN::Express::VALID, MNN::Express::INTS stride = {1, 1}, 
//MNN::Express::INTS dilate = {1, 1}, int group = 1
jerry_value_t Conv_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    float weight =  (float)jerry_get_number_value(args_p[0]); // float weight
    float bias   =  (float)jerry_get_number_value(args_p[1]); // float bias

    MNN::Express::VARP* x = 0;
    GET_VARP_PTR(args_p[2], x); //VARP x

    MNN::Express::INTS channel;
    int channel_length = jerry_get_array_length(args_p[3]);
    JSArray_TO_INTS(channel_length, channel, args_p[3]) // INTS channel

    MNN::Express::INTS kernelSize;
    int kernelSize_length = jerry_get_array_length(args_p[4]);
    JSArray_TO_INTS(kernelSize_length, kernelSize, args_p[4]) // INTS kernelSize
 
    enum MNN::Express::PaddingMode pad = (MNN::Express::PaddingMode)((unsigned long)jerry_get_number_value(args_p[5])); 

    MNN::Express::INTS stride;
    int stride_length = jerry_get_array_length(args_p[6]);
    JSArray_TO_INTS(stride_length, stride, args_p[6]) // INTS stride

    MNN::Express::INTS dilate;
    int dilate_length = jerry_get_array_length(args_p[7]);
    JSArray_TO_INTS(dilate_length, dilate, args_p[7]);

    int group = (int)((unsigned long)jerry_get_number_value(args_p[8]));

    MNN::Express::VARP* conv_ptr =  new MNN::Express::VARP
    (MNN::Express::_Conv(weight, bias, *x, channel, kernelSize, pad, stride, dilate, group));
    jerry_value_t conv_tensor = jerry_create_object();
    REGISTER_PTR_IN_JERRY(conv_tensor, conv_ptr, tensor)
    return conv_tensor;
}

// MNN::Express::VARP MNN::Express::_MaxPool(MNN::Express::VARP x, MNN::Express::INTS kernel, MNN::Express::INTS stride = {1, 1}
// , MNN::Express::PaddingMode pad = MNN::Express::VALID, MNN::Express::INTS pads = {0, 0})

jerry_value_t Maxpool_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* x = 0;
    GET_VARP_PTR(args_p[0], x)

    MNN::Express::INTS kernel;
    int kernel_length = jerry_get_array_length(args_p[1]);
    JSArray_TO_INTS(kernel_length, kernel, args_p[1]) // INTS kernelSize

    MNN::Express::INTS stride;
    int stride_length = jerry_get_array_length(args_p[2]);
    JSArray_TO_INTS(stride_length, stride, args_p[2]) // INTS stride

    enum MNN::Express::PaddingMode pad = (MNN::Express::PaddingMode)((unsigned long)jerry_get_number_value(args_p[3]));

    MNN::Express::INTS pads;
    int pads_length = jerry_get_array_length(args_p[4]);
    JSArray_TO_INTS(pads_length, stride, args_p[4]) // INTS stride 

    MNN::Express::VARP* maxpool_ptr =  new MNN::Express::VARP
    (MNN::Express::_MaxPool(*x, kernel, stride, pad, pads));
    jerry_value_t maxpool_tensor = jerry_create_object();
    REGISTER_PTR_IN_JERRY(maxpool_tensor, maxpool_ptr, tensor)

    return maxpool_tensor;
}


//  residualBlock(VARP x, INTS channels, int stride, int number)
jerry_value_t ResidualBlock_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* x = 0;
    GET_VARP_PTR(args_p[0], x)

    MNN::Express::INTS channel;
    int channel_length = jerry_get_array_length(args_p[1]);
    JSArray_TO_INTS(channel_length, channel, args_p[1]) // INTS channel


    int stride = (int)((unsigned long)jerry_get_number_value(args_p[2]));
    int number = (int)((unsigned long)jerry_get_number_value(args_p[3]));

    *x = residual(*x, {channel[0], channel[1]}, stride);
    for(int i = 1; i < number; ++i)
    {
        *x = residual(*x, {channel[0], channel[1]}, 1);
    }
    MNN::Express::VARP* residual_ptr = new MNN::Express::VARP(
        std::move(*x)
    );
    jerry_value_t residual_tensor = jerry_create_object();
    REGISTER_PTR_IN_JERRY(residual_tensor, residual_ptr, tensor);
    return residual_tensor;
}

jerry_value_t bottleNeckBlock_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    
    
    MNN::Express::VARP* x = 0;
    GET_VARP_PTR(args_p[0], x)

    MNN::Express::INTS channel;
    int channel_length = jerry_get_array_length(args_p[1]);
    JSArray_TO_INTS(channel_length, channel, args_p[1]) // INTS channel


    int stride = (int)((unsigned long)jerry_get_number_value(args_p[2]));
    int number = (int)((unsigned long)jerry_get_number_value(args_p[3]));
    
    *x = bottleNeck(*x, {channel[0], channel[1], channel[2]}, stride);
    for(int i = 1; i < number; ++i)
    {
        *x = bottleNeck(*x, {channel[2], channel[1], channel[2]}, 1);
    }
    MNN::Express::VARP* bottle_neck_ptr = new MNN::Express::VARP(
        std::move(*x)
    );
    jerry_value_t bottle_neck_tensor = jerry_create_object();
    REGISTER_PTR_IN_JERRY(bottle_neck_tensor,bottle_neck_ptr, tensor);
    return bottle_neck_tensor;
}


// MNN::Express::VARP MNN::Express::_AvePool(MNN::Express::VARP x, MNN::Express::INTS kernel, MNN::Express::INTS 
// stride = {1, 1}, MNN::Express::PaddingMode pad = MNN::Express::VALID, MNN::Express::INTS pads = {0, 0})

jerry_value_t AvePool_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* x = 0;
    GET_VARP_PTR(args_p[0], x)

    MNN::Express::INTS kernel;
    int kernel_length = jerry_get_array_length(args_p[1]);
    JSArray_TO_INTS(kernel_length, kernel, args_p[1]) // INTS kernelSize

    MNN::Express::INTS stride;
    int stride_length = jerry_get_array_length(args_p[2]);
    JSArray_TO_INTS(stride_length, stride, args_p[2]) // INTS stride

    enum MNN::Express::PaddingMode pad = (MNN::Express::PaddingMode)((unsigned long)jerry_get_number_value(args_p[3]));

    MNN::Express::INTS pads;
    int pads_length = jerry_get_array_length(args_p[4]);
    JSArray_TO_INTS(pads_length, stride, args_p[4]) // INTS stride 

    MNN::Express::VARP* avepool_ptr =  new MNN::Express::VARP
    (MNN::Express::_AvePool(*x, kernel, stride, pad, pads));
    jerry_value_t avepool_tensor = jerry_create_object();
    REGISTER_PTR_IN_JERRY(avepool_tensor, avepool_ptr, tensor)

    return avepool_tensor;
}

jerry_value_t Softmax_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
   
    MNN::Express::VARP* x = 0;
    GET_VARP_PTR(args_p[0], x)
    
   
    int axis = (int)((signed long)jerry_get_number_value(args_p[1]));
    // printf("%d\n", axis);
    // fflush(stdout);
    MNN::Express::VARP* softmax_ptr =  new MNN::Express::VARP
    (MNN::Express::_Softmax(*x, axis));
    jerry_value_t softmax_tensor = jerry_create_object();
    REGISTER_PTR_IN_JERRY(softmax_tensor, softmax_ptr, tensor)
    return softmax_tensor;
}

jerry_value_t runnet_prepare(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    using namespace MNN;
    using namespace MNN::Express;
    MNN::Express::VARP* netOutput = 0;
    GET_VARP_PTR(args_p[0], netOutput)
  
    MNN::ScheduleConfig* config = 0;
    GET_SCHEDULECONFIG_PTR(args_p[1], config)
 
    std::unique_ptr<NetT> netTable(new NetT);
    Variable::save({*netOutput}, netTable.get());
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = CreateNet(builder, netTable.get());
    builder.Finish(offset);
    const void* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();
    std::unique_ptr<Interpreter> net(Interpreter::createFromBuffer(buf, size));
    net->setSessionMode(MNN::Interpreter::Session_Release);
    auto session = net->createSession(*config);
    net->releaseModel();
    auto inputTensor = net->getSessionInput(session, NULL);
    std::shared_ptr<Tensor> inputTensorHost(Tensor::createHostTensorFromDevice(inputTensor, false));
    int eleSize = inputTensorHost->elementSize();
    for (int i = 0; i < eleSize; ++i) {
        inputTensorHost->host<float>()[i] = 0.0f;
    }
    auto outputTensor = net->getSessionOutput(session, NULL);
    std::shared_ptr<Tensor> outputTensorHost(Tensor::createHostTensorFromDevice(outputTensor, false));
   

    // Warming up...
    for (int i = 0; i < 3; ++i) {
        inputTensor->copyFromHostTensor(inputTensorHost.get());
        net->runSession(session);
        outputTensor->copyToHostTensor(outputTensorHost.get());
    }

    jerry_value_t runnable_net = jerry_create_object();
    REGISTER_PTR_IN_JERRY(runnable_net, inputTensor, inputTensor)
    REGISTER_PTR_IN_JERRY(runnable_net, outputTensor, outputTensor)

    std::shared_ptr<Tensor> *inputTensorHost_ptr = new std::shared_ptr<Tensor>(std::move(inputTensorHost));
    std::shared_ptr<Tensor> *outputTensorHost_ptr = new std::shared_ptr<Tensor>(std::move(outputTensorHost));

    REGISTER_PTR_IN_JERRY(runnable_net, inputTensorHost_ptr, inputTensorHost)
    REGISTER_PTR_IN_JERRY(runnable_net, outputTensorHost_ptr, outputTensorHost)

    std::unique_ptr<Interpreter> * net_ptr = new std::unique_ptr<Interpreter>(std::move(net));

    REGISTER_PTR_IN_JERRY(runnable_net, net_ptr, net)
    REGISTER_PTR_IN_JERRY(runnable_net, session, session)
    
    return runnable_net;

}

jerry_value_t runnet(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    using namespace MNN;
    using namespace MNN::Express;
    jerry_value_t runnable_net = args_p[0];

    Tensor* inputTensor_ptr = 0;
    GET_PTR_FROM_JS_OBJECT(runnable_net, MNN::Tensor, inputTensor, inputTensor_ptr)
    Tensor* outputTensor_ptr = 0;
    GET_PTR_FROM_JS_OBJECT(runnable_net, MNN::Tensor, outputTensor, outputTensor_ptr)

    std::shared_ptr<Tensor> *inputTensorHost_ptr = 0;
    std::shared_ptr<Tensor> *outputTensorHost_ptr = 0;
    GET_PTR_FROM_JS_OBJECT(runnable_net, std::shared_ptr<Tensor>, inputTensorHost, inputTensorHost_ptr)
    GET_PTR_FROM_JS_OBJECT(runnable_net, std::shared_ptr<Tensor>, outputTensorHost, outputTensorHost_ptr)

    std::unique_ptr<Interpreter> * net_ptr = 0;
    GET_PTR_FROM_JS_OBJECT(runnable_net, std::unique_ptr<Interpreter>, net, net_ptr)

    Session *session_ptr = 0;
    GET_PTR_FROM_JS_OBJECT(runnable_net, MNN::Session, session, session_ptr)
    inputTensor_ptr->copyFromHostTensor(inputTensorHost_ptr->get());
    (*net_ptr)->runSession(session_ptr);
    outputTensor_ptr->copyToHostTensor(outputTensorHost_ptr->get());
    return  jerry_create_undefined();
}

jerry_value_t getVARPInfo(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* x = 0;
    GET_VARP_PTR(args_p[0], x)

    int index = (int)((unsigned long)jerry_get_number_value(args_p[1]));
    int data = (*x)->getInfo()->dim[index];
    return jerry_create_number(data);
}

char* readJSFile(const char* file_path, int *file_length)
{
    int fd;
    while((fd = open(file_path, O_RDONLY)) == -1 && errno == EINTR);
    if(fd == -1)
    {
        printf("open js file error \n");
        return nullptr;
    }

    struct stat jsStat;
    if(fstat(fd, &jsStat) == -1)
    {
        printf("fstat error \n");
        return nullptr;
    }
    int bufSize = jsStat.st_size;
    *file_length = bufSize;
    char * jsScript = (char *)malloc((bufSize + 1)* sizeof(char *));
    int readLeft = bufSize, readLength = 0;
    while(readLeft > 0)
    {
        while((readLength = read(fd,jsScript, readLeft)) == -1 && errno == EINTR);
        if(readLength == -1)
        {
            printf("read js file error \n");
            free(jsScript);
            return nullptr;
        }
        readLeft -= readLength;
    }
    jsScript[bufSize] = '\0';
    return jsScript;
}

jerry_value_t convBlock_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* x = 0;
    GET_VARP_PTR(args_p[0], x)

    MNN::Express::INTS channels;
    int channel_length = jerry_get_array_length(args_p[1]);
    JSArray_TO_INTS(channel_length, channels, args_p[1]) // INTS channel


    int stride = (int)((unsigned long)jerry_get_number_value(args_p[2]));
    *x = convBlock(*x, channels, stride);
    MNN::Express::VARP* convBlock_ptr = new MNN::Express::VARP(
        std::move(*x)
    ); 
    jerry_value_t convBlock_tensor = jerry_create_object();
    REGISTER_PTR_IN_JERRY(convBlock_tensor, convBlock_ptr, tensor);
    return convBlock_tensor;  
}

jerry_value_t fireMoudle_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* x = 0;
    GET_VARP_PTR(args_p[0], x)

    int inputChannel = (int)((unsigned long)jerry_get_number_value(args_p[1]));
    int squeeze_1x1  = (int)((unsigned long)jerry_get_number_value(args_p[2]));
    int expand_1x1   = (int)((unsigned long)jerry_get_number_value(args_p[3]));
    int expand_3x3   = (int)((unsigned long)jerry_get_number_value(args_p[4]));
    *x = fireMoudle(*x, inputChannel, squeeze_1x1, expand_1x1, expand_3x3);
    MNN::Express::VARP* fireMoudle_ptr = new MNN::Express::VARP(
        std::move(*x)
    );
    jerry_value_t fireMoudle_tensor = jerry_create_object();
    REGISTER_PTR_IN_JERRY(fireMoudle_tensor, fireMoudle_ptr, tensor);
    return fireMoudle_tensor;
}

jerry_value_t inception_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* x = 0;
    GET_VARP_PTR(args_p[0], x)

    int inputChannelSet = (int)((unsigned long)jerry_get_number_value(args_p[1]));
    int channel_1x1 = (int)((unsigned long)jerry_get_number_value(args_p[2]));
    int channel_3x3_reduce = (int)((unsigned long)jerry_get_number_value(args_p[3]));
    int channel_3x3 = (int)((unsigned long)jerry_get_number_value(args_p[4]));
    int channel_5x5_reduce = (int)((unsigned long)jerry_get_number_value(args_p[5]));
    int channel_5x5 = (int)((unsigned long)jerry_get_number_value(args_p[6]));
    int channel_pool = (int)((unsigned long)jerry_get_number_value(args_p[7]));
    *x = inception(*x, inputChannelSet, channel_1x1, channel_3x3_reduce, channel_3x3,
    channel_5x5_reduce, channel_5x5, channel_pool);

    MNN::Express::VARP* inception_ptr = new MNN::Express::VARP(
        std::move(*x)
    );
    jerry_value_t inception_tensor = jerry_create_object();
    REGISTER_PTR_IN_JERRY(inception_tensor, inception_ptr, tensor);
    return inception_tensor;
}