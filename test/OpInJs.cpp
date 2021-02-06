#include "OpInJs.h"
#include <string.h>
#include <vector> 
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


#define REGISTER_PTR_IN_JERRY(tensor, ptr) \
        {\
        jerry_value_t prop_tensor = jerry_create_string((const jerry_char_t *)"tensor"); \
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


const int inputHeight = 224, inputWidth = 224, inputChannel = 2, outputChannel = 1;
const int kernelSize = 3, stride = 2, pad = 1, batch = 1;
const int height  = (inputHeight + 2 * pad - kernelSize) / stride + 1; // height = 3
const int width   = (inputWidth + 2 * pad - kernelSize) / stride + 1;  // width = 3

const std::vector<float> inputDataHelper = {
// channel 0
0.6345, 0.1219, 0.0424, 0.0501, 0.3934, 0.4311, 0.5961, 0.6642, 0.734, 0.062, 0.88, 0.503, 0.1638,
0.6367, 0.2151, 0.0795, 0.7693, 0.134, 0.4963, 0.7571, 0.5428, 0.3663, 0.2823, 0.7478, 0.579,
0.6345, 0.1219, 0.0424, 0.0501, 0.3934, 0.4311, 0.5961, 0.6642, 0.734, 0.062, 0.88, 0.503, 0.1638,
0.6367, 0.2151, 0.0795, 0.7693, 0.134, 0.4963, 0.7571, 0.5428, 0.3663, 0.2823, 0.7478, 0.579,
0.6345, 0.1219, 0.0424, 0.0501, 0.3934, 0.4311, 0.5961, 0.6642, 0.734, 0.062, 0.88, 0.503, 0.1638,
0.6367, 0.2151, 0.0795, 0.7693, 0.134, 0.4963, 0.7571, 0.5428, 0.3663, 0.2823, 0.7478, 0.579,
0.6345, 0.1219, 0.0424, 0.0501, 0.3934, 0.4311, 0.5961, 0.6642, 0.734, 0.062, 0.88, 0.503, 0.1638,
0.6367, 0.2151, 0.0795, 0.7693, 0.134, 0.4963, 0.7571, 0.5428, 0.3663, 0.2823, 0.7478, 0.579,
// channel 1
0.6917, 0.4047, 0.9673, 0.9111, 0.608, 0.4621, 0.6567, 0.3192, 0.726, 0.9066, 0.885, 0.3491, 0.7938,
0.2593, 0.3146, 0.6901, 0.2126, 0.649, 0.7919, 0.9838, 0.0672, 0.0357, 0.383, 0.5043, 0.2803,
0.6917, 0.4047, 0.9673, 0.9111, 0.608, 0.4621, 0.6567, 0.3192, 0.726, 0.9066, 0.885, 0.3491, 0.7938,
0.2593, 0.3146, 0.6901, 0.2126, 0.649, 0.7919, 0.9838, 0.0672, 0.0357, 0.383, 0.5043, 0.2803,
0.6917, 0.4047, 0.9673, 0.9111, 0.608, 0.4621, 0.6567, 0.3192, 0.726, 0.9066, 0.885, 0.3491, 0.7938,
0.2593, 0.3146, 0.6901, 0.2126, 0.649, 0.7919, 0.9838, 0.0672, 0.0357, 0.383, 0.5043, 0.2803,
0.6917, 0.4047, 0.9673, 0.9111, 0.608, 0.4621, 0.6567, 0.3192, 0.726, 0.9066, 0.885, 0.3491, 0.7938,
0.2593, 0.3146, 0.6901, 0.2126, 0.649, 0.7919, 0.9838, 0.0672, 0.0357, 0.383, 0.5043, 0.2803,
};

std::vector<float> inputData(10000,0);


const std::vector<float> filterData = {
// outputChannel = 0, inputChannel = 0
0.5567, 0.4559, 0.0203, 0.9659, 0.2679, 0.4117, 0.9696, 0.4567, 0.3787,
// outputChannel = 0, inputChannel = 1
0.3354, 0.2056, 0.0342, 0.023, 0.4683, 0.9966, 0.6097, 0.0873, 0.7917};
const std::vector<float> biasData   = {1.0};
const std::vector<float> outputData = {2.930293, 4.682340, 2.721255, 3.087505, 5.198602,
                            4.088373, 1.564287, 3.151330, 3.109602};

void generate_input_data()
{
    for(int i = 0; i < 224*224; ++i)
        inputData[i] = inputData[i % 200];
}

 
void register_js_op()
{
    jerry_value_t global_object = jerry_get_global_object();
    bind_function(global_object, "Input", generate_input);
    bind_function(global_object,"test", test_ptr_convert);
    bind_function(global_object, "Relu", Relu_js); //op
    bind_function(global_object, "getCurrentTime", get_current_time);
    bind_function(global_object, "convInput", generate_conv_input);
    bind_function(global_object, "convFilter", generate_conv_filter);
    bind_function(global_object, "convBias", generate_conv_bias);
    bind_function(global_object, "Convert", Convert_js); //op
    bind_function(global_object, "Conv", Conv_js);
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
    
    printf("%d\n", (unsigned long)input_ptr);
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
    unsigned long curr_time =  tv.tv_sec * 1000 + tv.tv_usec / 1000 ;
    // printf("%ld time is current time \n", curr_time);
    return jerry_create_number(curr_time);
}

//卷积input的生成
jerry_value_t generate_conv_input(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* conv_input_ptr = new MNN::Express::VARP
    (MNN::Express::_Input({batch, inputChannel, inputHeight, inputWidth}, MNN::Express::NCHW, halide_type_of<float>()));
    jerry_value_t tensor = jerry_create_object();
    REGISTER_PTR_IN_JERRY(tensor, conv_input_ptr)
    ::memcpy((*conv_input_ptr)->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));
    printf("inputdata size is %d\n", inputData.size());
    return tensor;
}

//卷积filter的生成
jerry_value_t generate_conv_filter(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* conv_filter_ptr = new MNN::Express::VARP
    (MNN::Express:: _Input({outputChannel, inputChannel, kernelSize, kernelSize}, MNN::Express::NCHW, halide_type_of<float>()));
    jerry_value_t filter = jerry_create_object();
    REGISTER_PTR_IN_JERRY(filter, conv_filter_ptr)
    ::memcpy((*conv_filter_ptr)->writeMap<float>(), filterData.data(), filterData.size() * sizeof(float));

    return filter;
}

jerry_value_t generate_conv_bias(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* conv_bias_ptr = new MNN::Express::VARP
    (MNN::Express::_Input({outputChannel}, MNN::Express::NCHW, halide_type_of<float>()));
    jerry_value_t bias = jerry_create_object();
    REGISTER_PTR_IN_JERRY(bias, conv_bias_ptr)
    ::memcpy((*conv_bias_ptr)->writeMap<float>(), biasData.data(), biasData.size() * sizeof(float));
    return bias;
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
    REGISTER_PTR_IN_JERRY(output_tensor, output_ptr)
    return output_tensor;
}

jerry_value_t Conv_js(const jerry_value_t func_value, /**< function object */
             const jerry_value_t this_val, /**< this arg */
             const jerry_value_t args_p[], /**< function arguments */
             const jerry_length_t args_cnt)
{
    MNN::Express::VARP* filter = 0, *bias = 0 , *input = 0;
    GET_VARP_PTR(args_p[0], filter)
    GET_VARP_PTR(args_p[1], bias)
    GET_VARP_PTR(args_p[2], input)
    MNN::Express::VARP* conv_result = new MNN::Express::VARP
    (MNN::Express::_Conv(*filter, *bias, *input, MNN::Express::CAFFE,
    {stride, stride}, {1, 1}, 2, {pad, pad}));
    jerry_value_t conv_result_tensor = jerry_create_object();
    REGISTER_PTR_IN_JERRY(conv_result_tensor, conv_result_tensor);
    (*conv_result)->readMap<float>();
    return conv_result_tensor;
}
