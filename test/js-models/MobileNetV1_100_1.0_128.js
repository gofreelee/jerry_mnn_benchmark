var inputSize = 128;
var poolSize = inputSize / 32;
var channels = new Array(6);
var numclass = 100;
var MobileNet_100 = 32, MobileNet_075 = 24, MobileNet_050 = 16, MobileNet_025 = 8;
channels[0] = MobileNet_100; 
for(var i = 1; i < 6; ++i)
{
    channels[i] = channels[0] * (1 << i);
}

var inputShape = new Array(4);
inputShape[0] = 1, inputShape[1] = 3, inputShape[2] = inputSize, inputShape[3] = inputSize;
var x = Input(inputShape, "NC4HW4");

var channel = new Array(2);
channel[0] = 3, channel[1] = channels[0];

var kernelSize = new Array(2);
kernelSize[0] = 3, kernelSize[0] = 3;

var SAME = 2;

var stride = new Array(2);
stride[0] = 2, stride[1] = 2;

var dilate = new Array(2);
dilate[0] = 1, dilate[1] = 1;
x = Conv(0.0, 0.0, x, inputShape, channel, SAME, stride, dilate, 1);

var convBlockChannel = new Array(2);
convBlockChannel[0] = channels[0], convBlockChannel[1] = channels[1];
x = convBlock(x, convBlockChannel, 1);
convBlockChannel[0] = channels[1], convBlockChannel[1] = channels[2];
x = convBlock(x, convBlockChannel, 2);
convBlockChannel[0] = channels[2];
x = convBlock(x, convBlockChannel, 1);
convBlockChannel[1] = channels[3];
x = convBlock(x, convBlockChannel, 2);
convBlockChannel[0] = channels[3];
x = convBlock(x, convBlockChannel, 1);
convBlockChannel[1] = channels[4];
x = convBlock(x, convBlockChannel, 2);
convBlockChannel[0] = channels[4];
for(var i = 0; i < 5; ++i)
{
    x = convBlock(x, convBlockChannel, 1);
}
convBlockChannel[1] = channels[5];
x = convBlock(x, convBlockChannel, 2);
convBlockChannel[0] = channels[5];
x = convBlock(x, convBlockChannel, 1);

var avePoolKernel = new Array(2);
avePoolKernel[0] = poolSize, avePoolKernel[1] = poolSize;
var avePoolStride = new Array(2);
avePoolStride[0] = 1, avePoolStride[1] = 1;
var VALID = 1;
x = AvePool(x, avePoolKernel, avePoolStride, VALID);
inputShape[0] = channels[5], inputShape[1] = numclass;
channel[0] = 1, channel[1] = 1;
stride[0] = 1, stride[1] = 1;
x = Conv(0.0, 0.0, x, inputShape, channel, VALID, stride, dilate, 1);
x = SoftMax(x, -1);

var net = prepare(x, config);
var sum_time = 0;
var loop = 4
for(var i = 0; i < loop; ++i)
{
    var start = getCurrentTime();
    runnet(net);
    var end = getCurrentTime();
    sum_time += (end - start);
}
print("MobileNetV1_100_1.0_128 ：" + (sum_time / loop / 1000) + " ms");