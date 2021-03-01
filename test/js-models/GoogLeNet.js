var inputShape = new Array(4);
inputShape[0] = 1, inputShape[1] = 3, inputShape[2] = 224, inputShape[3] = 224;
var x = Input(inputShape, 'NC4HW4');

var channel = new Array(2);
channel[0] = 3, channel[1] = 64;

var kernelSize = new Array(2);
kernelSize[0] = 7, kernelSize[1] = 7;

var SAME = 2, VALID = 1;

var stride = new Array(2);
stride[0] = 2, stride[1] = 2;

var dilate = new Array(2);
dilate[0] = 1, dilate[1] = 1;

x = Conv(0.0, 0.0, x, channel, kernelSize, SAME, stride, dilate, 1);

var maxPoolKernel = new Array(2);
maxPoolKernel[0] = 3, maxPoolKernel[1] = 3;

var maxPoolStride = new Array(2);
maxPoolStride[0] = 2, maxPoolStride[1] = 2;
x = MaxPool(x, maxPoolKernel, maxPoolStride, SAME);

channel[0] = 64, channel[1] = 192;
kernelSize[0] = 3, kernelSize[1] = 3;
stride[0] = 1, stride[1] = 1;

x = Conv(0.0, 0.0, x, channel, kernelSize, SAME, stride, dilate, 1);

x = MaxPool(x, maxPoolKernel, maxPoolStride, SAME);

x = inception(x, 192, 64, 96, 128, 16, 32, 32);

x = inception(x, 256, 128, 128, 192, 32, 96, 64);

x = MaxPool(x, maxPoolKernel, maxPoolStride, SAME);
x = inception(x, 480, 192, 96, 208, 16, 48, 64);
x = inception(x, 512, 160, 112, 224, 24, 64, 64);
x = inception(x, 512, 128, 128, 256, 24, 64, 64);
x = inception(x, 512, 112, 144, 288, 32, 64, 64);
x = inception(x, 512, 256, 160, 320, 32, 128, 128);
x = MaxPool(x, maxPoolKernel, maxPoolStride, SAME);
x = inception(x, 832, 256, 160, 320, 32, 128, 128);
x = inception(x, 832, 384, 192, 384, 48, 128, 128);

kernelSize[0] = 7, kernelSize[1] = 7;
stride[0] = 1, stride[1] = 1;

x = AvePool(x, kernelSize, stride, VALID);

channel[0] = 1024, channel[1] = 100;
kernelSize[0] = 1, kernelSize[1] = 1;
x = Conv(0.0, 0.0, x, channel, kernelSize, VALID, stride, dilate, 1);

x = SoftMax(x, -1);

var net = prepare(x, config);
var sum_time = 0;
var loop = 4;

for(var i = 0; i < loop; ++i)
{
    var start = getCurrentTime();
    runnet(net);
    var end = getCurrentTime();
    sum_time += (end - start);
}
print("GoogLeNet_100 ：" + (sum_time / loop / 1000) + " ms");
