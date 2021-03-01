var inputShape = new Array(4);
inputShape[0] = 1, inputShape[1] = 3, inputShape[2] = 224, inputShape[3] = 224;
var x = Input(inputShape, 'NC4HW4');

var channel = new Array(2);
channel[0] = 3, channel[1] = 96;

var kernelSize = new Array(2);
kernelSize[0] = 7, kernelSize[1] = 7;

var SAME = 2;

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
x = fireMoudle(x, 96, 16, 64, 64);
x = fireMoudle(x, 128, 16, 64, 64);
x = fireMoudle(x, 128, 32, 128, 128);
x = MaxPool(x, maxPoolKernel, maxPoolStride, SAME);
x = fireMoudle(x, 256, 32, 128, 128);
x = fireMoudle(x, 256, 48, 192, 192);
x = fireMoudle(x, 384, 48, 192, 192);
x = fireMoudle(x, 384, 64, 256, 256);
x = MaxPool(x, maxPoolKernel, maxPoolStride, SAME);
x = fireMoudle(x, 512, 64, 256, 256);
channel[0] = 512, channel[1] = 100;
kernelSize[0] = 1, kernelSize[1] = 1;
var VALID = 1;
stride[0] = 1, stride[1] = 1;
x = Conv(0.0, 0.0, x, channel, kernelSize, VALID, stride, dilate, 1)
kernelSize[0] = 14, kernelSize[1] = 14;
x = AvePool(x, kernelSize, stride, VALID);

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
print("SqueezeNet_100 ：" + (sum_time / loop / 1000) + " ms");