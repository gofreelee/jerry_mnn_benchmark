var NC4HW4 = 1, SAME = 2, VALID = 1;
var numClass = 1000;
var resNet18 = new Array(4);
for(var i = 0; i < 4; ++i)
{
    resNet18[i] = 2;
}

var resNet34 = new Array(4);
resNet34[0] = 3, resNet34[1] = 4, resNet34[2] = 6, resNet34[3] = 3;

var resNet50 = new Array(4);
resNet50[0] = 3, resNet50[1] = 4, resNet50[2] = 6, resNet50[3] = 3;

var resNet101 = new Array(4);
resNet101[0] = 3, resNet101[1] = 4, resNet101[2] = 23, resNet101[3] = 3;


var resNet152 = new Array(4);
resNet152[0] = 3, resNet152[1] = 8, resNet152[2] = 36, resNet152[3] = 3;

var channels = new Array(5);
channels[0] = 64, channels[1] = 64, channels[2] = 128, channels[3] = 256, channels[4] = 512;

var strides = new Array(4);
strides[0] = 1, strides[1] = 2, strides[2] = 2, strides[3] = 2;
var finalChannel = channels[4] * 4;


var inputFormat = new Array(4);
inputFormat[0] = 1, inputFormat[1] = 3, inputFormat[2] = 224, inputFormat[3] = 224;

var x = Input(inputFormat, 'NC4HW4');

var convChannel = new Array(2);
convChannel[0] = 3, convChannel[1] = 64;

var convKernelSize = new Array(2);
convKernelSize[0] = 7, convKernelSize[1] = 7;
var convStride = new Array(2);
convStride[0] = 2, convStride[1] = 2;
var convDilate = new Array(2);
convDilate[0] = 1, convDilate[1] = 1;

x = Conv(0.0, 0.0, x, convChannel, convKernelSize, SAME, convStride, convDilate, 1);
var maxPoolKernel = new Array(2);
maxPoolKernel[0] = 3, maxPoolKernel[1] = 3;
var maxPoolStride = new Array(2);
maxPoolStride[0] = 2, maxPoolStride[1] = 2;
x = MaxPool(x, maxPoolKernel, maxPoolStride, SAME);
var residualBlockChannel = new Array(2);


for(var i = 0 ; i < 4; ++i)
{
    residualBlockChannel[0] = channels[i];
    residualBlockChannel[1] = channels[i + 1];
    x = ResiduBlock(x, residualBlockChannel, strides[i], resNet18[i]);
}


var avePoolKernel = new Array(2);
avePoolKernel[0] = 7, avePoolKernel[1] = 7;

var avePoolStride = new Array(2);
avePoolStride[0] = 1, avePoolStride[1] = 1;

x = AvePool(x, avePoolKernel, avePoolStride, VALID);


var convChannel2nd = new Array(2);
convChannel2nd[0] = getVARPInfo(x, 1);
convChannel2nd[1] = numClass;


var convKernelSize2nd = new Array(2);
convKernelSize2nd[0] = 1, convKernelSize2nd[1] = 1;

var convStride2nd = new Array(2);
convStride2nd[0] = 1, convStride2nd[1] = 1;

var convDilate2nd = new Array(2);
convDilate2nd[0] = 1, convDilate2nd[1] = 1;

x = Conv(0.0, 0.0, x, convChannel2nd, convKernelSize2nd, VALID, convStride2nd, convDilate2nd, 1);


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
print("resNet_18 ：" + (sum_time / loop / 1000) + " ms");