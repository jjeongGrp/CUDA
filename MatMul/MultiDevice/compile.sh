#nvcc -arch=sm_52 -fmad=false MultiDeviceMatMul.cu -o MultiDeviceMatMul
nvcc -arch=sm_52 -fmad=false MatMul_MultiDevice.cu -o MatMul_MultiDevice
