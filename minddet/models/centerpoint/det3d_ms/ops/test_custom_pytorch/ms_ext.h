#include <string.h>
#include <torch/extension.h> // 头文件引用部分

// 将 mindspore kernel 的 inputs/outputs 转换为 pytorch 的 tensor
std::vector<at::Tensor> get_torch_tensors(int nparam, void** params, int* ndims, int64_t** shapes, const char** dtypes, c10::Device device) ;

// 将入参没有输出的pytorch 算子的计算结果拷贝到kernel的输出内存
void output_memcpy(void* output, const torch::Tensor &t) ;
