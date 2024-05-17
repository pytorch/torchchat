#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <iostream>


int main() {
  std::cout << "Caffe2 threadcount: " << caffe2::pthreadpool()->get_thread_count() << std::endl;
  std::cout << "Torch threadcount: " << torch::get_num_threads() << std::endl;
}
