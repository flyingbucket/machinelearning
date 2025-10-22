#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp parallel
  {
    int tid = omp_get_thread_num(); // 获取当前线程号
    printf("Hello from thread %d\n", tid);
  }
  return 0;
}
