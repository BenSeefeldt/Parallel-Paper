#include <omp.h>

main () {

int var1, var2, var3;

//Serial code

#pragma omp parallel private(var1, var2) shared(var3)
  {
  //Parallel Code
  }

//Serial Code Resumed

}
