#include <stdlib.h>
#include <stdio.h>

#include <sys/auxv.h>
#include <errno.h>
#include <dlfcn.h>

#include <cpuinfo.h>


typedef unsigned long (*getauxval_function_t)(unsigned long);

int main(int argc, char** argv) {
	void* libc = dlopen("libc.so", RTLD_NOW);
	if (libc == NULL) {
		fprintf(stderr, "Error: failed to load libc.so: %s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	getauxval_function_t getauxval = (getauxval_function_t) dlsym(libc, "getauxval");
	if (getauxval == NULL) {
		fprintf(stderr, "Error: failed to locate getauxval in libc.so: %s", dlerror());
		exit(EXIT_FAILURE);
	}

	printf("AT_HWCAP = 0x%08lX\n", getauxval(AT_HWCAP));
	#if CPUINFO_ARCH_ARM
		printf("AT_HWCAP2 = 0x%08lX\n", getauxval(AT_HWCAP2));
	#endif

	return 0;
}
