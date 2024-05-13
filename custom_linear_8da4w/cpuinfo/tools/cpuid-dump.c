#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>

#include <x86/cpuid.h>

static void print_cpuid(struct cpuid_regs regs, uint32_t eax) {
	printf("CPUID %08"PRIX32": %08"PRIX32"-%08"PRIX32"-%08"PRIX32"-%08"PRIX32"\n",
		eax, regs.eax, regs.ebx, regs.ecx, regs.edx);
}

static void print_cpuidex(struct cpuid_regs regs, uint32_t eax, uint32_t ecx) {
	printf("CPUID %08"PRIX32": %08"PRIX32"-%08"PRIX32"-%08"PRIX32"-%08"PRIX32" [SL %02"PRIX32"]\n",
		eax, regs.eax, regs.ebx, regs.ecx, regs.edx, ecx);
}

static void print_cpuid_vendor(struct cpuid_regs regs, uint32_t eax) {
	if (regs.ebx | regs.ecx | regs.edx) {
		char vendor_id[12];
		memcpy(&vendor_id[0], &regs.ebx, sizeof(regs.ebx));
		memcpy(&vendor_id[4], &regs.edx, sizeof(regs.edx));
		memcpy(&vendor_id[8], &regs.ecx, sizeof(regs.ecx));
		printf("CPUID %08"PRIX32": %08"PRIX32"-%08"PRIX32"-%08"PRIX32"-%08"PRIX32" [%.12s]\n",
			eax, regs.eax, regs.ebx, regs.ecx, regs.edx, vendor_id);
	} else {
		print_cpuid(regs, eax);
	}
}

static void print_cpuid_brand_string(struct cpuid_regs regs, uint32_t eax) {
	char brand_string[16];
	memcpy(&brand_string[0], &regs.eax, sizeof(regs.eax));
	memcpy(&brand_string[4], &regs.ebx, sizeof(regs.ebx));
	memcpy(&brand_string[8], &regs.ecx, sizeof(regs.ecx));
	memcpy(&brand_string[12], &regs.edx, sizeof(regs.edx));
	printf("CPUID %08"PRIX32": %08"PRIX32"-%08"PRIX32"-%08"PRIX32"-%08"PRIX32" [%.16s]\n",
		eax, regs.eax, regs.ebx, regs.ecx, regs.edx, brand_string);
}

int main(int argc, char** argv) {
	const uint32_t max_base_index = cpuid(0).eax;
	uint32_t max_structured_index = 0, max_trace_index = 0, max_socid_index = 0;
	bool has_sgx = false;
	for (uint32_t eax = 0; eax <= max_base_index; eax++) {
		switch (eax) {
			case UINT32_C(0x00000000):
				print_cpuid_vendor(cpuid(eax), eax);
				break;
			case UINT32_C(0x00000004):
				for (uint32_t ecx = 0; ; ecx++) {
					const struct cpuid_regs regs = cpuidex(eax, ecx);
					if ((regs.eax & UINT32_C(0x1F)) == 0) {
						break;
					}
					print_cpuidex(regs, eax, ecx);
				}
				break;
			case UINT32_C(0x00000007):
				for (uint32_t ecx = 0; ecx <= max_structured_index; ecx++) {
					const struct cpuid_regs regs = cpuidex(eax, ecx);
					if (ecx == 0) {
						max_structured_index = regs.eax;
						has_sgx = !!(regs.ebx & UINT32_C(0x00000004));
					}
					print_cpuidex(regs, eax, ecx);
				}
				break;
			case UINT32_C(0x0000000B):
				for (uint32_t ecx = 0; ; ecx++) {
					const struct cpuid_regs regs = cpuidex(eax, ecx);
					if ((regs.ecx & UINT32_C(0x0000FF00)) == 0) {
						break;
					}
					print_cpuidex(regs, eax, ecx);
				}
				break;
			case UINT32_C(0x00000012):
				if (has_sgx) {
					for (uint32_t ecx = 0; ; ecx++) {
						const struct cpuid_regs regs = cpuidex(eax, ecx);
						if (ecx >= 2 && (regs.eax & UINT32_C(0x0000000F)) == 0) {
							break;
						}
						print_cpuidex(regs, eax, ecx);
					}
				}
				break;
			case UINT32_C(0x00000014):
				for (uint32_t ecx = 0; ecx <= max_trace_index; ecx++) {
					const struct cpuid_regs regs = cpuidex(eax, ecx);
					if (ecx == 0) {
						max_trace_index = regs.eax;
					}
					print_cpuidex(regs, eax, ecx);
				}
				break;
			case UINT32_C(0x00000017):
				for (uint32_t ecx = 0; ecx <= max_socid_index; ecx++) {
					const struct cpuid_regs regs = cpuidex(eax, ecx);
					if (ecx == 0) {
						max_socid_index = regs.eax;
					}
					print_cpuidex(regs, eax, ecx);
				}
				break;
			default:
				print_cpuid(cpuidex(eax, 0), eax);
				break;
		}
	}

	const uint32_t max_extended_index = cpuid(UINT32_C(0x80000000)).eax;
	for (uint32_t eax = UINT32_C(0x80000000); eax <= max_extended_index; eax++) {
		switch (eax) {
			case UINT32_C(0x80000000):
				print_cpuid_vendor(cpuid(eax), eax);
				break;
			case UINT32_C(0x80000002):
			case UINT32_C(0x80000003):
			case UINT32_C(0x80000004):
				print_cpuid_brand_string(cpuid(eax), eax);
				break;
			default:
				print_cpuid(cpuidex(eax, 0), eax);
		}
	}
}
