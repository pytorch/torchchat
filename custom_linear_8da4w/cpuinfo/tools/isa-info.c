#include <stdio.h>
#include <stdlib.h>

#include <cpuinfo.h>

int main(int argc, char** argv) {
	if (!cpuinfo_initialize()) {
		fprintf(stderr, "failed to initialize CPU information\n");
		exit(EXIT_FAILURE);
	}

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64

	printf("Scalar instructions:\n");
#if CPUINFO_ARCH_X86
		printf("\tx87 FPU: %s\n", cpuinfo_has_x86_fpu() ? "yes" : "no");
		printf("\tCMOV: %s\n", cpuinfo_has_x86_cmov() ? "yes" : "no");
#endif
		printf("\tLAHF/SAHF: %s\n", cpuinfo_has_x86_lahf_sahf() ? "yes" : "no");
		printf("\tLZCNT: %s\n", cpuinfo_has_x86_lzcnt() ? "yes" : "no");
		printf("\tPOPCNT: %s\n", cpuinfo_has_x86_popcnt() ? "yes" : "no");
		printf("\tTBM: %s\n", cpuinfo_has_x86_tbm() ? "yes" : "no");
		printf("\tBMI: %s\n", cpuinfo_has_x86_bmi() ? "yes" : "no");
		printf("\tBMI2: %s\n", cpuinfo_has_x86_bmi2() ? "yes" : "no");
		printf("\tADCX/ADOX: %s\n", cpuinfo_has_x86_adx() ? "yes" : "no");


	printf("Memory instructions:\n");
		printf("\tMOVBE: %s\n", cpuinfo_has_x86_movbe() ? "yes" : "no");
		printf("\tPREFETCH: %s\n", cpuinfo_has_x86_prefetch() ? "yes" : "no");
		printf("\tPREFETCHW: %s\n", cpuinfo_has_x86_prefetchw() ? "yes" : "no");
		printf("\tPREFETCHWT1: %s\n", cpuinfo_has_x86_prefetchwt1() ? "yes" : "no");
		printf("\tCLZERO: %s\n", cpuinfo_has_x86_clzero() ? "yes" : "no");


	printf("SIMD extensions:\n");
		printf("\tMMX: %s\n", cpuinfo_has_x86_mmx() ? "yes" : "no");
		printf("\tMMX+: %s\n", cpuinfo_has_x86_mmx_plus() ? "yes" : "no");
		printf("\t3dnow!: %s\n", cpuinfo_has_x86_3dnow() ? "yes" : "no");
		printf("\t3dnow!+: %s\n", cpuinfo_has_x86_3dnow_plus() ? "yes" : "no");
		printf("\t3dnow! Geode: %s\n", cpuinfo_has_x86_3dnow_geode() ? "yes" : "no");
		printf("\tDAZ: %s\n", cpuinfo_has_x86_daz() ? "yes" : "no");
		printf("\tSSE: %s\n", cpuinfo_has_x86_sse() ? "yes" : "no");
		printf("\tSSE2: %s\n", cpuinfo_has_x86_sse2() ? "yes" : "no");
		printf("\tSSE3: %s\n", cpuinfo_has_x86_sse3() ? "yes" : "no");
		printf("\tSSSE3: %s\n", cpuinfo_has_x86_ssse3() ? "yes" : "no");
		printf("\tSSE4.1: %s\n", cpuinfo_has_x86_sse4_1() ? "yes" : "no");
		printf("\tSSE4.2: %s\n", cpuinfo_has_x86_sse4_2() ? "yes" : "no");
		printf("\tSSE4a: %s\n", cpuinfo_has_x86_sse4a() ? "yes" : "no");
		printf("\tMisaligned SSE: %s\n", cpuinfo_has_x86_misaligned_sse() ? "yes" : "no");
		printf("\tAVX: %s\n", cpuinfo_has_x86_avx() ? "yes" : "no");
		printf("\tFMA3: %s\n", cpuinfo_has_x86_fma3() ? "yes" : "no");
		printf("\tFMA4: %s\n", cpuinfo_has_x86_fma4() ? "yes" : "no");
		printf("\tXOP: %s\n", cpuinfo_has_x86_xop() ? "yes" : "no");
		printf("\tF16C: %s\n", cpuinfo_has_x86_f16c() ? "yes" : "no");
		printf("\tAVX2: %s\n", cpuinfo_has_x86_avx2() ? "yes" : "no");
		printf("\tAVX512F: %s\n", cpuinfo_has_x86_avx512f() ? "yes" : "no");
		printf("\tAVX512PF: %s\n", cpuinfo_has_x86_avx512pf() ? "yes" : "no");
		printf("\tAVX512ER: %s\n", cpuinfo_has_x86_avx512er() ? "yes" : "no");
		printf("\tAVX512CD: %s\n", cpuinfo_has_x86_avx512cd() ? "yes" : "no");
		printf("\tAVX512DQ: %s\n", cpuinfo_has_x86_avx512dq() ? "yes" : "no");
		printf("\tAVX512BW: %s\n", cpuinfo_has_x86_avx512bw() ? "yes" : "no");
		printf("\tAVX512VL: %s\n", cpuinfo_has_x86_avx512vl() ? "yes" : "no");
		printf("\tAVX512IFMA: %s\n", cpuinfo_has_x86_avx512ifma() ? "yes" : "no");
		printf("\tAVX512VBMI: %s\n", cpuinfo_has_x86_avx512vbmi() ? "yes" : "no");
		printf("\tAVX512VBMI2: %s\n", cpuinfo_has_x86_avx512vbmi2() ? "yes" : "no");
		printf("\tAVX512BITALG: %s\n", cpuinfo_has_x86_avx512bitalg() ? "yes" : "no");
		printf("\tAVX512VPOPCNTDQ: %s\n", cpuinfo_has_x86_avx512vpopcntdq() ? "yes" : "no");
		printf("\tAVX512VNNI: %s\n", cpuinfo_has_x86_avx512vnni() ? "yes" : "no");
		printf("\tAVX512BF16: %s\n", cpuinfo_has_x86_avx512bf16() ? "yes" : "no");
		printf("\tAVX512FP16: %s\n", cpuinfo_has_x86_avx512fp16() ? "yes" : "no");
		printf("\tAVX512VP2INTERSECT: %s\n", cpuinfo_has_x86_avx512vp2intersect() ? "yes" : "no");
		printf("\tAVX512_4VNNIW: %s\n", cpuinfo_has_x86_avx512_4vnniw() ? "yes" : "no");
		printf("\tAVX512_4FMAPS: %s\n", cpuinfo_has_x86_avx512_4fmaps() ? "yes" : "no");
                printf("\tAVXVNNI: %s\n", cpuinfo_has_x86_avxvnni() ? "yes" : "no");


	printf("Multi-threading extensions:\n");
		printf("\tMONITOR/MWAIT: %s\n", cpuinfo_has_x86_mwait() ? "yes" : "no");
		printf("\tMONITORX/MWAITX: %s\n", cpuinfo_has_x86_mwaitx() ? "yes" : "no");
#if CPUINFO_ARCH_X86
		printf("\tCMPXCHG8B: %s\n", cpuinfo_has_x86_cmpxchg8b() ? "yes" : "no");
#endif
		printf("\tCMPXCHG16B: %s\n", cpuinfo_has_x86_cmpxchg16b() ? "yes" : "no");
		printf("\tHLE: %s\n", cpuinfo_has_x86_hle() ? "yes" : "no");
		printf("\tRTM: %s\n", cpuinfo_has_x86_rtm() ? "yes" : "no");
		printf("\tXTEST: %s\n", cpuinfo_has_x86_xtest() ? "yes" : "no");
		printf("\tRDPID: %s\n", cpuinfo_has_x86_rdpid() ? "yes" : "no");


	printf("Cryptography extensions:\n");
		printf("\tAES: %s\n", cpuinfo_has_x86_aes() ? "yes" : "no");
		printf("\tVAES: %s\n", cpuinfo_has_x86_vaes() ? "yes" : "no");
		printf("\tPCLMULQDQ: %s\n", cpuinfo_has_x86_pclmulqdq() ? "yes" : "no");
		printf("\tVPCLMULQDQ: %s\n", cpuinfo_has_x86_vpclmulqdq() ? "yes" : "no");
		printf("\tGFNI: %s\n", cpuinfo_has_x86_gfni() ? "yes" : "no");
		printf("\tRDRAND: %s\n", cpuinfo_has_x86_rdrand() ? "yes" : "no");
		printf("\tRDSEED: %s\n", cpuinfo_has_x86_rdseed() ? "yes" : "no");
		printf("\tSHA: %s\n", cpuinfo_has_x86_sha() ? "yes" : "no");


	printf("Profiling instructions:\n");
#if CPUINFO_ARCH_X86
		printf("\tRDTSC: %s\n", cpuinfo_has_x86_rdtsc() ? "yes" : "no");
#endif
		printf("\tRDTSCP: %s\n", cpuinfo_has_x86_rdtscp() ? "yes" : "no");
		printf("\tMPX: %s\n", cpuinfo_has_x86_mpx() ? "yes" : "no");


	printf("System instructions:\n");
		printf("\tCLWB: %s\n", cpuinfo_has_x86_clwb() ? "yes" : "no");
		printf("\tFXSAVE/FXSTOR: %s\n", cpuinfo_has_x86_fxsave() ? "yes" : "no");
		printf("\tXSAVE/XSTOR: %s\n", cpuinfo_has_x86_xsave() ? "yes" : "no");

#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */

#if CPUINFO_ARCH_ARM
	printf("Instruction sets:\n");
		printf("\tThumb: %s\n", cpuinfo_has_arm_thumb() ? "yes" : "no");
		printf("\tThumb 2: %s\n", cpuinfo_has_arm_thumb2() ? "yes" : "no");
		printf("\tARMv5E: %s\n", cpuinfo_has_arm_v5e() ? "yes" : "no");
		printf("\tARMv6: %s\n", cpuinfo_has_arm_v6() ? "yes" : "no");
		printf("\tARMv6-K: %s\n", cpuinfo_has_arm_v6k() ? "yes" : "no");
		printf("\tARMv7: %s\n", cpuinfo_has_arm_v7() ? "yes" : "no");
		printf("\tARMv7 MP: %s\n", cpuinfo_has_arm_v7mp() ? "yes" : "no");
		printf("\tARMv8: %s\n", cpuinfo_has_arm_v8() ? "yes" : "no");
		printf("\tIDIV: %s\n", cpuinfo_has_arm_idiv() ? "yes" : "no");

	printf("Floating-Point support:\n");
		printf("\tVFPv2: %s\n", cpuinfo_has_arm_vfpv2() ? "yes" : "no");
		printf("\tVFPv3: %s\n", cpuinfo_has_arm_vfpv3() ? "yes" : "no");
		printf("\tVFPv3+D32: %s\n", cpuinfo_has_arm_vfpv3_d32() ? "yes" : "no");
		printf("\tVFPv3+FP16: %s\n", cpuinfo_has_arm_vfpv3_fp16() ? "yes" : "no");
		printf("\tVFPv3+FP16+D32: %s\n", cpuinfo_has_arm_vfpv3_fp16_d32() ? "yes" : "no");
		printf("\tVFPv4: %s\n", cpuinfo_has_arm_vfpv4() ? "yes" : "no");
		printf("\tVFPv4+D32: %s\n", cpuinfo_has_arm_vfpv4_d32() ? "yes" : "no");
		printf("\tVJCVT: %s\n", cpuinfo_has_arm_jscvt() ? "yes" : "no");

	printf("SIMD extensions:\n");
		printf("\tWMMX: %s\n", cpuinfo_has_arm_wmmx() ? "yes" : "no");
		printf("\tWMMX 2: %s\n", cpuinfo_has_arm_wmmx2() ? "yes" : "no");
		printf("\tNEON: %s\n", cpuinfo_has_arm_neon() ? "yes" : "no");
		printf("\tNEON-FP16: %s\n", cpuinfo_has_arm_neon_fp16() ? "yes" : "no");
		printf("\tNEON-FMA: %s\n", cpuinfo_has_arm_neon_fma() ? "yes" : "no");
		printf("\tNEON VQRDMLAH/VQRDMLSH: %s\n", cpuinfo_has_arm_neon_rdm() ? "yes" : "no");
		printf("\tNEON FP16 arithmetics: %s\n", cpuinfo_has_arm_neon_fp16_arith() ? "yes" : "no");
		printf("\tNEON complex: %s\n", cpuinfo_has_arm_fcma() ? "yes" : "no");
		printf("\tNEON VSDOT/VUDOT: %s\n", cpuinfo_has_arm_neon_dot() ? "yes" : "no");
		printf("\tNEON VFMLAL/VFMLSL: %s\n", cpuinfo_has_arm_fhm() ? "yes" : "no");

	printf("Cryptography extensions:\n");
		printf("\tAES: %s\n", cpuinfo_has_arm_aes() ? "yes" : "no");
		printf("\tSHA1: %s\n", cpuinfo_has_arm_sha1() ? "yes" : "no");
		printf("\tSHA2: %s\n", cpuinfo_has_arm_sha2() ? "yes" : "no");
		printf("\tPMULL: %s\n", cpuinfo_has_arm_pmull() ? "yes" : "no");
		printf("\tCRC32: %s\n", cpuinfo_has_arm_crc32() ? "yes" : "no");
#endif /* CPUINFO_ARCH_ARM */
#if CPUINFO_ARCH_ARM64
	printf("Instruction sets:\n");
		printf("\tARM v8.1 atomics: %s\n", cpuinfo_has_arm_atomics() ? "yes" : "no");
		printf("\tARM v8.1 SQRDMLxH: %s\n", cpuinfo_has_arm_neon_rdm() ? "yes" : "no");
		printf("\tARM v8.2 FP16 arithmetics: %s\n", cpuinfo_has_arm_fp16_arith() ? "yes" : "no");
		printf("\tARM v8.2 FHM: %s\n", cpuinfo_has_arm_fhm() ? "yes" : "no");
		printf("\tARM v8.2 BF16: %s\n", cpuinfo_has_arm_bf16() ? "yes" : "no");
		printf("\tARM v8.2 Int8 dot product: %s\n", cpuinfo_has_arm_neon_dot() ? "yes" : "no");
		printf("\tARM v8.2 Int8 matrix multiplication: %s\n", cpuinfo_has_arm_i8mm() ? "yes" : "no");
		printf("\tARM v8.3 JS conversion: %s\n", cpuinfo_has_arm_jscvt() ? "yes" : "no");
		printf("\tARM v8.3 complex: %s\n", cpuinfo_has_arm_fcma() ? "yes" : "no");

	printf("SIMD extensions:\n");
		printf("\tARM SVE: %s\n", cpuinfo_has_arm_sve() ? "yes" : "no");
		printf("\tARM SVE 2: %s\n", cpuinfo_has_arm_sve2() ? "yes" : "no");

	printf("Cryptography extensions:\n");
		printf("\tAES: %s\n", cpuinfo_has_arm_aes() ? "yes" : "no");
		printf("\tSHA1: %s\n", cpuinfo_has_arm_sha1() ? "yes" : "no");
		printf("\tSHA2: %s\n", cpuinfo_has_arm_sha2() ? "yes" : "no");
		printf("\tPMULL: %s\n", cpuinfo_has_arm_pmull() ? "yes" : "no");
		printf("\tCRC32: %s\n", cpuinfo_has_arm_crc32() ? "yes" : "no");
#endif

}
