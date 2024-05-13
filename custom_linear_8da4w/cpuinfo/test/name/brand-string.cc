#include <gtest/gtest.h>

#include <string>
#include <algorithm>
#include <cstring>


extern "C" uint32_t cpuinfo_x86_normalize_brand_string(
	const char* raw_name, char* normalized_name);


inline std::string normalize_brand_string(const char name[48]) {
	char normalized_name[48];
	cpuinfo_x86_normalize_brand_string(name, normalized_name);
	return std::string(normalized_name);
}

TEST(BRAND_STRING, intel) {
	EXPECT_EQ("",
		normalize_brand_string("Genuine Intel(R) CPU                  @ 2.33GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("                   Genuine Intel(R) CPU 3.00GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("                Genuine Intel(R) CPU  @ 2.60GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("Genuine Intel(R) CPU             0000 @ 1.73GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("         Genuine Intel(R) CPU         @ 728\0MHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("                   Genuine Intel(R) CPU 3.46GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("          Genuine Intel(R) CPU        @ 1.66GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("Genuine Intel(R) CPU             0000 @ 2.40GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("Genuine Intel(R) processor               800MHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("                Genuine Intel(R) CPU  @ 2.40GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("Genuine Intel(R) CPU            0     @ 1.60GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("Genuine Intel(R) CPU                  @ 2.66GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("Genuine Intel(R) CPU             000  @ 2.13GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("Genuine Intel(R) CPU           @ 0000 @ 2.67GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("Genuine Intel(R) CPU             000  @ 2>13GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("Genuine Intel(R) CPU           @ 0000 @ 1.87GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("Genuine Intel(R) CPU                  @ 2.13GHz\0"));
	EXPECT_EQ("",
		normalize_brand_string("Genuine Intel(R) CPU             000  @ 3.20GHz\0"));
	EXPECT_EQ("4000",
		normalize_brand_string("         Genuine Intel(R) CPU   4000  @ 1.00GHz\0"));
	EXPECT_EQ("5Y70",
		normalize_brand_string("Intel(R) Processor 5Y70 CPU @ 1.10GHz\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom 230",
		normalize_brand_string("         Intel(R) Atom(TM) CPU  230   @ 1.60GHz\0"));
	EXPECT_EQ("Atom 330",
		normalize_brand_string("         Intel(R) Atom(TM) CPU  330   @ 1.60GHz\0"));
	EXPECT_EQ("Atom C2750",
		normalize_brand_string("        Intel(R) Atom(TM) CPU  C2750  @ 2.40GHz\0"));
	EXPECT_EQ("Atom C2758",
		normalize_brand_string("        Intel(R) Atom(TM) CPU  C2758  @ 2.40GHz\0"));
	EXPECT_EQ("Atom D2500",
		normalize_brand_string("        Intel(R) Atom(TM) CPU D2500   @ 1.86GHz\0"));
	EXPECT_EQ("Atom D2700",
		normalize_brand_string("        Intel(R) Atom(TM) CPU D2700   @ 2.13GHz\0"));
	EXPECT_EQ("Atom D525",
		normalize_brand_string("         Intel(R) Atom(TM) CPU D525   @ 1.80GHz\0"));
	EXPECT_EQ("Atom N455",
		normalize_brand_string("         Intel(R) Atom(TM) CPU N455   @ 1.66GHz\0"));
	EXPECT_EQ("Atom S1260",
		normalize_brand_string("        Intel(R) Atom(TM) CPU S1260   @ 2.00GHz\0"));
	EXPECT_EQ("Atom Z2460",
		normalize_brand_string("         Intel(R) Atom(TM) CPU Z2460  @ 1.60GHz\0"));
	EXPECT_EQ("Atom Z2760",
		normalize_brand_string("         Intel(R) Atom(TM) CPU Z2760  @ 1.80GHz\0"));
	EXPECT_EQ("Atom Z3740",
		normalize_brand_string("        Intel(R) Atom(TM) CPU  Z3740  @ 1.33GHz\0"));
	EXPECT_EQ("Atom Z3745",
		normalize_brand_string("        Intel(R) Atom(TM) CPU  Z3745  @ 1.33GHz\0"));
	EXPECT_EQ("Atom Z670",
		normalize_brand_string("         Intel(R) Atom(TM) CPU Z670   @ 1.50GHz\0"));
	EXPECT_EQ("Atom x7-Z8700",
		normalize_brand_string("      Intel(R) Atom(TM) x7-Z8700  CPU @ 1.60GHz\0"));
	EXPECT_EQ("Celeron 1.70GHz",
		normalize_brand_string("                Intel(R) Celeron(R) CPU 1.70GHz\0"));
	EXPECT_EQ("Celeron 2.00GHz",
		normalize_brand_string("                Intel(R) Celeron(R) CPU 2.00GHz\0"));
	EXPECT_EQ("Celeron 2.53GHz",
		normalize_brand_string("                Intel(R) Celeron(R) CPU 2.53GHz\0"));
	EXPECT_EQ("Celeron 215",
		normalize_brand_string("Intel(R) Celeron(R) CPU          215  @ 1.33GHz\0"));
	EXPECT_EQ("Celeron 420",
		normalize_brand_string("Intel(R) Celeron(R) CPU          420  @ 1.60GHz\0"));
	EXPECT_EQ("Celeron 600MHz",
		normalize_brand_string("Intel(R) Celeron(R) processor            600MHz\0"));
	EXPECT_EQ("Celeron D 3.06GHz",
		normalize_brand_string("              Intel(R) Celeron(R) D CPU 3.06GHz\0"));
	EXPECT_EQ("Celeron G1610",
		normalize_brand_string("        Intel(R) Celeron(R) CPU G1610 @ 2.60GHz\0"));
	EXPECT_EQ("Celeron J1900",
		normalize_brand_string("      Intel(R) Celeron(R) CPU  J1900  @ 1.99GHz\0"));
	EXPECT_EQ("Celeron J3455",
		normalize_brand_string("Intel(R) Celeron(R) CPU J3455 @ 1.50GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Celeron M 1300MHz",
		normalize_brand_string("Intel(R) Celeron(R) M processor         1300MHz\0"));
	EXPECT_EQ("Celeron M 430",
		normalize_brand_string("Intel(R) Celeron(R) M CPU        430  @ 1.73GHz\0"));
	EXPECT_EQ("Celeron N3050",
		normalize_brand_string("      Intel(R) Celeron(R) CPU  N3050  @ 1.60GHz\0"));
	EXPECT_EQ("Celeron N3150",
		normalize_brand_string("      Intel(R) Celeron(R) CPU  N3150  @ 1.60GHz\0"));
	EXPECT_EQ("Core 2 6300",
		normalize_brand_string("Intel(R) Core(TM)2 CPU          6300  @ 1.86GHz\0"));
	EXPECT_EQ("Core 2 6700",
		normalize_brand_string("Intel(R) Core(TM)2 CPU          6700  @ 2.66GHz\0"));
	EXPECT_EQ("Core 2 Duo P8400",
		normalize_brand_string("Intel(R) Core(TM)2 Duo CPU     P8400  @ 2.26GHz\0"));
	EXPECT_EQ("Core 2 Duo T8300",
		normalize_brand_string("Intel(R) Core(TM)2 Duo CPU     T8300  @ 2.40GHz\0"));
	EXPECT_EQ("Core 2 Extreme X9650",
		normalize_brand_string("Intel(R) Core(TM)2 Extreme CPU X9650  @ 3.00GHz\0"));
	EXPECT_EQ("Core 2 Quad 2.66GHz",
		normalize_brand_string("Intel(R) Core(TM)2 Quad CPU           @ 2.66GHz\0"));
	EXPECT_EQ("Core 2 Quad Q6600",
		normalize_brand_string("Intel(R) Core(TM)2 Quad CPU    Q6600  @ 2.40GHz\0"));
	EXPECT_EQ("Core 2 Quad Q9300",
		normalize_brand_string("Intel(R) Core(TM)2 Quad  CPU   Q9300  @ 2.50GHz\0"));
	EXPECT_EQ("Core 2 T5600",
		normalize_brand_string("Intel(R) Core(TM)2 CPU         T5600  @ 1.83GHz\0"));
	EXPECT_EQ("Core 820Q",
		normalize_brand_string("Intel(R) Core(TM) CPU          Q 820  @ 1.73GHz\0"));
	EXPECT_EQ("Core i3 380M",
		normalize_brand_string("Intel(R) Core(TM) i3 CPU       M 380  @ 2.53GHz\0"));
	EXPECT_EQ("Core i5 480M",
		normalize_brand_string("Intel(R) Core(TM) i5 CPU       M 480  @ 2.67GHz\0"));
	EXPECT_EQ("Core i5 650",
		normalize_brand_string("Intel(R) Core(TM) i5 CPU         650  @ 3.20GHz\0"));
	EXPECT_EQ("Core i5 750",
		normalize_brand_string("Intel(R) Core(TM) i5 CPU         750  @ 2.67GHz\0"));
	EXPECT_EQ("Core i5-2400",
		normalize_brand_string("        Intel(R) Core(TM) i5-2400 CPU @ 3.10GHz\0"));
	EXPECT_EQ("Core i5-2450M",
		normalize_brand_string("       Intel(R) Core(TM) i5-2450M CPU @ 2.50GHz\0"));
	EXPECT_EQ("Core i5-5250U",
		normalize_brand_string("Intel(R) Core(TM) i5-5250U CPU @ 1.60GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i5-6400T",
		normalize_brand_string("Intel(R) Core(TM) i5-6400T CPU @ 2.20GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i5-7200U",
		normalize_brand_string("Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i7 720Q",
		normalize_brand_string("Intel(R) Core(TM) i7 CPU       Q 720  @ 1.60GHz\0"));
	EXPECT_EQ("Core i7 860",
		normalize_brand_string("Intel(R) Core(TM) i7 CPU         860  @ 2.80GHz\0"));
	EXPECT_EQ("Core i7 990X",
		normalize_brand_string("Intel(R) Core(TM) i7 CPU       X 990  @ 3.47GHz\0"));
	EXPECT_EQ("Core i7-2600",
		normalize_brand_string("        Intel(R) Core(TM) i7-2600 CPU @ 3.40GHz\0"));
	EXPECT_EQ("Core i7-2600K",
		normalize_brand_string("       Intel(R) Core(TM) i7-2600K CPU @ 3.40GHz\0"));
	EXPECT_EQ("Core i7-3770K",
		normalize_brand_string("       Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz\0"));
	EXPECT_EQ("Core i7-3960X",
		normalize_brand_string("       Intel(R) Core(TM) i7-3960X CPU @ 3.30GHz\0"));
	EXPECT_EQ("Core i7-4500U",
		normalize_brand_string("Intel(R) Core(TM) i7-4500U CPU @ 1.80GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i7-4770",
		normalize_brand_string("Intel(R) Core(TM) i7-4770 CPU @ 3.40GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i7-4770R",
		normalize_brand_string("Intel(R) Core(TM) i7-4770R CPU @ 3.20GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i7-4930K",
		normalize_brand_string("       Intel(R) Core(TM) i7-4930K CPU @ 3.40GHz\0"));
	EXPECT_EQ("Core i7-5775C",
		normalize_brand_string("Intel(R) Core(TM) i7-5775C CPU @ 3.30GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i7-5820K",
		normalize_brand_string("Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i7-6500U",
		normalize_brand_string("Intel(R) Core(TM) i7-6500U CPU @ 2.50GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i7-6800K",
		normalize_brand_string("Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i7-6850K",
		normalize_brand_string("Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i7-6950X",
		normalize_brand_string("Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i7-7700K",
		normalize_brand_string("Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i7-7800X",
		normalize_brand_string("Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i9-7900X",
		normalize_brand_string("Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core m3-6Y30",
		normalize_brand_string("Intel(R) Core(TM) m3-6Y30 CPU @ 0.90GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Pentium 4 1.60GHz",
		normalize_brand_string("              Intel(R) Pentium(R) 4 CPU 1.60GHz\0"));
	EXPECT_EQ("Pentium 4 2.40GHz",
		normalize_brand_string("              Intel(R) Pentium(R) 4 CPU 2.40GHz\0"));
	EXPECT_EQ("Pentium 4 2.80GHz",
		normalize_brand_string("              Intel(R) Pentium(R) 4 CPU 2.80GHz\0"));
	EXPECT_EQ("Pentium 4 3.00GHz",
		normalize_brand_string("              Intel(R) Pentium(R) 4 CPU 3.00GHz\0"));
	EXPECT_EQ("Pentium 4 3.20GHz",
		normalize_brand_string("              Intel(R) Pentium(R) 4 CPU 3.20GHz\0"));
	EXPECT_EQ("Pentium 4 3.46GHz",
		normalize_brand_string("              Intel(R) Pentium(R) 4 CPU 3.46GHz\0"));
	EXPECT_EQ("Pentium 4 3.73GHz",
		normalize_brand_string("              Intel(R) Pentium(R) 4 CPU 3.73GHz\0"));
	EXPECT_EQ("Pentium D 2.80GHz",
		normalize_brand_string("              Intel(R) Pentium(R) D CPU 2.80GHz\0"));
	EXPECT_EQ("Pentium Dual E2220",
		normalize_brand_string("Intel(R) Pentium(R) Dual  CPU  E2220  @ 2.40GHz\0"));
	EXPECT_EQ("Pentium G840",
		normalize_brand_string("         Intel(R) Pentium(R) CPU G840 @ 2.80GHz\0"));
	EXPECT_EQ("Pentium III 1266MHz",
		normalize_brand_string("Intel(R) Pentium(R) III CPU family      1266MHz\0"));
	EXPECT_EQ("Pentium M 1.60GHz",
		normalize_brand_string("        Intel(R) Pentium(R) M processor 1.60GHz\0"));
	EXPECT_EQ("Pentium M 2.00GHz",
		normalize_brand_string("Intel(R) Pentium(R) M CPU        000  @ 2.00GHz\0"));
	EXPECT_EQ("Pentium N4200",
		normalize_brand_string("Intel(R) Pentium(R) CPU N4200 @ 1.10GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Pentium T4200",
		normalize_brand_string("Pentium(R) Dual-Core CPU       T4200  @ 2.00GHz\0"));
	EXPECT_EQ("Pentium T4500",
		normalize_brand_string("Pentium(R) Dual-Core CPU       T4500  @ 2.30GHz\0"));
	EXPECT_EQ("Xeon 2.66GHz",
		normalize_brand_string("                  Intel(R) Xeon(TM) CPU 2.66GHz\0"));
	EXPECT_EQ("Xeon 2.80GHz",
		normalize_brand_string("                  Intel(R) Xeon(TM) CPU 2.80GHz\0"));
	EXPECT_EQ("Xeon 3.06GHz",
		normalize_brand_string("                  Intel(R) Xeon(TM) CPU 3.06GHz\0"));
	EXPECT_EQ("Xeon 3.20GHz",
		normalize_brand_string("                  Intel(R) Xeon(TM) CPU 3.20GHz\0"));
	EXPECT_EQ("Xeon 3.40GHz",
		normalize_brand_string("                  Intel(R) Xeon(TM) CPU 3.40GHz\0"));
	EXPECT_EQ("Xeon D-1540",
		normalize_brand_string("Intel(R) Xeon(R) CPU D-1540 @ 2.00GHz\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Xeon E3-1230 v2",
		normalize_brand_string("      Intel(R) Xeon(R) CPU E3-1230 V2 @ 3.30GHz\0"));
	EXPECT_EQ("Xeon E3-1245 v3",
		normalize_brand_string("Intel(R) Xeon(R) CPU E3-1245 v3 @ 3.40GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Xeon E5-2660 v3",
		normalize_brand_string("Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Xeon E5-2696 v4",
		normalize_brand_string("Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Xeon E5-2697 v2",
		normalize_brand_string("      Intel(R) Xeon(R) CPU E5-2697 v2 @ 2.70GHz\0"));
	EXPECT_EQ("Xeon E5-2697 v4",
		normalize_brand_string("Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Xeon E5-2699 v3",
		normalize_brand_string("Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Xeon E5462",
		normalize_brand_string("Intel(R) Xeon(R) CPU           E5462  @ 2.80GHz\0"));
	EXPECT_EQ("Xeon E7-4870",
		normalize_brand_string("       Intel(R) Xeon(R) CPU E7- 4870  @ 2.40GHz\0"));
	EXPECT_EQ("Xeon E7-8870",
		normalize_brand_string("       Intel(R) Xeon(R) CPU E7- 8870  @ 2.40GHz\0"));
	EXPECT_EQ("Xeon E7450",
		normalize_brand_string("Intel(R) Xeon(R) CPU           E7450  @ 2.40GHz\0"));
	EXPECT_EQ("Xeon E7520",
		normalize_brand_string("Intel(R) Xeon(R) CPU           E7520  @ 1.87GHz\0"));
	EXPECT_EQ("Xeon Gold 6130",
		normalize_brand_string("Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Xeon Gold 6154",
		normalize_brand_string("Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Xeon L5320",
		normalize_brand_string("Intel(R) Xeon(R) CPU           L5320  @ 1.86GHz\0"));
	EXPECT_EQ("Xeon Phi 7210",
		normalize_brand_string("Intel(R) Xeon Phi(TM) CPU 7210 @ 1.30GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Xeon Platinum 8180",
		normalize_brand_string("Intel(R) Xeon(R) Platinum 8180 CPU @ 2.50GHz\0\0\0\0"));
	EXPECT_EQ("Xeon X3210",
		normalize_brand_string("Intel(R) Xeon(R) CPU           X3210  @ 2.13GHz\0"));
	EXPECT_EQ("Xeon X3323",
		normalize_brand_string("Intel(R) Xeon(R) CPU           X3323  @ 2.50GHz\0"));
	EXPECT_EQ("Xeon X5667",
		normalize_brand_string("Intel(R) Xeon(R) CPU           X5667  @ 3.07GHz\0"));
	EXPECT_EQ("Xeon X6550",
		normalize_brand_string("Intel(R) Xeon(R) CPU           X6550  @ 2.00GHz\0"));
}

TEST(BRAND_STRING, intel_android) {
	EXPECT_EQ("Atom N2600",
		normalize_brand_string("Intel(R) Atom(TM) CPU N2600   @ 1.60GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Sofia3GR",
		normalize_brand_string("Intel(R) Atom(TM) CPU Sofia3GR\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z2420",
		normalize_brand_string("Intel(R) Atom(TM) CPU Z2420  @ 1.20GHz\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z2460",
		normalize_brand_string("Intel(R) Atom(TM) CPU Z2460  @ 1.60GHz\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z2480",
		normalize_brand_string("Intel(R) Atom(TM) CPU Z2480  @ 2.00GHz\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z2520",
		normalize_brand_string("Intel(R) Atom(TM) CPU Z2520  @ 1.20GHz\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z2560",
		normalize_brand_string("Intel(R) Atom(TM) CPU Z2560  @ 1.60GHz\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z2580",
		normalize_brand_string("Intel(R) Atom(TM) CPU Z2580  @ 2.00GHz\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3460",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3460  @ 1.06GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3480",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3480  @ 1.33GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3530",
		normalize_brand_string("Intel(R) Atom(TM) CPU Z3530 @ 1.33GHz\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3530",
		normalize_brand_string("Intel(R) Atom(TM) CPU Z3530  @ 1.33GHz\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3530",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3530  @ 1.33GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3560",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3560  @ 1.00GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3560",
		normalize_brand_string("Intel(R) Atom(TM) CPU Z3560  @ 1.83GHz\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3560",
		normalize_brand_string("Intel(R) Atom(TM) CPU Z3560 @ 1.83GHz\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3580",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3580  @ 1.33GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3580",
		normalize_brand_string("Intel(R) Atom(TM) CPU Z3580  @ 2.33GHz\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3590",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3590  @ 1.33GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3735D",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3735D @ 1.33GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3735E",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3735E @ 1.33GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3735F",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3735F @ 1.33GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3735G",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3735G @ 1.33GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3736F",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3736F @ 1.33GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3736G",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3736G @ 1.33GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom Z3745",
		normalize_brand_string("Intel(R) Atom(TM) CPU  Z3745  @ 1.33GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom x5-Z8300",
		normalize_brand_string("Intel(R) Atom(TM) x5-Z8300  CPU @ 1.44GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom x5-Z8350",
		normalize_brand_string("Intel(R) Atom(TM) x5-Z8350  CPU @ 1.44GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom x5-Z8500",
		normalize_brand_string("Intel(R) Atom(TM) x5-Z8500  CPU @ 1.44GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom x5-Z8550",
		normalize_brand_string("Intel(R) Atom(TM) x5-Z8550  CPU @ 1.44GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom x7-Z8700",
		normalize_brand_string("Intel(R) Atom(TM) x7-Z8700  CPU @ 1.60GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Atom x7-Z8750",
		normalize_brand_string("Intel(R) Atom(TM) x7-Z8750  CPU @ 1.60GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Celeron 847",
		normalize_brand_string("Intel(R) Celeron(R) CPU 847 @ 1.10GHz\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Celeron N3060",
		normalize_brand_string("Intel(R) Celeron(R) CPU  N3060  @ 1.60GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Celeron N3160",
		normalize_brand_string("Intel(R) Celeron(R) CPU  N3160  @ 1.60GHz\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i3-2100",
		normalize_brand_string("Intel(R) Core(TM) i3-2100 CPU @ 3.10GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i3-2120",
		normalize_brand_string("Intel(R) Core(TM) i3-2120 CPU @ 3.30GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i3-3110M",
		normalize_brand_string("Intel(R) Core(TM) i3-3110M CPU @ 2.40GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i3-3217U",
		normalize_brand_string("Intel(R) Core(TM) i3-3217U CPU @ 1.80GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i3-3220",
		normalize_brand_string("Intel(R) Core(TM) i3-3220 CPU @ 3.30GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i3-4005U",
		normalize_brand_string("Intel(R) Core(TM) i3-4005U CPU @ 1.70GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i3-5005U",
		normalize_brand_string("Intel(R) Core(TM) i3-5005U CPU @ 2.00GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i5-2467M",
		normalize_brand_string("Intel(R) Core(TM) i5-2467M CPU @ 1.60GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i5-3210M",
		normalize_brand_string("Intel(R) Core(TM) i5-3210M CPU @ 2.50GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i5-3230M",
		normalize_brand_string("Intel(R) Core(TM) i5-3230M CPU @ 2.60GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i5-3470",
		normalize_brand_string("Intel(R) Core(TM) i5-3470 CPU @ 3.20GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i5-4210U",
		normalize_brand_string("Intel(R) Core(TM) i5-4210U CPU @ 1.70GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i5-4460",
		normalize_brand_string("Intel(R) Core(TM) i5-4460  CPU @ 3.20GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i5-5200U",
		normalize_brand_string("Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i5-6200U",
		normalize_brand_string("Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i5-6400",
		normalize_brand_string("Intel(R) Core(TM) i5-6400 CPU @ 2.70GHz\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Core i7-4790",
		normalize_brand_string("Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz\0\0\0\0\0\0\0\0\0"));
}

TEST(BRAND_STRING, amd) {
	EXPECT_EQ("",
		normalize_brand_string("AMD Processor model unknown\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("",
		normalize_brand_string("AMD Engineering Sample\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("A10-4600M",
		normalize_brand_string("AMD A10-4600M APU with Radeon(tm) HD Graphics  \0"));
	EXPECT_EQ("A10-5800K",
		normalize_brand_string("AMD A10-5800K APU with Radeon(tm) HD Graphics  \0"));
	EXPECT_EQ("A10-6800K",
		normalize_brand_string("AMD A10-6800K APU with Radeon(tm) HD Graphics  \0"));
	EXPECT_EQ("A10-7850K",
		normalize_brand_string("AMD A10-7850K APU with Radeon(TM) R7 Graphics  \0"));
	EXPECT_EQ("A12-9700P",
		normalize_brand_string("AMD A12-9700P RADEON R7, 10 COMPUTE CORES 4C+6G\0"));
	EXPECT_EQ("A12-9800",
		normalize_brand_string("AMD A12-9800 RADEON R7, 12 COMPUTE CORES 4C+8G \0"));
	EXPECT_EQ("A4-5000",
		normalize_brand_string("AMD A4-5000 APU with Radeon(TM) HD Graphics    \0"));
	EXPECT_EQ("A6-6310",
		normalize_brand_string("AMD A6-6310 APU with AMD Radeon R4 Graphics    \0"));
	EXPECT_EQ("A8-3850",
		normalize_brand_string("AMD A8-3850 APU with Radeon(tm) HD Graphics\0\0\0\0\0"));
	EXPECT_EQ("A8-6410",
		normalize_brand_string("AMD A8-6410 APU with AMD Radeon R5 Graphics    \0"));
	EXPECT_EQ("A8-7670K",
		normalize_brand_string("AMD A8-7670K Radeon R7, 10 Compute Cores 4C+6G \0"));
	EXPECT_EQ("A9-9410",
		normalize_brand_string("AMD A9-9410 RADEON R5, 5 COMPUTE CORES 2C+3G   \0"));
	EXPECT_EQ("Athlon",
		normalize_brand_string("AMD Athlon(tm) Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Athlon 5350",
		normalize_brand_string("AMD Athlon(tm) 5350 APU with Radeon(tm) R3     \0"));
	EXPECT_EQ("Athlon 64 2800+",
		normalize_brand_string("AMD Athlon(tm) 64 Processor 2800+\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Athlon 64 3200+",
		normalize_brand_string("AMD Athlon(tm) 64 Processor 3200+\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Athlon 64 X2 3800+",
		normalize_brand_string("AMD Athlon(tm) 64 X2 Dual Core Processor 3800+\0\0"));
	EXPECT_EQ("Athlon 64 X2 4000+",
		normalize_brand_string("AMD Athlon(tm) 64 X2 Dual Core Processor 4000+\0\0"));
	EXPECT_EQ("Athlon 64 X2 6000+",
		normalize_brand_string("AMD Athlon(tm) 64 X2 Dual Core Processor 6000+\0\0"));
	EXPECT_EQ("Athlon 64 X2 6400+",
		normalize_brand_string("AMD Athlon(tm) 64 X2 Dual Core Processor 6400+\0\0"));
	EXPECT_EQ("Athlon 7750",
		normalize_brand_string("AMD Athlon(tm) 7750 Dual-Core Processor\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Athlon II X2 245",
		normalize_brand_string("AMD Athlon(tm) II X2 245 Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Athlon II X4 620",
		normalize_brand_string("AMD Athlon(tm) II X4 620 Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Athlon XP",
		normalize_brand_string("Athlon XP (Palomin?00+\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Athlon XP 2200+",
		normalize_brand_string("AMD Athlon(tm) XP 2200+\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Athlon XP 3200+",
		normalize_brand_string("AMD Athlon(tm) XP 3200+\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("C-50",
		normalize_brand_string("AMD C-50 Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Duron",
		normalize_brand_string("AMD Duron(tm) processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("E-350",
		normalize_brand_string("AMD E-350 Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("E-450",
		normalize_brand_string("AMD E-450 APU with Radeon(tm) HD Graphics\0\0\0\0\0\0\0"));
	EXPECT_EQ("E2-3000M",
		normalize_brand_string("AMD E2-3000M APU with Radeon(tm) HD Graphics\0\0\0\0"));
	EXPECT_EQ("FX-6100",
		normalize_brand_string("AMD FX(tm)-6100 Six-Core Processor             \0"));
	EXPECT_EQ("FX-8150",
		normalize_brand_string("AMD FX(tm)-8150 Eight-Core Processor           \0"));
	EXPECT_EQ("FX-8350",
		normalize_brand_string("AMD FX(tm)-8350 Eight-Core Processor           \0"));
	EXPECT_EQ("FX-8800P",
		normalize_brand_string("AMD FX-8800P Radeon R7, 12 Compute Cores 4C+8G \0"));
	EXPECT_EQ("G-T56N",
		normalize_brand_string("AMD G-T56N Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("GX-212JC",
		normalize_brand_string("AMD GX-212JC SOC with Radeon(TM) R2E Graphics  \0"));
	EXPECT_EQ("Geode",
		normalize_brand_string("Geode(TM) Integrated Processor by AMD PCS\0\0\0\0\0\0\0"));
	EXPECT_EQ("K5",
		normalize_brand_string("AMD-K5(tm) Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("K6",
		normalize_brand_string("AMD-K6tm w/ multimedia extensions\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("K6 3D",
		normalize_brand_string("AMD-K6(tm) 3D processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("K6 3D+",
		normalize_brand_string("AMD-K6(tm) 3D+ Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("K6-III",
		normalize_brand_string("AMD-K6(tm)-III Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("K7",
		normalize_brand_string("AMD-K7(tm) Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Opteron 2210 HE",
		normalize_brand_string("Dual-Core AMD Opteron(tm) Processor 2210 HE\0\0\0\0\0"));
	EXPECT_EQ("Opteron 2344 HE",
		normalize_brand_string("Quad-Core AMD Opteron(tm) Processor 2344 HE\0\0\0\0\0"));
	EXPECT_EQ("Opteron 2347 HE",
		normalize_brand_string("Quad-Core AMD Opteron(tm) Processor 2347 HE\0\0\0\0\0"));
	EXPECT_EQ("Opteron 2378",
		normalize_brand_string("Quad-Core AMD Opteron(tm) Processor 2378\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Opteron 240 HE",
		normalize_brand_string("AMD Opteron(tm) Processor 240 HE\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Opteron 2431",
		normalize_brand_string("Six-Core AMD Opteron(tm) Processor 2431\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Opteron 248",
		normalize_brand_string("AMD Opteron(tm) Processor 248\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Opteron 4176 HE",
		normalize_brand_string("AMD Opteron(tm) Processor 4176 HE\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Opteron 6180 SE",
		normalize_brand_string("AMD Opteron(tm) Processor 6180 SE\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Opteron 6274",
		normalize_brand_string("AMD Opteron(TM) Processor 6274                 \0"));
	EXPECT_EQ("Opteron 8220 SE",
		normalize_brand_string("Dual-Core AMD Opteron(tm) Processor 8220 SE\0\0\0\0\0"));
	EXPECT_EQ("Phenom 9500",
		normalize_brand_string("AMD Phenom(tm) 9500 Quad-Core Processor\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Phenom II 42 TWKR Black Edition",
		normalize_brand_string("AMD Phenom(tm) II 42 TWKR Black Edition\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Phenom II X2 550",
		normalize_brand_string("AMD Phenom(tm) II X2 550 Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Phenom II X4 940",
		normalize_brand_string("AMD Phenom(tm) II X4 940 Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Phenom II X4 955",
		normalize_brand_string("AMD Phenom(tm) II X4 955 Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Phenom II X4 965",
		normalize_brand_string("AMD Phenom(tm) II X4 965 Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Phenom II X6 1055T",
		normalize_brand_string("AMD Phenom(tm) II X6 1055T Processor\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Ryzen 5 1500X",
		normalize_brand_string("AMD Ryzen 5 1500X Quad-Core Processor          \0"));
	EXPECT_EQ("Ryzen 7 1700X",
		normalize_brand_string("AMD Ryzen 7 1700X Eight-Core Processor         \0"));
	EXPECT_EQ("Ryzen 7 1800X",
		normalize_brand_string("AMD Ryzen 7 1800X Eight-Core Processor         \0"));
	EXPECT_EQ("Ryzen Threadripper 1950X",
		normalize_brand_string("AMD Ryzen Threadripper 1950X 16-Core Processor \0"));
	EXPECT_EQ("Sempron 140",
		normalize_brand_string("AMD Sempron(tm) 140 Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Sempron 2600+",
		normalize_brand_string("AMD Sempron(tm) Processor 2600+\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Sempron 2800+",
		normalize_brand_string("AMD Sempron(tm) Processor 2800+\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Sempron 3000+",
		normalize_brand_string("AMD Sempron(tm) Processor 3000+\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Turion RM-70",
		normalize_brand_string("AMD Turion Dual-Core RM-70\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Turion X2 Ultra ZM-82",
		normalize_brand_string("AMD Turion(tm) X2 Ultra Dual-Core Mobile ZM-82\0\0"));
}

TEST(BRAND_STRING, via) {
	EXPECT_EQ("C3 Ezra",
		normalize_brand_string("VIA C3 Ezra\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("C7-M 1200MHz",
		normalize_brand_string("                     VIA C7-M Processor 1200MHz\0"));
	EXPECT_EQ("CNA 1800MHz",
		normalize_brand_string("                      VIA CNA processor 1800MHz "));
	EXPECT_EQ("CNA 2667MHz",
		normalize_brand_string("                      VIA CNA processor 2667MHz "));
	EXPECT_EQ("Eden X4 C4250",
		normalize_brand_string("                      VIA Eden X4 C4250@1.2+GHz\0"));
	EXPECT_EQ("Esther 1500MHz",
		normalize_brand_string("                   VIA Esther processor 1500MHz\0"));
	EXPECT_EQ("Ezra",
		normalize_brand_string("VIA Ezra\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("L4700",
		normalize_brand_string("                  VIA QuadCore L4700 @ 1.2+ GHz\0"));
	EXPECT_EQ("Nano 1800MHz",
		normalize_brand_string("               VIA Nano processor      @1800MHz\0"));
	EXPECT_EQ("Nano L2200",
		normalize_brand_string("               VIA Nano processor L2200@1600MHz\0"));
	EXPECT_EQ("Nano L3050",
		normalize_brand_string("                         VIA Nano L3050@1800MHz\0"));
	EXPECT_EQ("Nehemiah",
		normalize_brand_string("VIA Nehemiah\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Samuel",
		normalize_brand_string("VIA Samuel\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Samuel 2",
		normalize_brand_string("VIA Samuel 2\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("Samuel M",
		normalize_brand_string("VIA Samuel\0\0M\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
}

TEST(BRAND_STRING, transmeta) {
	EXPECT_EQ("Crusoe TM5800",
		normalize_brand_string("Transmeta(tm) Crusoe(tm) Processor TM5800\0\0\0\0\0\0\0"));
	EXPECT_EQ("Efficeon TM8000",
		normalize_brand_string("Transmeta Efficeon(tm) Processor TM8000\0\0\0\0\0\0\0\0\0"));
}

TEST(BRAND_STRING, other) {
	EXPECT_EQ("",
		normalize_brand_string("\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("",
		normalize_brand_string("Quad-Core Processor (up to 1.4GHz)             \0"));
	EXPECT_EQ("Geode",
		normalize_brand_string("Geode(TM) Integrated Processor by National Semi\0"));
	EXPECT_EQ("MediaGX",
		normalize_brand_string("Cyrix MediaGXtm MMXtm Enhanced\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
	EXPECT_EQ("WinChip 2-3D",
		normalize_brand_string("IDT WinChip 2-3D\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"));
}
