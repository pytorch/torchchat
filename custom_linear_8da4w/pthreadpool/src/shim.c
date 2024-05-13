/* Standard C headers */
#include <stddef.h>

/* Public library header */
#include <pthreadpool.h>

/* Internal library headers */
#include "threadpool-utils.h"


struct pthreadpool {
};

static const struct pthreadpool static_pthreadpool = { };


struct pthreadpool* pthreadpool_create(size_t threads_count) {
	if (threads_count <= 1) {
		return (struct pthreadpool*) &static_pthreadpool;
	}

	return NULL;
}

size_t pthreadpool_get_threads_count(struct pthreadpool* threadpool) {
	return 1;
}

void pthreadpool_parallelize_1d(
	struct pthreadpool* threadpool,
	pthreadpool_task_1d_t task,
	void* argument,
	size_t range,
	uint32_t flags)
{
	for (size_t i = 0; i < range; i++) {
		task(argument, i);
	}
}

void pthreadpool_parallelize_1d_with_thread(
	struct pthreadpool* threadpool,
	pthreadpool_task_1d_with_thread_t task,
	void* argument,
	size_t range,
	uint32_t flags)
{
	for (size_t i = 0; i < range; i++) {
		task(argument, 0, i);
	}
}

void pthreadpool_parallelize_1d_with_uarch(
	pthreadpool_t threadpool,
	pthreadpool_task_1d_with_id_t task,
	void* argument,
	uint32_t default_uarch_index,
	uint32_t max_uarch_index,
	size_t range,
	uint32_t flags)
{
	for (size_t i = 0; i < range; i++) {
		task(argument, default_uarch_index, i);
	}
}

void pthreadpool_parallelize_1d_tile_1d(
	pthreadpool_t threadpool,
	pthreadpool_task_1d_tile_1d_t task,
	void* argument,
	size_t range,
	size_t tile,
	uint32_t flags)
{
	for (size_t i = 0; i < range; i += tile) {
		task(argument, i, min(range - i, tile));
	}
}

void pthreadpool_parallelize_2d(
	struct pthreadpool* threadpool,
	pthreadpool_task_2d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			task(argument, i, j);
		}
	}
}

void pthreadpool_parallelize_2d_with_thread(
	struct pthreadpool* threadpool,
	pthreadpool_task_2d_with_thread_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			task(argument, 0, i, j);
		}
	}
}

void pthreadpool_parallelize_2d_tile_1d(
	pthreadpool_t threadpool,
	pthreadpool_task_2d_tile_1d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t tile_j,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j += tile_j) {
			task(argument, i, j, min(range_j - j, tile_j));
		}
	}
}

void pthreadpool_parallelize_2d_tile_1d_with_uarch(
	pthreadpool_t threadpool,
	pthreadpool_task_2d_tile_1d_with_id_t task,
	void* argument,
	uint32_t default_uarch_index,
	uint32_t max_uarch_index,
	size_t range_i,
	size_t range_j,
	size_t tile_j,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j += tile_j) {
			task(argument, default_uarch_index, i, j, min(range_j - j, tile_j));
		}
	}
}

void pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
	pthreadpool_t threadpool,
	pthreadpool_task_2d_tile_1d_with_id_with_thread_t task,
	void* argument,
	uint32_t default_uarch_index,
	uint32_t max_uarch_index,
	size_t range_i,
	size_t range_j,
	size_t tile_j,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j += tile_j) {
			task(argument, default_uarch_index, 0, i, j, min(range_j - j, tile_j));
		}
	}
}

void pthreadpool_parallelize_2d_tile_2d(
	pthreadpool_t threadpool,
	pthreadpool_task_2d_tile_2d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t tile_i,
	size_t tile_j,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i += tile_i) {
		for (size_t j = 0; j < range_j; j += tile_j) {
			task(argument, i, j, min(range_i - i, tile_i), min(range_j - j, tile_j));
		}
	}
}

void pthreadpool_parallelize_2d_tile_2d_with_uarch(
	pthreadpool_t threadpool,
	pthreadpool_task_2d_tile_2d_with_id_t task,
	void* argument,
	uint32_t default_uarch_index,
	uint32_t max_uarch_index,
	size_t range_i,
	size_t range_j,
	size_t tile_i,
	size_t tile_j,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i += tile_i) {
		for (size_t j = 0; j < range_j; j += tile_j) {
			task(argument, default_uarch_index, i, j,
				min(range_i - i, tile_i), min(range_j - j, tile_j));
		}
	}
}

void pthreadpool_parallelize_3d(
	pthreadpool_t threadpool,
	pthreadpool_task_3d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k++) {
				task(argument, i, j, k);
			}
		}
	}
}

void pthreadpool_parallelize_3d_tile_1d(
	pthreadpool_t threadpool,
	pthreadpool_task_3d_tile_1d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t tile_k,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k += tile_k) {
				task(argument, i, j, k, min(range_k - k, tile_k));
			}
		}
	}
}

void pthreadpool_parallelize_3d_tile_1d_with_thread(
	pthreadpool_t threadpool,
	pthreadpool_task_3d_tile_1d_with_thread_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t tile_k,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k += tile_k) {
				task(argument, 0, i, j, k, min(range_k - k, tile_k));
			}
		}
	}
}

void pthreadpool_parallelize_3d_tile_1d_with_uarch(
	pthreadpool_t threadpool,
	pthreadpool_task_3d_tile_1d_with_id_t task,
	void* argument,
	uint32_t default_uarch_index,
	uint32_t max_uarch_index,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t tile_k,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k += tile_k) {
				task(argument, default_uarch_index, i, j, k, min(range_k - k, tile_k));
			}
		}
	}
}

void pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
	pthreadpool_t threadpool,
	pthreadpool_task_3d_tile_1d_with_id_with_thread_t task,
	void* argument,
	uint32_t default_uarch_index,
	uint32_t max_uarch_index,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t tile_k,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k += tile_k) {
				task(argument, default_uarch_index, 0, i, j, k, min(range_k - k, tile_k));
			}
		}
	}
}

void pthreadpool_parallelize_3d_tile_2d(
	pthreadpool_t threadpool,
	pthreadpool_task_3d_tile_2d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t tile_j,
	size_t tile_k,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j += tile_j) {
			for (size_t k = 0; k < range_k; k += tile_k) {
				task(argument, i, j, k,
					min(range_j - j, tile_j), min(range_k - k, tile_k));
			}
		}
	}
}

void pthreadpool_parallelize_3d_tile_2d_with_uarch(
	pthreadpool_t threadpool,
	pthreadpool_task_3d_tile_2d_with_id_t task,
	void* argument,
	uint32_t default_uarch_index,
	uint32_t max_uarch_index,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t tile_j,
	size_t tile_k,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j += tile_j) {
			for (size_t k = 0; k < range_k; k += tile_k) {
				task(argument, default_uarch_index, i, j, k,
					min(range_j - j, tile_j), min(range_k - k, tile_k));
			}
		}
	}
}

void pthreadpool_parallelize_4d(
	pthreadpool_t threadpool,
	pthreadpool_task_4d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t range_l,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k++) {
				for (size_t l = 0; l < range_l; l++) {
					task(argument, i, j, k, l);
				}
			}
		}
	}
}

void pthreadpool_parallelize_4d_tile_1d(
	pthreadpool_t threadpool,
	pthreadpool_task_4d_tile_1d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t range_l,
	size_t tile_l,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k++) {
				for (size_t l = 0; l < range_l; l += tile_l) {
					task(argument, i, j, k, l, min(range_l - l, tile_l));
				}
			}
		}
	}
}

void pthreadpool_parallelize_4d_tile_2d(
	pthreadpool_t threadpool,
	pthreadpool_task_4d_tile_2d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t range_l,
	size_t tile_k,
	size_t tile_l,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k += tile_k) {
				for (size_t l = 0; l < range_l; l += tile_l) {
					task(argument, i, j, k, l,
						min(range_k - k, tile_k), min(range_l - l, tile_l));
				}
			}
		}
	}
}

void pthreadpool_parallelize_4d_tile_2d_with_uarch(
	pthreadpool_t threadpool,
	pthreadpool_task_4d_tile_2d_with_id_t task,
	void* argument,
	uint32_t default_uarch_index,
	uint32_t max_uarch_index,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t range_l,
	size_t tile_k,
	size_t tile_l,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k += tile_k) {
				for (size_t l = 0; l < range_l; l += tile_l) {
					task(argument, default_uarch_index, i, j, k, l,
						min(range_k - k, tile_k), min(range_l - l, tile_l));
				}
			}
		}
	}
}

void pthreadpool_parallelize_5d(
	pthreadpool_t threadpool,
	pthreadpool_task_5d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t range_l,
	size_t range_m,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k++) {
				for (size_t l = 0; l < range_l; l++) {
					for (size_t m = 0; m < range_m; m++) {
						task(argument, i, j, k, l, m);
					}
				}
			}
		}
	}
}

void pthreadpool_parallelize_5d_tile_1d(
	pthreadpool_t threadpool,
	pthreadpool_task_5d_tile_1d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t range_l,
	size_t range_m,
	size_t tile_m,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k++) {
				for (size_t l = 0; l < range_l; l++) {
					for (size_t m = 0; m < range_m; m += tile_m) {
						task(argument, i, j, k, l, m, min(range_m - m, tile_m));
					}
				}
			}
		}
	}
}

void pthreadpool_parallelize_5d_tile_2d(
	pthreadpool_t threadpool,
	pthreadpool_task_5d_tile_2d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t range_l,
	size_t range_m,
	size_t tile_l,
	size_t tile_m,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k++) {
				for (size_t l = 0; l < range_l; l += tile_l) {
					for (size_t m = 0; m < range_m; m += tile_m) {
						task(argument, i, j, k, l, m,
							min(range_l - l, tile_l), min(range_m - m, tile_m));
					}
				}
			}
		}
	}
}

void pthreadpool_parallelize_6d(
	pthreadpool_t threadpool,
	pthreadpool_task_6d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t range_l,
	size_t range_m,
	size_t range_n,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k++) {
				for (size_t l = 0; l < range_l; l++) {
					for (size_t m = 0; m < range_m; m++) {
						for (size_t n = 0; n < range_n; n++) {
							task(argument, i, j, k, l, m, n);
						}
					}
				}
			}
		}
	}
}

void pthreadpool_parallelize_6d_tile_1d(
	pthreadpool_t threadpool,
	pthreadpool_task_6d_tile_1d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t range_l,
	size_t range_m,
	size_t range_n,
	size_t tile_n,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k++) {
				for (size_t l = 0; l < range_l; l++) {
					for (size_t m = 0; m < range_m; m++) {
						for (size_t n = 0; n < range_n; n += tile_n) {
							task(argument, i, j, k, l, m, n, min(range_n - n, tile_n));
						}
					}
				}
			}
		}
	}
}

void pthreadpool_parallelize_6d_tile_2d(
	pthreadpool_t threadpool,
	pthreadpool_task_6d_tile_2d_t task,
	void* argument,
	size_t range_i,
	size_t range_j,
	size_t range_k,
	size_t range_l,
	size_t range_m,
	size_t range_n,
	size_t tile_m,
	size_t tile_n,
	uint32_t flags)
{
	for (size_t i = 0; i < range_i; i++) {
		for (size_t j = 0; j < range_j; j++) {
			for (size_t k = 0; k < range_k; k++) {
				for (size_t l = 0; l < range_l; l++) {
					for (size_t m = 0; m < range_m; m += tile_m) {
						for (size_t n = 0; n < range_n; n += tile_n) {
							task(argument, i, j, k, l, m, n,
								min(range_m - m, tile_m), min(range_n - n, tile_n));
						}
					}
				}
			}
		}
	}
}

void pthreadpool_destroy(struct pthreadpool* threadpool) {
}
